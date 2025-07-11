#!/usr/bin/env python
# ───────
#  TARGETED HEARING PARSER – version 25 (memory-efficient, no precheck)
#      • removed precheck process for speed
#      • added memory clearing in batch loop
#      • parse all XMLs (not just substantive ones)
#      • maintains rerun capability
# ───────
import json, re, itertools, gc, lxml.etree as ET, pandas as pd
from pathlib import Path
from datetime import date
import signal  #need this to help rerun the script when TDM studio kicks me off

# ── config ─────────────
MIN_WORDS  = 500
EARLIEST   = date(1873, 1, 1)
BATCH_SIZE = 25

BASE_DIR   = Path("/home/ec2-user/SageMaker/data")
CORPUS_DIR = BASE_DIR / "Congress_Hearings"
OUT_DIR    = BASE_DIR / "output_files";  OUT_DIR.mkdir(exist_ok=True)

SELECTED   = OUT_DIR / "selected_xmls.json"
OUT_CLEAN  = OUT_DIR / "selected_hearings_clean.csv"
OUT_DROP   = OUT_DIR / "selected_hearings_discarded.csv"
OUT_NO_INT = OUT_DIR / "selected_no_intro.csv"

dash = "─"*110
log  = print

# ──  build roster ──────
class TimeoutError(Exception):
    pass
def _time_limit(sec: int = 60):
    """Context-manager: raise TimeoutError after *sec* seconds inside the block"""
    def _handler(signum, frame):
        raise TimeoutError
    return signal.signal(signal.SIGALRM, _handler), signal.alarm(sec)

def load(p): return json.load(p.open())

def flat(d, p="", sep="."):
    out={}
    for k,v in d.items():
        key=f"{p}{sep}{k}" if p else k
        out.update(flat(v,key,sep) if isinstance(v,dict) else {key:v})
    return out

def canon(r):
    if r.get("id",{}).get("wikipedia"):
        return r["id"]["wikipedia"].title()
    nm=r.get("name",{})
    return f"{nm.get('first','').title()} {nm.get('last','').title()}".strip()

rec_cur  = load(CORPUS_DIR/"legislators-current.json")
rec_hist = load(CORPUS_DIR/"legislators-historical.json")
rec_exec = load(CORPUS_DIR/"executive2.json")

rows=[]
for rec in itertools.chain(rec_cur,rec_hist,rec_exec):
    name=canon(rec)
    base=flat(rec); base["Name"]=name
    spans=rec.get("terms",[])+rec.get("positions",[])
    if not spans:
        spans=[{"start":"0000-00-00","end":"9999-12-31"}]
    for s in spans:
        rows.append({**base,"start":s["start"],"end":s["end"]})

term_df=pd.DataFrame(rows)[["Name","start","end"]]
term_df=term_df[term_df["Name"].str.strip()!=""]
meta_df=term_df.drop_duplicates("Name")
log(f"▶ ROSTER  |  {len(term_df):,} term rows, {meta_df['Name'].nunique():,} unique people\n")

# ──  regex library ──────────────────────────────────────╮
INTRO_W   = 2_000
BULLET    = r"[*•●]"

# fuzzy (±1-char) variants for Mr./Mrs./Ms. – final period **required**
#    ONE optional letter may slip between "M" and the core
#    any number of spaces may precede the period
FUZZY_MR  = r"M\w?r\s*\."      # Mr., Mir. , M r. , Mkr.  …
FUZZY_MRS = r"M\w?rs\s*\."     # Mrs., Mras. , M rs. …
FUZZY_MS  = r"M\w?s\s*\."      # Ms., M s. , Mbs. …
FUZZY_DR  = r"D\w?r\s*\."       # Dr., D r. , Der. …   

TITLE_WORD = (f"{FUZZY_MR}|{FUZZY_MRS}|{FUZZY_MS}|{FUZZY_DR}|Miss|Mx|"
              "Prof|Hon|Judge|Justice|Rev|Pastor|Senator|Representative|General|Colonel|President|Lieutenant|Captain|Corporal|The Honorable|Professor|Pastor|Admiral|Major|Sergeant|"
              "Congressman|Congresswoman|Delegate|Chairman|Chairwoman|Chairperson")

ROLE_TOK   = r"CHAIR(?:MAN|WOMAN|PERSON)?|VICE\s+CHAIR|RANKING\s+MEMBER"
FUZZY_CHAIRMAN = r"""
    (?:
        C[Hh][Aa][Ii1l][Rr]{1,2}[Mm][Aa][Nn]   |   # good variants like 'Chairman'
        CHAIRMAN?                              |   # missing final N
        CHAIRMA[NM]                            |   # common typos
        CHA?I?R?M?A?N?                         |   # soft fallback
        [Cc][Hh][Aa1i][Rr][Mm][Aa][Nn]         |   # transpositions, leetspeak
        CH[A-Z]{4,7}                           |   # 5+ capital letters starting CH
        CH[A-Z]{1,3}MAN                        |   # short rubbish in middle
        CHAIRMA[A-Z]                           |   # extra final letters
        C[A-Z]{5,8}                            # rubbish near beginning
    )
"""
CR_SPEAKER_RE = re.compile(rf"""
    (?:\s{{0,8}}|<bullet>|{BULLET})\s*
    (?P<n>
        (?:
            (?:(?:{TITLE_WORD})\s+)
            [A-Z][A-Za-z'\-\.]*(?:\s+[A-Z][A-Za-z'\-\.]*)*
            (?:\s+of\s+[A-Z][A-Za-z\s]+)?
        )
      |
        (?:The\s+)?(?:VICE\s+)?PRESIDENT(?:\s+pro\s+tempore)?
      |
        (?:The\s+(?:ACTING\s+|Acting\s+)?SPEAKER(?:\s+pro\s+tempore)?)
      |
        (?:The\s+(?:ACTING\s+|Acting\s+)?{FUZZY_CHAIRMAN})   # ← uses new fuzzy version
      |
        (?:The\s+PRESIDING\s+OFFICER)
      |
        (?:The\s+CLERK)
      |
        (?:The\s+CHIEF\s+JUSTICE)
      |
        (?:{FUZZY_MR}\s+Counsel\s+[A-Z]+)
    )
    (?:\s*\([^)]+\))?
    \s*[:.]\s+
""", re.I | re.X)

#lots of regex taken from other people on GitHub e.g.,
#https://github.com/unitedstates/congressional-record
VERDATE_ART_RE = re.compile(r"VerDate[\s\S]*?(?=\b[a-z]{5,}\w*)", re.M)
ROLLCALL_RE    = re.compile(r"\[Roll(?:call)?(?:\s+Vote)?\s+No\.\s+\d+.*?\]", re.I)
RECORDER_START_RE = re.compile(
    r"^\s+(?:The (?:assistant )?legislative clerk read as follows|"
    r"The nomination considered and confirmed is as follows|"
    r"The nomination was confirmed|"
    r"The (?:assistant )?legislative clerk|"
    r"There being no objection,|"
    r"The resolution .*?was agreed to\.|"
    r"The preamble was agreed to\.|"
    r"The resolution .*?reads as follows|"
    r"The assistant editor .*?proceeded to call the roll|"
    r"The bill clerk (?:proceeded to )?call(?:ed)? the roll\.?|"
    r"The motion was agreed to\.|"
    r"The question was taken(?:;|\.)|"
    r"The yeas and nays (?:were ordered|resulted.*?, as follows:)|"
    r"The result was announced.*?, as follows:|"
    r"Amendment No\. \d+.*?is as follows:|"
    r"The text of the.*?is as follows|"
    r"amended(?: to read)? as follows:|"
    r"The material (?:previously )?referred to (?:by.*?)?is as follows:|"
    r"There was no objection|"
    r"The amendment.*?was agreed to|"
    r"The motion to table was .*|"
    r"The following bills and joint resolutions were introduced.*|"
    r"The vote was taken by electronic device|"
    r"A recorded vote was ordered"
    r").*", re.I | re.M)

STMT_PAT        = re.compile(r"STATEMENTS? OF\s+[A-Z][A-Z'\-]+(?:\s+[A-Z][A-Z'\-]+){0,2}")
BRACKET_RE      = re.compile(r"\[\s*[A-Za-z]+\.*\s*\]", re.I)
TITLE_TRAIL_RE  = re.compile(rf"(?:{TITLE_WORD})(?:\s+[A-Z][A-Za-z'\-]+)?\s*$", re.I)
ORPHAN_TITLE_RE = re.compile(rf"\s+(?:{TITLE_WORD})\.\s*$", re.I)
CUE_LAST_RE     = re.compile(r"\b([A-Z][A-Z'\-]+)\s*[:\.]\s+")
ARTIFACT_RE = re.compile(
    r"""
    \b
    (?:
        [A-Z]{2,}                # first ALLCAP word
        (?:\s+(?:OF|AT|TO|IN|AND|THE|FOR|WITH|ON|BY))?  # optional preposition
        \s+
    ){2,}                       # repeat 2+ times = total 3+
    [A-Z]{2,}[,.]?\s*           # ending ALLCAP word with optional punctuation
    """,
    re.VERBOSE
)



DECOR           = r"[<>\[\]()*_~\s]*"
INTRO_ROLE_PAT  = re.compile(
    rf"{DECOR}(?:THE\s+)?(?P<role>{ROLE_TOK}){DECOR}"
    rf"(?P<full>(?:[A-Z][\w.''\-]+\s+?)+?)\s*,", re.I)

FALLBACK_CHAIR_PAT = re.compile(
    rf"{FUZZY_CHAIRMAN}\s+(?:,?\s+(?:{FUZZY_MR}|{FUZZY_MRS}|{FUZZY_MS})\s+)?"
    rf"(?P<surname>[A-Z][A-Z'\-]+)[:\.]", re.I | re.X)

PLACEHOLDER="__UNKNOWN_CHAIR__"

#list of state names for name recognition (senator names come after states)
states_long=['alabama','alaska','arizona','arkansas','california','colorado','connecticut','delaware',
             'district of columbia','florida','georgia','hawaii','idaho','illinois','indiana','iowa',
             'kansas','kentucky','louisiana','maine','maryland','massachusetts','michigan','minnesota',
             'mississippi','missouri','montana','nebraska','nevada','new hampshire','new jersey',
             'new mexico','new york','north carolina','north dakota','ohio','oklahoma','oregon',
             'pennsylvania','rhode island','south carolina','south dakota','tennessee','texas','utah',
             'vermont','virginia','washington','west virginia','wisconsin','wyoming']

STATE_NAME_FALLBACK_RE = re.compile(
    rf"\b({'|'.join(states_long)})\b,?\s+([A-Z][A-Z'.\-]+(?:\s+[A-Z][A-Z'.\-]+){{0,3}})",
    re.I)

# ─  helpers ────────────────────────────────────────────╮
def build_intro_name_regex(names):
    mid = r"(?:\s+(?:[A-Z][\w'\-.]*|[A-Z]\.))*"
    suf = r"(?:\s+(?:Jr\.?|Sr\.?|II|III|IV|V))?"
    pieces = []
    for full in names:
        parts = full.split()
        if len(parts) < 2: continue
        pieces.append(rf"{re.escape(parts[0])}{mid}\s+{re.escape(parts[-1])}{suf}")
    return re.compile(r"\b(" + "|".join(pieces) + r")\b", re.I)

# — regex helpers (put near the other helpers) ————————————
UWS = r"[ \u00A0\t\r\n\f\v\u2000-\u200B\u202F\u205F\u3000]+"   # +NBSP
CAPS_TOKEN = r"[A-Z]{2,}[A-Za-z ']*[,.·]?"                    # relaxed ALL-CAPS word
# ----------------------------------------------------------------

CAPS_BLOCK_RE = re.compile(
    rf"""
        (?:\s*\d+[,\s]+)?              #  this group is OPTIONAL
        (?:{CAPS_TOKEN}{UWS}?[,]*){{3,}}
        (?:{UWS}?[,\.]?\s*\d+)?        #  this group is OPTIONAL
    """, re.X
)
def clean_segment(seg: str) -> str:
    # 1️⃣ dashes surrounded by spaces/em-dashes → single space
    seg = re.sub(r"\s*[\-–—]{1,2}\s*", " ", seg)
    # 2️⃣ dashes *inside* a token  "po-licy" → "policy", "re-move" → "remove"
    seg = re.sub(r"(\w)[\-–—]{1,2}(\w)", r"\1\2", seg)
    # 3️⃣ boiler-plate CAPS headings
    seg = CAPS_BLOCK_RE.sub(" ", seg)
    # 4️⃣ collapse double spaces
    return re.sub(r"\s{2,}", " ", seg).strip()

def scrub_artifacts(txt:str)->str:
    txt=VERDATE_ART_RE.sub("",txt)
    txt=ROLLCALL_RE.sub("",txt)
    txt=RECORDER_START_RE.sub("",txt)
    return txt

word_pat=re.compile(r"[A-Za-z0-9]")

# ── STREAM set-up & RERUN guard ─────────────────────
def _seen(csv_path, col="File"):
    """Return set of filenames already present in a CSV (empty if file absent)."""
    return (set(pd.read_csv(csv_path, usecols=[col])[col])
            if csv_path.exists() and csv_path.stat().st_size else set())

already_done = _seen(OUT_CLEAN) | _seen(OUT_DROP) | _seen(OUT_NO_INT)
log(f"▶ RERUN  |  {len(already_done):,} XMLs already parsed – will be skipped\n")

all_xmls = sorted(p for p in CORPUS_DIR.glob("*.xml")      # add '**/*.xml' if nested
                  if p.name not in already_done)
log(f"▶ STREAM |  {len(all_xmls):,} XMLs left to process\n")

# ── XML parsing function ──────────────────────────
def parse_xml_file(fp: Path) -> dict:
    """
    Parse a single XML file and extract date, title, and full text.
    Returns a dict with the parsed data or None if parsing fails.
    """
    try:
        total_words, dtx, ttl = 0, "0000-00-00", ""
        text_chunks = []
        
        for _, el in ET.iterparse(str(fp), events=("end",),
                                  tag=("Text", "NumericDate", "Title"),
                                  huge_tree=True):
            if el.tag == "Text" and el.text:
                total_words += len(el.text.split())
                text_chunks.append(el.text)
            elif el.tag == "NumericDate" and el.text:
                dtx = el.text
            elif el.tag == "Title" and el.text and not ttl:
                ttl = el.text.strip()
            el.clear()
            
        # Basic validity check
        try:
            y, m, d = map(int, dtx.split("-"))
            date_ok = date(y, m, d) >= EARLIEST
        except Exception:
            date_ok = False
            
        if not date_ok:
            return None
            
        return {
            "file_path": fp,
            "date": dtx,
            "title": ttl,
            "text": " ".join(text_chunks),
            "word_count": total_words,
        }
        
    except ET.XMLSyntaxError as e:
        log(f"[BAD XML] {fp.name}: {e} – skipped")
        return None
    except Exception as e:
        log(f"[ERROR] {fp.name}: {e} – skipped")
        return None

# ── main parser with diagnostics ────────────────────
def clean_hearing_from_data(fd):
    text, dtx, title = fd["text"], fd["date"], fd["title"]
    active = term_df[(term_df["start"] <= dtx) & (term_df["end"] >= dtx)]
    if active.empty:
        active = term_df

    last2name = {n.split()[-1].upper(): n for n in active["Name"]}
    last2name[PLACEHOLDER] = "Unknown Chair"

    intro_name_re = build_intro_name_regex(active["Name"])

    text = scrub_artifacts(text)
    if (c := text.find("[Whereupon,")) != -1:
        text = text[:c]
    while (m := STMT_PAT.search(text)):
        nxt = CR_SPEAKER_RE.search(text, m.end())
        text = text[:m.start()] + text[(nxt.start() if nxt else len(text)):]

    intro = " ".join(text.split()[:INTRO_W])

    # ── 6-a  roster hits
    intro_hits = [m.group(1).title() for m in intro_name_re.finditer(intro)]

    # ── 6-b  fallback "STATE → ALL-CAPS" hits
    for m in STATE_NAME_FALLBACK_RE.finditer(intro):  #another way of getting names
        caps = m.group(2)
        if caps.isupper():
            intro_hits.append(" ".join(p.title() for p in caps.split()))

    keep = {n.split()[-1].upper() for n in intro_hits}

    # ── 6-c  role ↔ last-name mapping inside intro
    role_last = {}
    for m in INTRO_ROLE_PAT.finditer(intro):
        role_raw = m.group("role").upper()
        role_norm = (
            "VICE CHAIR" if "VICE" in role_raw else
            "RANKING MEMBER" if "RANKING" in role_raw else "CHAIR"
        )
        full = m.group("full").strip()
        last = full.split()[-1].upper()
        role_last[role_norm] = last
        last2name.setdefault(last, full.title())
        keep.add(last)

    if "CHAIR" not in role_last and (m := FALLBACK_CHAIR_PAT.search(intro)):
        last = m.group("surname").upper()
        role_last["CHAIR"] = last
        last2name.setdefault(last, f"Chair {last.title()}")
        keep.add(last)

    keep.add(PLACEHOLDER)

    drop = {
        l for l in {m.group(1).upper() for m in CUE_LAST_RE.finditer(text)} - keep
        if re.search(rf"\b{l.title()}\b", intro)
    }

    # ── 6-d  main cue scan
    segs, spk, pos, keep_block = [], [], 0, False
    total_cues = 0

    for m in CR_SPEAKER_RE.finditer(text):
        total_cues += 1
        cue = m.group("n")
        last = cue.split()[-1].upper().strip("().")
        if last not in last2name:
            last2name[last] = cue.title()

        # Is this a chair cue?  (e.g. "THE CHAIRMAN.")
        is_chair_cue = bool(re.match(r"(?:THE\s+)?(?:ACTING\s+)?CHAIR", cue, re.I))
        keep_this = (last in keep) or is_chair_cue

        if is_chair_cue and last not in last2name:
            last = PLACEHOLDER

        if keep_block and not keep_this:
            raw = text[pos:m.start()].strip()
            if raw:
                raw = BRACKET_RE.sub("", raw)
                raw = ARTIFACT_RE.sub("", raw)
                segs.append(clean_segment(raw))
            keep_block = False

        if keep_this:
            if keep_block:
                raw = text[pos:m.start()].strip()
                if raw:
                    raw = BRACKET_RE.sub("", raw)
                    raw = ARTIFACT_RE.sub("", raw)
                    segs.append(clean_segment(raw))
            spk.append(last)
            keep_block, pos = True, m.end()
        else:
            keep_block, pos = False, m.end()

    if keep_block:
        raw = text[pos:].strip()
        if raw:
            raw = BRACKET_RE.sub("", raw)
            raw = ARTIFACT_RE.sub("", raw)
            segs.append(clean_segment(raw))

    log(f"[PARSE] {fd['file_path'].name:40}  intro_hits={len(intro_hits):<3} "
        f"keep={len(keep):<3}  cues_found={total_cues:<5}  kept={len(spk):<4}  segs={len(segs):<4}")

    meta = {
        "File": fd['file_path'].name,
        "Date": dtx,
        "HearingTitle": title,
        "intro_hits": len(intro_hits),
        "cues_found": total_cues
    }

    rows = [{
        "Date": dtx,
        "File": fd['file_path'].name,
        "HearingTitle": title,
        "Name": last2name.get(l, f"({l.title()})"),
        "SpeakerTitle": None,
        "Text": tx
    } for l, tx in zip(spk, segs)]

    return rows, meta

# ── main processing loop ─────────────────
first_clean_write = first_drop_write = first_no_intro_write = True
processed = 0
batch_count = 0

# Process files in batches
for i in range(0, len(all_xmls), BATCH_SIZE):
    batch_count += 1
    batch_files = all_xmls[i:i + BATCH_SIZE]
    
    log(f"\n{dash}\n▶ BATCH {batch_count}  ({len(batch_files)} files)\n")
    
    rows, dropped, no_intro = [], [], []
    
    for fp in batch_files:
        # Parse XML file
        fd = parse_xml_file(fp)
        if fd is None:
            dropped.append({
                "File": fp.name,
                "Date": "unknown",
                "HearingTitle": "unknown",
                "Reason": "XML parse error"
            })
            continue
            
        # Process the parsed data
        try:
            signal.signal(signal.SIGALRM,
                          lambda *a, **k: (_ for _ in ()).throw(TimeoutError()))
            signal.alarm(60)                       # start 60-sec timer
            r, meta = clean_hearing_from_data(fd)  # do the heavy work
            signal.alarm(0)                        # cancel timer
        except TimeoutError:
            log(f"[TIMEOUT] {fd['file_path'].name} – skipped")
            dropped.append({
                "File": fd['file_path'].name,
                "Date": fd["date"],
                "HearingTitle": fd["title"],
                "Reason": "Timeout"
            })
            continue
        except Exception as e:
            log(f"[ERROR] {fd['file_path'].name}: {e} – skipped")
            dropped.append({
                "File": fd['file_path'].name,
                "Date": fd["date"],
                "HearingTitle": fd["title"],
                "Reason": str(e)
            })
            continue
            
        if r:
            rows.extend(r)
        else:
            dropped.append(meta)
            
        if meta["intro_hits"] == 0:
            no_intro.append(meta)
            
        processed += 1
        
        # Clear memory after processing each file
        del fd, r, meta
        
    # Write results incrementally
    if rows:
        df = pd.DataFrame(rows)
        agg = (df.groupby(["Date", "File", "HearingTitle", "Name"],
                          as_index=False)
                  .agg({"Text": " ".join}))
        agg = agg.merge(meta_df, on="Name", how="left")
        agg.to_csv(OUT_CLEAN, mode="a", index=False,
                   header=first_clean_write)
        first_clean_write = False
        log(f"  • wrote {len(agg):,} rows")
        del df, agg

    if dropped:
        pd.DataFrame(dropped).to_csv(OUT_DROP, mode="a", index=False,
                                     header=first_drop_write)
        first_drop_write = False
        log(f"  • logged {len(dropped):,} discarded hearings")

    if no_intro:
        pd.DataFrame(no_intro).to_csv(OUT_NO_INT, mode="a", index=False,
                                      header=first_no_intro_write)
        first_no_intro_write = False
        log(f"  • noted {len(no_intro):,} intro-less files")

    # Clear batch memory
    del rows, dropped, no_intro
    gc.collect()

log(f"\n{dash}")
log(f"FINISHED – processed {processed:,} XML files")
log(f"Clean CSV       : {OUT_CLEAN}")
log(f"Discarded CSV   : {OUT_DROP}")
log(f"No-intro CSV    : {OUT_NO_INT}")