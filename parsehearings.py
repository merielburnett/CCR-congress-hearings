#!usrbinenv python.txt
#!/usr/bin/env python

#  TARGETED HEARING PARSER – version 25 (memory-efficient, no precheck)
#      removed precheck process for speed
#      added memory clearing in batch loop
#       parse all XMLs (not just substantive ones)
#      maintains rerun capability

import json, re, itertools, gc, lxml.etree as ET, pandas as pd
from pathlib import Path
from datetime import date
import signal  #need this to help rerun the script when TDM studio kicks me off
import sys   # added this

# ── config ──
MIN_WORDS  = 500
EARLIEST   = date(1873, 1, 1)
BATCH_SIZE = 25

BASE_DIR   = Path("/home/ec2-user/SageMaker/data")
CORPUS_DIR = BASE_DIR / "Congress_Hearings"
OUT_DIR    = BASE_DIR / "output_files"
OUT_DIR.mkdir(parents=True, exist_ok=True)

#rite sample outputs for testing. Change names back for full runs.
OUT_CLEAN  = OUT_DIR / "sample_hearings_clean.csv"
OUT_DROP   = OUT_DIR / "sample_hearings_discarded.csv"
OUT_NO_INT = OUT_DIR / "sample_no_intro.csv"

TEST_MAX = None  # ← set to None to disable

dash = "─"*110
log  = print

#print the target paths so we know what will be written
log("▶ OUTPUT PATHS")
log(f"   OUT_DIR   : {OUT_DIR.resolve()}")
log(f"   OUT_CLEAN : {OUT_CLEAN.resolve()}")
log(f"   OUT_DROP  : {OUT_DROP.resolve()}")
log(f"   OUT_NO_INT: {OUT_NO_INT.resolve()}\n")


# ──  build roster ─
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

import re

INTRO_W = 2_000                              # chars to scan at top of file
BULLET  = r"[*•●]"

# — Titles (final period REQUIRED) --------------------------------------------
FUZZY_MR  = r"M\w?r\."        # Mr., Mir., M r.
FUZZY_MRS = r"M\w?rs\."       # Mrs., Mras.
FUZZY_MS  = r"M\w?s\."        # Ms., M s.
FUZZY_DR  = r"D\w?r\."        # Dr., D r.

TITLE_WORD = (
    f"{FUZZY_MR}|{FUZZY_MRS}|{FUZZY_MS}|{FUZZY_DR}|"
    r"Miss|Mx|Prof\.?|Hon\.?|Judge|Justice|Rev\.?|Pastor|"
    r"Senator|Representative|General|Colonel|President|Lieutenant|Captain|Corporal|Governor|Corporal|Minister"
    r"The Honorable|Professor|Admiral|Major|Sergeant|"
    r"Congressman|Congresswoman|Delegate|Chairman|Chairwoman|Chairperson"
)

# — Role tokens used in introductory lines ------------------------------------
ROLE_TOK = r"(?:CHAIR(?:MAN|WOMAN|PERSON)?|VICE\s+CHAIR|RANKING\s+MEMBER)"

# — BOUNDED fuzzy CHAIR (tolerates typos like CHAiRmaN / CHarman, but won’t eat text)
FUZZY_CHAIR = r"""
(?:
    C(?:H|K)        # C + H (or occasional K)
    [Aa]            # A
    [Ii1l]?         # I (or 1 / l) sometimes missing
    [Rr]            # R
    (?:                             # optional suffixes
        \s*[- ]?\s*
        (?:
            M(?:A?N)?               # MAN / MN / M
          | W(?:O?MAN)              # WOMAN / WMAN
          | PERSON
        )
    )?
)
"""

# — Institutional speaker labels that stand in for the chair -------------------
ROLE_SPEAKER = rf"(?:The\s+(?:ACTING\s+)?(?:PRESIDING\s+OFFICER|{FUZZY_CHAIR}))"
INSTITUTIONAL = (
    r"(?:The\s+(?:VICE\s+)?PRESIDENT(?:\s+pro\s+tempore)?|"
    r"The\s+SPEAKER(?:\s+pro\s+tempore)?|"
    r"The\s+CLERK|"
    r"The\s+CHIEF\s+JUSTICE)"
)

# — Name tokens – bounded & must end with '.' or ':' (NO comma) ----------------
# Unicode‑aware versions that accept À‑Ö, Ø‑Þ, à‑ö, ø‑ÿ
ALPHA_U = r"A-ZÀ-ÖØ-Þ"        # upper‑case Latin incl. accents
ALPHA_L = r"a-zà-öø-ÿ"        # lower‑case Latin incl. accents

INIT       = rf"(?:[{ALPHA_U}]\.)"
CAP_NAME   = rf"(?:[{ALPHA_U}][{ALPHA_L}]+(?:-[{ALPHA_U}][{ALPHA_L}]+)?)"
ALLCAP     = rf"(?:[{ALPHA_U}]{{2,}}(?:'[{ALPHA_U}]{{2,}})?)"
SUFFIX     = r"(?:Jr\.?|Sr\.?|II|III|IV|V)"

NAME_TOKEN = rf"(?:{ALLCAP}|{CAP_NAME}|{INIT})"
NAME_BLOCK = rf"(?:{NAME_TOKEN})(?:\s+{NAME_TOKEN}){{0,3}}(?:\s+{SUFFIX})?"
OF_TAIL = rf"(?:\s+of\s+[{ALPHA_U}][{ALPHA_L}]+(?:\s+[{ALPHA_U}][{ALPHA_L}]+)?)?"

LEAD_MARGIN = rf"(?:\s{{0,8}}|<bullet>|{BULLET})?"   # optional whitespace/bullet prefix

# **Primary speaker-cue regex**
#  Works anywhere in a line (no ^ anchor), but **requires** final '.' or ':'
#  Thus “Mr. Donaldson,” will NOT match; “Mr. Donaldson.” / “Mr. Donaldson:” will.
CR_SPEAKER_RE = re.compile(
    rf"""
    (?P<prefix>{LEAD_MARGIN})\s*
    (?P<n>
        (?:
            (?:(?:{TITLE_WORD})\s+){NAME_BLOCK}{OF_TAIL}   # titled personal name
          | (?:{ROLE_SPEAKER})                              # The CHAIR / PRESIDING OFFICER …
          | (?:{INSTITUTIONAL})                             # institutional roles
          | (?:{FUZZY_MR}\s+Counsel\s+{ALLCAP})             # Mr. Counsel SMITH
        )
    )
    \s*(?:\([^)]+\)\s*)?                 # optional parenthetical
    (?P<delim>[.:])\s+                   # REQUIRED delimiter: '.' or ':'
    """,
    re.I | re.X,
)

# — Boilerplate & artifact removal --------------------------------------------
VERDATE_ART_RE = re.compile(r"VerDate[\s\S]*?(?=\b[a-z]{5,}\w*)", re.M)
ROLLCALL_RE    = re.compile(r"\[Roll(?:call)?(?:\s+Vote)?\s+No\.\s+\d+.*?\]", re.I | re.S)

RECORDER_START_RE = re.compile(
    r"""(?mi)
    ^\s*(
        The\ (?:assistant\ )?legislative\ clerk\ read\ as\ follows|
        The\ nomination\ (?:considered\ and\ confirmed|was\ confirmed) .*|
        There\ being\ no\ objection,|
        The\ (?:assistant\ )?legislative\ clerk|
        The\ resolution\ .*?was\ agreed\ to\.|
        The\ preamble\ was\ agreed\ to\.|
        The\ resolution\ .*?reads\ as\ follows|
        The\ assistant\ editor .*? proceeded\ to\ call\ the\ roll|
        The\ bill\ clerk (?:proceeded\ to\ )?call(?:ed)?\ the\ roll\.?|
        The\ question\ was\ taken(?:;|\.)|
        The\ yeas\ and\ nays (?:were\ ordered|resulted.*?,\ as\ follows:)|
        The\ result\ was\ announced.*?,\ as\ follows:|
        Amendment\ No\.\ \d+.*?is\ as\ follows:|
        The\ text\ of\ the.*?is\ as\ follows|
        amended(?:\ to\ read)?\ as\ follows:|
        The\ material (?:previously\ )?referred\ to (?:by.*?)?is\ as\ follows:|
        There\ was\ no\ objection|
        The\ amendment.*?was\ agreed\ to|
        The\ motion\ to\ table\ was .*|
        The\ following\ bills\ and\ joint\ resolutions\ were\ introduced.*|
        The\ vote\ was\ taken\ by\ electronic\ device|
        A\ recorded\ vote\ was\ ordered
    ).*$
    """,
    re.I | re.M | re.S | re.X
)

STMT_PAT        = re.compile(r"STATEMENTS? OF\s+[A-Z][A-Z'\-]+(?:\s+[A-Z][A-Z'\-]+){0,2}")
BRACKET_RE      = re.compile(r"\[\s*[A-Za-z]+\.*\s*\]", re.I)
TITLE_TRAIL_RE  = re.compile(rf"(?:{TITLE_WORD})(?:\s+[A-Z][A-Za-z'\-]+)?\s*$", re.I)
ORPHAN_TITLE_RE = re.compile(rf"\s+(?:{TITLE_WORD})\.\s*$", re.I)
CUE_LAST_RE     = re.compile(r"\b([A-Z][A-Z'\-]+)\s*[:\.]\s+")

# Relaxed ALL‑CAPS block killer (committee headings etc.)
UWS        = r"[ \u00A0\t\r\n\f\v\u2000-\u200B\u202F\u205F\u3000]+"
CAPS_TOKEN = r"[A-Z]{2,}[A-Za-z ']*[,.·]?"
CAPS_BLOCK_RE = re.compile(
    rf"""
        (?:\s*\d+[,\s]+)?              
        (?:{CAPS_TOKEN}{UWS}?[,]*){{3,}}
        (?:{UWS}?[,\.]?\s*\d+)?        
    """,
    re.X
)

ARTIFACT_RE = CAPS_BLOCK_RE

def clean_segment(seg: str) -> str:
    """
    Segment-level tidy **after** speaker-cue extraction:
        remove TABLE I./II./… blocks (window capped at 500 chars)
        strip dot-leader runs but keep the surrounding text
        de-hyphenate & dash spacing
        kill ALL-CAP blocks
        collapse surplus whitespace
    """
    # TABLE blocks – non-greedy, look-ahead window ≤ 500 chars
    seg = re.sub(
        r"TABLE\s+[IVXLC]+\.\s*(?:.(?!\b[A-Z][a-z]+\b(?:\s+\w+){7,})){0,500}",
        " ",
        seg,
        flags=re.I | re.S,
    )

    # dot-leader runs (“..........”) → single space (preserve text)
    seg = re.sub(r"\.{6,}", " ", seg)

    # dash & hyphen fixes
    seg = re.sub(r"\s*[\-–—]{1,2}\s*", " ", seg)          # spaced dashes -space
    seg = re.sub(r"(\w)[\-–—]{1,2}(\w)", r"\1\2", seg)    # de-hyphenate inside tokens

    # ALL-CAP heading/artifact killer
    seg = CAPS_BLOCK_RE.sub(" ", seg)

    # final whitespace collapse
    return re.sub(r"\s{2,}", " ", seg).strip()

def scrub_artifacts(txt: str) -> str:
    """
    Remove boiler-plate PLUS:
        • TABLE I./II./… blocks (roman-numeral tables)  
          – cut from “TABLE X.” up to the first *normal* sentence
            (≥ 8 words beginning with a capital letter).
        • dotted-leader lines such as
              “All personal health services ............ 4.3”
          – any run of ≥ 6 periods nukes the whole line.
    """
    # original removals
    txt = VERDATE_ART_RE.sub("", txt)
    txt = ROLLCALL_RE.sub("", txt)
    txt = RECORDER_START_RE.sub("", txt)

    # — keep ONLY the basic boiler-plate removals here —
    txt = VERDATE_ART_RE.sub("", txt)
    txt = ROLLCALL_RE.sub("", txt)
    txt = RECORDER_START_RE.sub("", txt)
    return txt


word_pat = re.compile(r"[A-Za-z0-9]")

# — Intro: discover the chair's personal name so all role cues map to it --------
CAP_NAME   = r"(?:[A-Z][a-z]+(?:-[A-Z][a-z]+)?)"
INIT       = r"(?:[A-Z]\.)"
SUFFIX     = r"(?:Jr\.?|Sr\.?|II|III|IV|V)"
INTRO_NAME = rf"{CAP_NAME}(?:\s+(?:{CAP_NAME}|{INIT})){{0,3}}\s+{CAP_NAME}(?:\s+{SUFFIX})?"

DECOR = r"[<>\[\]()*_~\s]*"
INTRO_ROLE_PAT = re.compile(
    rf"""
    (?:
        ^|\n
    )
    {DECOR}(?:THE\s+)?(?P<role>{ROLE_TOK}){DECOR}
    (?P<full>{INTRO_NAME})
    \s*,?
    """,
    re.I | re.M | re.X
)

FALLBACK_CHAIR_PAT = re.compile(
    rf"""
    (?:^|\n)\s*
    {FUZZY_CHAIR}\s+
    (?:(?:{FUZZY_MR}|{FUZZY_MRS}|{FUZZY_MS})\s+)?
    (?P<surname>[A-Z][A-Z'\-]+)
    \s*[.:]
    """,
    re.I | re.M | re.X
)

PLACEHOLDER = "__UNKNOWN_CHAIR__"

# ── discover CHAIR personal name inside the intro window ──────────────────────
def find_chair_name(intro_text: str) -> str | None:
    """
    Return the chair’s personal name (full name when possible, otherwise
    at least the surname) discovered inside the *intro* window.

    Search order:
        1) “THE CHAIR / VICE CHAIR … <Full-Name>,”  (existing INTRO_ROLE_PAT)
        2) “OPENING STATEMENT [HON] <FULL-CAP NAME> … CHAIRMAN”
        3) “Chairman|Chairwoman <Name …>” roster style lines
        4) Fallback fuzzy pattern (CHAIR … SURNAME.)  (existing)
    """
    # 1  original strict pattern  (e.g. “THE CHAIR Hon. Susan Collins,”)
    m = INTRO_ROLE_PAT.search(intro_text)
    if m:
        return m.group("full").strip()

    # 2  OPENING-STATEMENT lines  (e.g. “OPENING STATEMENT HON MAX BAUCUS … CHAIRMAN”)
    m = re.search(
        r"""
        OPENING\W+STATEMENT        # opening-statement header
        \W+(?:HON\.?\W+)?          # optional HON.
        (?P<full>                  # capture full name in caps
            [A-Z][A-Z'\-]*(?:\s+[A-Z][A-Z'\-]*){1,4}
        )
        \W+.*?\bCHAIR(?:MAN|WOMAN|PERSON)\b   # … until the word CHAIRMAN/CHAIRWOMAN
        """,
        intro_text,
        flags=re.I | re.S | re.X,
    )
    if m:
        return m.group("full").title()

    # 3  roster lines  (e.g. “Chairman JOHN ROCKEFELLER IV West Virginia” or
    #                     “FRED UPTON Michigan, Chairman JOE BARTON Texas”)
    #    – look both after and before the word “Chairman/Chairwoman/Chair”
    m = re.search(
        r"""
        (?:\bCHAIR(?:MAN|WOMAN|PERSON)\b\W+
           (?P<after>[A-Z][A-Z'\-]*(?:\s+[A-Z][A-Z'\-]*){0,4})      # after-pattern
        |
           (?P<before>[A-Z][A-Z'\-]*(?:\s+[A-Z][A-Z'\-]*){0,4})\W+
           ,?\s*\bCHAIR(?:MAN|WOMAN|PERSON)\b                        # before-pattern
        )
        """,
        intro_text,
        flags=re.I | re.X,
    )
    if m:
        full = (m.group("after") or m.group("before")).strip()
        # keep only last word as surname when the string is long (e.g. includes first name)
        surname = full.split()[-1].title()
        return full.title() if len(full.split()) > 1 else surname

    # 4  original fuzzy fallback ( “CHAiRmaN SURNAME.” )
    m = FALLBACK_CHAIR_PAT.search(intro_text)
    if m:
        return m.group("surname").title()

    # Nothing found
    return None

def normalize_chair(speaker_name: str, chair_full: str | None) -> str:
    """
    Map any role form ('The CHAIR', fuzzy CHAiRmaN, PRESIDING OFFICER, etc.)
    to the discovered chair personal name. If unknown, return a stable placeholder.
    """
    if re.search(rf"^(?:\s*(?:The\s+)?)?(?:ACTING\s+)?(?:PRESIDING\s+OFFICER|{FUZZY_CHAIR})\b",
                 speaker_name, flags=re.I | re.X):
        return chair_full if chair_full else PLACEHOLDER
    return speaker_name

# : state-name fallback
states_long = [
    'alabama','alaska','arizona','arkansas','california','colorado','connecticut','delaware',
    'district of columbia','florida','georgia','hawaii','idaho','illinois','indiana','iowa',
    'kansas','kentucky','louisiana','maine','maryland','massachusetts','michigan','minnesota',
    'mississippi','missouri','montana','nebraska','nevada','new hampshire','new jersey',
    'new mexico','new york','north carolina','north dakota','ohio','oklahoma','oregon',
    'pennsylvania','rhode island','south carolina','south dakota','tennessee','texas','utah',
    'vermont','virginia','washington','west virginia','wisconsin','wyoming'
]
STATE_NAME_FALLBACK_RE = re.compile(
    rf"\b({'|'.join(states_long)})\b,?\s+([A-Z][A-Z'.\-]+(?:\s+[A-Z][A-Z'.\-]+){{0,3}})",
    re.I
)

def build_intro_name_regex(names):
    """Optional helper if you pre-collect a roster of full names."""
    mid = r"(?:\s+(?:[A-Z][\w'\-.]*|[A-Z]\.))*"
    suf = r"(?:\s+(?:Jr\.?|Sr\.?|II|III|IV|V))?"
    pieces = []
    for full in names:
        parts = full.split()
        if len(parts) < 2:
            continue
        pieces.append(rf"{re.escape(parts[0])}{mid}\s+{re.escape(parts[-1])}{suf}")
    return re.compile(r"\b(" + "|".join(pieces) + r")\b", re.I)

# ── STREAM set-up & RERUN guard ─────────────────────
def _seen(csv_path, col="File"):
    """Return set of filenames already present in a CSV (empty if file absent)."""
    return (set(pd.read_csv(csv_path, usecols=[col])[col])
            if csv_path.exists() and csv_path.stat().st_size else set())

already_done = _seen(OUT_CLEAN) | _seen(OUT_DROP) | _seen(OUT_NO_INT)
log(f"▶ RERUN  |  {len(already_done):,} XMLs already parsed – will be skipped\n")

# --- Build file list (FULL-RUN mode) ------------------------------------------
all_xmls = sorted(
    p for p in CORPUS_DIR.glob("*.xml")
    if p.name not in already_done           # skip files already parsed
)
log(f"▶ STREAM |  {len(all_xmls):,} XMLs left to process\n")

# If nothing to do, exit cleanly
if not all_xmls:
    log("No XML files found for processing; exiting.")
    sys.exit(0)

# ── XML parsing function ──────────────────────────

def parse_xml_file(fp: Path) -> dict | None:
    """
    Parse a single XML file and extract date, title, and full text.
    Returns a dict with the parsed data or None if parsing fails or filtered.
    """
    try:
        total_words, dtx, ttl = 0, "0000-00-00", ""
        text_chunks = []

        for _, el in ET.iterparse(str(fp), events=("end",),
                                  tag=("Text", "NumericDate", "Title"),
                                  huge_tree=True):
            if el.tag == "Text" and el.text:
                txt = el.text
                total_words += len(txt.split())
                text_chunks.append(txt)
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

        if MIN_WORDS and total_words < MIN_WORDS:
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

    # active roster for the hearing date (fallback: full roster)
    active = term_df[(term_df["start"] <= dtx) & (term_df["end"] >= dtx)]
    if active.empty:
        active = term_df

    # map LAST→FullName for quick lookups; include placeholder for unknown chair
    last2name = {n.split()[-1].upper(): n for n in active["Name"]}
    last2name[PLACEHOLDER] = "Unknown Chair"

    intro_name_re = build_intro_name_regex(active["Name"])

    # scrub boilerplate artifacts
    text = scrub_artifacts(text)

    # truncate at common termination marker
    if (c := text.find("[Whereupon,")) != -1:
        text = text[:c]

    # excise "STATEMENTS OF ..." blocks (title + following content until next cue)
    while (m := STMT_PAT.search(text)):
        nxt = CR_SPEAKER_RE.search(text, m.end())
        text = text[:m.start()] + text[(nxt.start() if nxt else len(text)):]
    
    # intro window (by words)
    intro = " ".join(text.split()[:INTRO_W])

    # ── 6-a  roster hits in intro
    intro_hits = [m.group(1).title() for m in intro_name_re.finditer(intro)]

    # ── 6-b  fallback STATE → ALL‑CAPS surname patterns
    for m in STATE_NAME_FALLBACK_RE.finditer(intro):
        caps = m.group(2)
        if caps.isupper():
            intro_hits.append(" ".join(p.title() for p in caps.split()))

    keep = {n.split()[-1].upper() for n in intro_hits}

    # ── 6-c  role ↔ last‑name mapping inside intro
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

    # ── discover explicit chair personal name from intro (robust fallback)
    chair_full = find_chair_name(intro)
    chair_last = None
    if chair_full:
        chair_last = chair_full.split()[-1].upper()
        last2name[chair_last] = chair_full  # ensure mapping to personal name
        keep.add(chair_last)

    if "CHAIR" not in role_last and (m := FALLBACK_CHAIR_PAT.search(intro)):
        last = m.group("surname").upper()
        role_last["CHAIR"] = last
        last2name.setdefault(last, f"Chair {last.title()}")
        keep.add(last)

    # always keep placeholder bucket (for unknown chair)
    keep.add(PLACEHOLDER)

    # candidates to drop: last names that appear as cues but weren’t kept,
    # yet are mentioned in the intro text (helps filter witnesses-only mentions)
    drop = {
        l
        for l in {m.group(1).upper() for m in CUE_LAST_RE.finditer(text)} - keep
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

        # chair / presiding‑officer cue? (map to discovered chair when possible)
        is_chair_cue = bool(re.search(r"(?:\bPRESIDING\s+OFFICER\b|" + FUZZY_CHAIR + r")",
                                      cue, re.I | re.X))
        if is_chair_cue:
            if chair_last:
                last = chair_last       # normalize to discovered chair surname
            else:
                last = PLACEHOLDER      # stable bucket if unknown

        keep_this = (last in keep) or is_chair_cue

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

    # tail segment
    if keep_block:
        raw = text[pos:].strip()
        if raw:
            raw = BRACKET_RE.sub("", raw)
            raw = ARTIFACT_RE.sub("", raw)
            segs.append(clean_segment(raw))

    log(
        f"[PARSE] {fd['file_path'].name:40}  intro_hits={len(intro_hits):<3} "
        f"keep={len(keep):<3}  cues_found={total_cues:<5}  kept={len(spk):<4}  segs={len(segs):<4}"
    )

    chair_known_flag = chair_full is not None

    meta = {
        "File": fd['file_path'].name,
        "Date": dtx,
        "HearingTitle": title,
        "intro_hits": len(intro_hits),
        "cues_found": total_cues,
        "chair_known": int(chair_known_flag)   # 1 = found, 0 = unknown
    }

    rows = []
    for l, tx in zip(spk, segs):
        w = tx.split()
        if not w:
            continue
        for i in range(0, len(w), 500):
            chunk = " ".join(w[i:i + 500])
            rows.append({
                "Date": dtx,
                "File": fd['file_path'].name,
                "HearingTitle": title,
                "Name": last2name.get(l, f"({l.title()})"),
                "SpeakerTitle": None,
                "Text": chunk
            })

    return rows, meta


# ── main processing loop ─────────────────
first_clean_write     = not (OUT_CLEAN.exists()  and OUT_CLEAN.stat().st_size  > 0)
first_drop_write      = not (OUT_DROP.exists()   and OUT_DROP.stat().st_size   > 0)
first_no_intro_write  = not (OUT_NO_INT.exists() and OUT_NO_INT.stat().st_size > 0)

processed      = 0
batch_count    = 0
known_chair    = 0   # how many XMLs where chair name was discovered
unknown_chair  = 0   # how many XMLs still Unknown Chair

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
                "Reason": "XML parse error or filtered"
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

        if meta["chair_known"]:
            known_chair += 1
        else:
            unknown_chair += 1

        processed += 1

        # Clear memory after processing each file
        del fd, r, meta

    # Write results incrementally
    if rows:
        df = pd.DataFrame(rows)
        df = df.merge(meta_df, on="Name", how="left")   # add roster columns
        df.to_csv(OUT_CLEAN, mode="a", index=False, header=first_clean_write)
        first_clean_write = False
        log(f"  • wrote {len(df):,} rows")
        del df

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

# Ensure files exist even if no rows were written (useful in TEST mode)
if not OUT_CLEAN.exists():
    # minimal header to make the CSV readable when opened
    pd.DataFrame(columns=["Date","File","HearingTitle","Name","SpeakerTitle","Text"]) \
      .to_csv(OUT_CLEAN, index=False)
if not OUT_DROP.exists():
    pd.DataFrame(columns=["File","Date","HearingTitle","Reason"]) \
      .to_csv(OUT_DROP, index=False)
if not OUT_NO_INT.exists():
    pd.DataFrame(columns=["File","Date","HearingTitle","intro_hits","cues_found"]) \
      .to_csv(OUT_NO_INT, index=False)

log(f"Clean CSV       : {OUT_CLEAN.resolve()}")
log(f"Discarded CSV   : {OUT_DROP.resolve()}")
log(f"No-intro CSV    : {OUT_NO_INT.resolve()}")

log(f"Known chairs    : {known_chair}")
log(f"Unknown chairs  : {unknown_chair}")

# Show what actually landed in the output directory
log("\n▶ OUT_DIR listing")
for p in sorted(OUT_DIR.glob("*.csv")):
    log(f"   {p.name:30}  {p.stat().st_size:>10,} bytes")

if TEST_MAX is not None:
    sys.exit(0)   # ← stop after the test batch
