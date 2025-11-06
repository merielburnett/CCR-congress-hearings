#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CHAIRPERSON FINDER — FULL RUN (low-RAM server, proxy-safe, resume-safe)

RAM-reduction strategies:
• Server/CLI both use small context & batches: CTX_SIZE=512, BATCH=16, UBATCH=16
• HTTP requests set cache_prompt:false (prevents cache growth)
• Optional periodic server restart to reclaim allocator memory (no effect on results)
• Stream XML filenames instead of allocating a huge list

Output CSV: /home/ec2-user/SageMaker/data/output_files/chairpersons_full.csv
Columns    : File, Chair
"""

import os, sys, csv, json, shlex, textwrap, subprocess, time, socket, http.client, atexit, gc
from pathlib import Path
from datetime import date
import unicodedata
import lxml.etree as ET

# ── Bypass proxies for localhost (important on AppStream) ─────────────────────
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
os.environ.setdefault("no_proxy", "127.0.0.1,localhost")

# ── Paths & config ────────────────────────────────────────────────────────────
BASE_DIR    = Path("/home/ec2-user/SageMaker/data")
XML_DIR     = BASE_DIR / "Congress_Hearings"
OUT_DIR     = BASE_DIR / "output_files"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV     = OUT_DIR / "chairpersons_full.csv"   # resume-safe target
EARLIEST    = date(1870, 1, 1)
INTRO_WORDS = 200
FLUSH_EVERY = 50    # append to CSV every N rows
RESTART_EVERY = 5000  # restart server every N processed files to reclaim RAM (0 to disable)

# LLM (CPU build)
LLM_DIR     = BASE_DIR / "llm_bundle"
LLAMAFILE   = LLM_DIR / "llamafile"
MODEL_GGUF  = BASE_DIR / "llm_bundle" / "llama.cpp" / "models" / "Phi-3-mini-4k-instruct-q4.gguf"

# Server mode settings
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8095

# Generation knobs (kept conservative)
N_TOKENS       = 64
TEMP           = 0.10
N_GPU_LAYERS   = 0
THREADS        = 6

# *** Low-RAM core knobs ***
CTX_SIZE       = 512
BATCH          = 16
UBATCH         = 16

SEED           = 42
TIMEOUT_SEC    = 120

# Tokens of interest
CHAIR_WORDS = {"chair", "chairman", "chairwoman", "chairperson"}

# Words to ignore in fallback name
HEADER_NOUNS = {
    'committee','subcommittee','commission','security','cooperation','europe','united','states',
    'house','senate','document','number','available','website','gpo','govinfo','gov','minority',
    'majority','report','hearing','session','congress','resolution','oversight','investigation',
    'department','agency','office','services','program'
}

SUFFIXES = {"jr.", "sr.", "ii", "iii", "iv", "v"}
TITLES   = {"mr.", "mrs.", "ms.", "dr.", "hon.", "sen.", "rep.", "senator", "representative"}

# ── CSV helpers (resume-safe) ─────────────────────────────────────────────────
def ensure_header(csv_path: Path):
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=["File", "Chair"])
            wr.writeheader()

def load_done(csv_path: Path) -> set[str]:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()
    done = set()
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = (row.get("File") or "").strip()
            if fn:
                done.add(fn)
    return done

def append_rows(csv_path: Path, rows: list[dict]):
    if not rows:
        return
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["File", "Chair"])
        for r in rows:
            wr.writerow(r)

# ── Start a persistent llama.cpp server (or fall back) ────────────────────────
_server_proc = None

def _port_open(host: str, port: int, timeout=0.75) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False

def _http_post_json(host: str, port: int, path: str, payload: dict, timeout: float):
    """Proxy-free localhost POST using http.client."""
    body = json.dumps(payload).encode("utf-8")
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        conn.request("POST", path, body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        data = resp.read()
        return resp.status, data
    finally:
        try:
            conn.close()
        except Exception:
            pass

def start_llama_server() -> bool:
    """
    Start ./llamafile --server and wait until /completion responds.
    Returns True if server ready, else False (fall back to CLI if needed).
    """
    global _server_proc

    if not LLAMAFILE.exists() or not os.access(LLAMAFILE, os.X_OK):
        return False

    # If something is already listening, assume it’s ready.
    if _port_open(SERVER_HOST, SERVER_PORT):
        return True

    cmd = [
        str(LLAMAFILE),
        "--server",
        "--host", SERVER_HOST,
        "--port", str(SERVER_PORT),
        "-m", str(MODEL_GGUF),
        "-ngl", str(N_GPU_LAYERS),
        "-t", str(THREADS),
        "-c", str(CTX_SIZE),
        "-b", str(BATCH),
        "--ubatch", str(UBATCH),
        "--no-display-prompt",
    ]
    try:
        _server_proc = subprocess.Popen(
            cmd,
            cwd=str(LLM_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            text=False,
            close_fds=True,
        )
    except OSError:
        _server_proc = None
        return False

    # Wait up to ~25s for readiness with a tiny completion
    ready_by = time.time() + 25
    test_payload = {"prompt": "OK", "n_predict": 4, "temperature": 0.0, "cache_prompt": False}

    while time.time() < ready_by:
        if _port_open(SERVER_HOST, SERVER_PORT):
            try:
                status, data = _http_post_json(SERVER_HOST, SERVER_PORT, "/completion", test_payload, timeout=1.5)
                if status == 200:
                    return True
            except Exception:
                pass
        time.sleep(0.5)

    stop_llama_server()
    return False

def stop_llama_server():
    global _server_proc
    if _server_proc and _server_proc.poll() is None:
        try:
            _server_proc.terminate()
        except Exception:
            pass
    _server_proc = None

def restart_llama_server_if_needed(processed_count: int):
    """Optional: restart the server periodically to reclaim memory."""
    if RESTART_EVERY and processed_count and (processed_count % RESTART_EVERY == 0):
        stop_llama_server()
        time.sleep(1.0)
        start_llama_server()

import atexit
atexit.register(stop_llama_server)

# ── LLM runners: HTTP first, CLI fallback ─────────────────────────────────────
USE_HTTP = False

def run_llama_http(prompt: str) -> str:
    payload = {
        "prompt": prompt,
        "n_predict": N_TOKENS,
        "temperature": TEMP,
        "seed": SEED,
        "stop": [],
        "cache_prompt": False,   # prevent cache growth across requests
    }
    status, data = _http_post_json(SERVER_HOST, SERVER_PORT, "/completion", payload, timeout=TIMEOUT_SEC)
    txt = data.decode("utf-8", errors="ignore")
    if status != 200:
        return txt
    # Server JSON can be {"content": "..."} or {"completion": "..."}; parse if possible
    try:
        obj = json.loads(txt)
        return obj.get("content") or obj.get("completion") or txt
    except Exception:
        return txt

def run_llama_cli(prompt: str) -> str:
    cmd = textwrap.dedent(f"""
        cd {shlex.quote(str(LLM_DIR))} && \
        ./llamafile \
          -m {shlex.quote(str(MODEL_GGUF))} \
          -p {shlex.quote(prompt)} \
          -n {int(N_TOKENS)} \
          -ngl {int(N_GPU_LAYERS)} \
          -t {int(THREADS)} \
          -c {int(CTX_SIZE)} \
          -b {int(BATCH)} \
          --ubatch {int(UBATCH)} \
          --temp {TEMP} \
          --seed {int(SEED)} \
          --no-display-prompt
    """).strip()
    proc = subprocess.run(
        ["/bin/bash", "-lc", cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=TIMEOUT_SEC,
        env={**os.environ, "GGML_VERBOSE":"0"}
    )
    return proc.stdout

def run_llama(prompt: str) -> str:
    if USE_HTTP:
        return run_llama_http(prompt)
    return run_llama_cli(prompt)

# ── Prompt (same as the working version) ──────────────────────────────────────
PROMPT_TEMPLATE = """\
You will be given a SHORT, LOWER-CASED roster slice from the very start of a U.S. congressional
hearing. The slice begins at the first word “committee” or “commission” and ends a few words after
the first occurrence of the keyword “chair / chairman / chairwoman / chairperson”.

Your job: return the PERSONAL NAME of the **single** chair of THIS hearing.

Helpful clues:
- The chair’s name usually appears BEFORE the “chair*” word.
- It is often the first person listed immediately after the title/description line.
- Sometimes the name appears AFTER the “chair*” word (e.g., “chairman jack brooks presiding”).
- Names may include initials (e.g., “j.”), hyphens, apostrophes, accents, and suffixes (jr., iii).
- Ignore states/affiliations; return just the person’s name (e.g., “patrick j. leahy”).
- If you are unsure, still give your BEST GUESS rather than “none”.

Return exactly one JSON object on a single line:
{{"chair":"<full name in lower case>","confidence":<float 0..1>}}

SNIPPET (lower-cased):
---
{snippet_lc}
---
"""

def parse_json_anywhere(text: str):
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except Exception:
            pass
    a, b = text.find("{"), text.rfind("}")
    if 0 <= a < b:
        try:
            return json.loads(text[a:b+1])
        except Exception:
            return None
    return None

# ── XML streaming + date gate ─────────────────────────────────────────────────
def parse_xml_header_and_intro(fp: Path):
    dtx = "0000-00-00"
    words = []
    try:
        for _, el in ET.iterparse(str(fp), events=("end",),
                                  tag=("Text", "NumericDate"),
                                  huge_tree=True):
            if el.tag == "NumericDate" and el.text:
                dtx = el.text.strip()
            elif el.tag == "Text" and el.text:
                need = INTRO_WORDS - len(words)
                if need > 0:
                    words.extend(el.text.split()[:max(0, need)])
            el.clear()
            # free parsed siblings promptly
            while el.getprevious() is not None:
                del el.getparent()[0]
            if len(words) >= INTRO_WORDS and dtx != "0000-00-00":
                break
    except Exception:
        pass
    return dtx, " ".join(words)

def date_ok(dtx: str) -> bool:
    parts = (dtx or "").split("-")
    if len(parts) != 3:
        return False
    try:
        y, m, d = map(int, parts)
        return date(y, m, d) >= EARLIEST
    except Exception:
        return False

# ── Segment builder (committee/commission → chair+3) ──────────────────────────
def build_segment(text: str, post_words: int = 3):
    words = text.split()
    lowers = [w.lower() for w in words]

    # first committee/commission
    start = None
    for i, w in enumerate(lowers):
        if w.startswith("committee") or w.startswith("commission"):
            start = i
            break
    if start is None:
        return None, None

    # first chair token after that
    end_chair_idx = None
    for i in range(start, len(lowers)):
        if lowers[i] in CHAIR_WORDS:
            end_chair_idx = min(len(words), i + 1 + post_words)
            break
    if end_chair_idx is None:
        return None, None

    seg_orig = " ".join(words[start:end_chair_idx])
    seg_lc   = seg_orig.lower()
    return seg_orig, seg_lc

# ── Fallback: first “full name” (no brittle regex) ────────────────────────────
def is_letterlike_token(tok: str) -> bool:
    core = tok.strip(" ,.;:()[]{}")
    if not core:
        return False
    if len(core) <= 3 and core.endswith(".") and core[0].isalpha():
        return True
    for ch in core:
        if ch in "-'’.":
            continue
        cat = unicodedata.category(ch)
        if not (cat.startswith("L") or ch.isalpha()):
            return False
    return True

def likely_header_word(tok: str) -> bool:
    return tok.strip(" ,.;:()").lower() in HEADER_NOUNS

def to_title(tok: str) -> str:
    c = tok.strip(" ,.;:()")
    if not c:
        return ""
    if len(c) <= 3 and c.endswith(".") and c[0].isalpha():
        return c.upper()
    return c[:1].upper() + c[1:].lower()

def tokenize_with_punct(s: str):
    seps = {",", ".", "(", ")", ";", ":"}
    out, cur = [], []
    for ch in s:
        if ch.isspace():
            if cur: out.append("".join(cur)); cur=[]
        elif ch in seps:
            if cur: out.append("".join(cur)); cur=[]
            out.append(ch)
        else:
            cur.append(ch)
    if cur: out.append("".join(cur))
    return out

def fallback_first_full_name(segment_original: str) -> str | None:
    toks = tokenize_with_punct(segment_original)
    run = []
    def flush():
        nonlocal run
        if run:
            cnt = sum(1 for t in run if t not in {",", ".", ";", ":"})
            if 1 <= cnt <= 6:
                parts = []
                for t in run:
                    low = t.strip(" ,.;:()").lower()
                    if low in TITLES and not parts:
                        continue
                    if low in SUFFIXES:
                        parts.append(low.upper().rstrip(".") + ".")
                        continue
                    if is_letterlike_token(t) and not likely_header_word(t):
                        parts.append(to_title(t))
                parts = [p for p in parts if p]
                if len(parts) >= 2:
                    return " ".join(parts[:6]).strip()
            run = []
        return None

    for t in toks:
        if t in {",", ".", ";", ":"}:
            name = flush()
            if name:
                return name
            run = []
            continue
        if is_letterlike_token(t) and not likely_header_word(t):
            run.append(t)
        else:
            name = flush()
            if name:
                return name
            run = []
    return flush()

# ── Main (full run, low-RAM server, resume-safe) ──────────────────────────────
def main():
    global USE_HTTP

    print("────────────────────────────────────────────────────────────────────────")
    print("CHAIRPERSON FINDER — FULL RUN (low-RAM server, proxy-safe, resume-safe)")
    print(f"XML dir    : {XML_DIR}")
    print(f"Model      : {MODEL_GGUF}")
    print(f"Llamafile  : {LLAMAFILE}")
    print(f"Output CSV : {OUT_CSV}")
    print("────────────────────────────────────────────────────────────────────────")

    if not MODEL_GGUF.exists():
        print("❌ model .gguf not found; check path."); return
    if not XML_DIR.exists():
        print("❌ XML directory not found."); return

    # Try to start HTTP server; else fall back to CLI mode.
    USE_HTTP = start_llama_server()
    print(f"LLM mode   : {'HTTP server (warm, low-RAM)' if USE_HTTP else 'CLI fallback (cold, low-RAM)'}")

    ensure_header(OUT_CSV)
    already_done = load_done(OUT_CSV)
    print(f"▶ Resume info: {len(already_done):,} files already in CSV; will skip them.")

    out_rows = []
    processed = 0
    written   = len(already_done)
    started   = time.time()

    try:
        # Stream filenames to avoid building a huge list in memory
        for fp in XML_DIR.iterdir():
            if fp.suffix.lower() != ".xml":
                continue

            stem = fp.stem
            if stem in already_done:
                continue

            dtx, intro = parse_xml_header_and_intro(fp)
            if not dtx or not date_ok(dtx):
                continue
            if not intro.strip():
                continue

            seg_orig, seg_lc = build_segment(intro, post_words=3)
            if not seg_orig:
                continue

            prompt = PROMPT_TEMPLATE.format(snippet_lc=seg_lc[:1400])

            try:
                raw = run_llama(prompt)
            except Exception:
                # If the server hiccups mid-run, try CLI for this file
                if USE_HTTP:
                    try:
                        raw = run_llama_cli(prompt)
                    except Exception:
                        raw = ""
                else:
                    raw = ""

            name_from_llm = None
            obj = parse_json_anywhere(raw)
            if isinstance(obj, dict):
                val = obj.get("chair")
                if isinstance(val, str):
                    v = val.strip().lower()
                    if v and v != "none":
                        name_from_llm = " ".join(w[:1].upper() + w[1:] for w in v.split())

            if name_from_llm:
                out_rows.append({"File": stem, "Chair": name_from_llm})
            else:
                fallback = fallback_first_full_name(seg_orig)
                if fallback:
                    out_rows.append({"File": stem, "Chair": fallback})

            processed += 1

            if len(out_rows) >= FLUSH_EVERY:
                append_rows(OUT_CSV, out_rows)
                written += len(out_rows)
                already_done.update(r["File"] for r in out_rows)
                out_rows.clear()
                gc.collect()

                # light heartbeat every ~200 processed
                if processed % (FLUSH_EVERY * 4) == 0:
                    elapsed = int(time.time() - started)
                    print(f"… processed={processed:,} | written={written:,} | elapsed={elapsed}s")

            # Optional periodic server refresh to reclaim RAM
            restart_llama_server_if_needed(processed)

        if out_rows:
            append_rows(OUT_CSV, out_rows)
            written += len(out_rows)
            out_rows.clear()

    finally:
        # Stop server to free RAM when done. (Comment out to keep it warm across runs.)
        stop_llama_server()

    total_written = len(load_done(OUT_CSV))
    print("────────────────────────────────────────────────────────────────────────")
    print(f"DONE. CSV rows (unique files with chair): {total_written:,}")
    print(f"CSV path: {OUT_CSV}")

if __name__ == "__main__":
    main()
