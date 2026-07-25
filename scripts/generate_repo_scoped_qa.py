#!/usr/bin/env python3
"""Generate REPO-SCOPED judgment QA from the SAME 6 views that
build_repo_multiview.py embedded -- so target scope matches input scope
(a repo-level embedding must be paired with repo-level questions, not
commit-scoped ones). Reads data/docs/multiview_sources.jsonl (per-repo
view_text) and writes data/qna/repo_scoped_qa.jsonl keyed by the same
doc_id (= repo name).

gemini-3.6-flash: reasoning mandatory -> generous max_tokens.
Tier-A/B discipline: judgment answers, short, no exact file/line lists.

Usage:
  python scripts/generate_repo_scoped_qa.py --limit 3
  python scripts/generate_repo_scoped_qa.py
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, sys, time
from pathlib import Path
from typing import Dict, List
from openai import OpenAI

HERE = Path(__file__).resolve().parent; REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))
from memory_lora.data_paths import QNA_DIR, DOCS_DIR, CACHE_DIR, ensure_dirs  # noqa: E402

def _load_dotenv(p: Path):
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip())
_load_dotenv(REPO_ROOT / ".env")
MODEL = "google/gemini-3.6-flash"; BASE_URL = "https://openrouter.ai/api/v1"

SYSTEM = (
 "You are a 20-year tech lead writing onboarding Q&A for a NEW engineer joining "
 "a repository. You are given 6 views of the repo: its call/import graph, "
 "architecture (readme/tree), git history, test contracts, code conventions, "
 "and ops/build config. Produce a JSON array of 8-12 objects with keys "
 "'aspect','question','answer'. aspect in: architecture, data_flow, why, "
 "contracts, conventions, impact, ops. Rules: (1) questions are REPO-LEVEL and "
 "test JUDGMENT a senior would have ('what layer owns X', 'what convention does "
 "this repo use for Y', 'how does data flow through Z', 'why is this structured "
 "this way'), NOT trivia about one commit. (2) answers SHORT -- one phrase or "
 "sentence, copyable, NO file-path or line-number lists. (3) 'impact' = the KIND "
 "of thing that breaks/changes, not an exact file enumeration. (4) ground every "
 "answer in the provided views; invent nothing. Output ONLY the JSON array."
)

def ckey(*p): return hashlib.sha256("||".join(p).encode()).hexdigest()[:24]

def gen(client, ctx, cache_dir, retries=4) -> List[Dict]:
    cf = cache_dir / f"rsqa_{ckey(MODEL, ctx)}.json"
    if cf.exists():
        raw = cf.read_text()
    else:
        last = None
        for a in range(retries):
            try:
                r = client.chat.completions.create(model=MODEL,
                    messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": ctx}],
                    max_tokens=4000, temperature=0.6)
                raw = r.choices[0].message.content or ""; cf.write_text(raw); break
            except Exception as e:  # noqa: BLE001
                last = e; time.sleep(2 ** a)
        else:
            return []
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m: return []
    try: items = json.loads(m.group(0))
    except json.JSONDecodeError: return []
    out = []
    for it in items:
        q = (it.get("question") or "").strip(); a = (it.get("answer") or "").strip()
        asp = (it.get("aspect") or "general").strip()
        if q and a and len(a) < 240: out.append({"aspect": asp, "question": q, "answer": a})
    return out

def build_ctx(repo: str, vt: Dict[str, str]) -> str:
    parts = [f"REPOSITORY: {repo}\n"]
    for k in ["v_arch", "v_graph", "v_history", "v_contracts", "v_conventions", "v_ops"]:
        parts.append(f"=== {k} ===\n{vt.get(k,'')}\n")
    return "\n".join(parts)

def main():
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default=str(DOCS_DIR / "multiview_sources.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    globals()["MODEL"] = args.model
    ensure_dirs()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key: raise SystemExit("OPENROUTER_API_KEY not set")
    client = OpenAI(base_url=BASE_URL, api_key=key)

    src = [json.loads(l) for l in open(args.sources)]
    if args.limit: src = src[: args.limit]
    outp = QNA_DIR / "repo_scoped_qa.jsonl"
    done = set()
    if outp.exists():
        done = {json.loads(l)["doc_id"] for l in open(outp)}
    todo = [r for r in src if r["doc_id"] not in done]
    out_f = outp.open("a")
    lock = threading.Lock()
    n_docs = [0]; n_qa = [0]; t0 = time.time()

    def work(row):
        doc_id = row["doc_id"]
        qas = gen(client, build_ctx(row["repo"], row["view_text"]), CACHE_DIR)
        if not qas: return
        with lock:
            for qa in qas:
                out_f.write(json.dumps({"doc_id": doc_id, "doc_version": "head",
                    "split": "train", "qna_split": "train", "aspect": qa["aspect"],
                    "question": qa["question"], "prefix": f"Q: {qa['question']}\nA:",
                    "target": " " + qa["answer"], "lang": row.get("lang", "unknown")}) + "\n")
                n_qa[0] += 1
            n_docs[0] += 1
            if n_docs[0] % 25 == 0:
                print(f"  {n_docs[0]}/{len(todo)} repos, {n_qa[0]} QA "
                      f"({n_docs[0]/max(1e-9,(time.time()-t0)/60):.1f}/min)", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for _ in as_completed([ex.submit(work, r) for r in todo]):
            pass
    out_f.close()
    print(f"\nDone. {n_docs[0]} repos -> {n_qa[0]} repo-scoped judgment QA.", flush=True)

if __name__ == "__main__":
    main()
