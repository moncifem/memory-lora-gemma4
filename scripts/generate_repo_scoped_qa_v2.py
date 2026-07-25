#!/usr/bin/env python3
"""ENHANCED repo-scoped QA generator (batch2 / novelty pass).

Same 6-view input as generate_repo_scoped_qa.py, but the prompt asks for a
BROADER, NOVEL set of judgment aspects that the earlier passes under-covered --
so the new repos teach the model NEW capabilities, not just more of the same:

  existing aspects : architecture, data_flow, why, contracts, conventions, impact, ops
  NEW aspects here : debugging (failure modes / how to diagnose),
                     security (attack surface / trust boundaries / validation),
                     performance (hot paths / scaling / cost),
                     concurrency (races / ordering / data consistency),
                     migration (how to evolve/refactor safely / deprecation),
                     testing_strategy (what a good test here asserts)

Still Tier-A/B discipline: short judgment answers, NO exact file/line lists.
Appends to the SAME data/qna/repo_scoped_qa.jsonl (resume-safe by doc_id) so the
next-day assemble picks it up automatically.

Usage:
  ./venv/bin/python scripts/generate_repo_scoped_qa_v2.py \
      --sources data/docs/multiview_sources.jsonl \
      --model google/gemma-4-31b-it --workers 10 --only-repos data/batch2_novel_repos.txt
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
MODEL = "google/gemma-4-31b-it"; BASE_URL = "https://openrouter.ai/api/v1"

SYSTEM = (
 "You are a 20-year staff engineer writing ADVANCED onboarding Q&A for a senior "
 "engineer joining a repository. You get 6 views: call/import graph, architecture "
 "(readme/tree), git history, test contracts, code conventions, and ops/build "
 "config. Produce a JSON array of 10-14 objects with keys 'aspect','question',"
 "'answer'. aspect in: debugging, security, performance, concurrency, migration, "
 "testing_strategy, data_flow, impact. Rules: (1) questions are REPO-LEVEL and "
 "test HARD JUDGMENT a senior would reason about -- 'where would a bug in X most "
 "likely surface', 'what is the trust boundary / attack surface here', 'what is "
 "the likely hot path or scaling bottleneck', 'what ordering/consistency "
 "assumption could break under concurrency', 'how would you evolve Y without "
 "breaking callers', 'what must a good test here assert'. (2) answers SHORT -- one "
 "phrase or sentence, copyable, NO file-path or line-number lists. (3) ground "
 "every answer in the provided views; invent nothing; if a view is thin, reason "
 "from what IS present. Output ONLY the JSON array."
)

def ckey(*p): return hashlib.sha256("||".join(p).encode()).hexdigest()[:24]

def gen(client, ctx, cache_dir, retries=4) -> List[Dict]:
    cf = cache_dir / f"rsqa2_{ckey(MODEL, ctx)}.json"
    if cf.exists():
        raw = cf.read_text()
    else:
        for a in range(retries):
            try:
                r = client.chat.completions.create(model=MODEL,
                    messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": ctx}],
                    max_tokens=4000, temperature=0.6)
                raw = r.choices[0].message.content or ""; cf.write_text(raw); break
            except Exception:  # noqa: BLE001
                time.sleep(2 ** a)
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
    ap.add_argument("--only-repos", default="", help="file of repo\\tlang -- restrict to these repos (the novel batch)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    globals()["MODEL"] = args.model
    ensure_dirs()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key: raise SystemExit("OPENROUTER_API_KEY not set")
    client = OpenAI(base_url=BASE_URL, api_key=key)

    only = None
    if args.only_repos and Path(args.only_repos).exists():
        only = {l.split("\t")[0].split()[0].strip() for l in open(args.only_repos) if l.strip()}

    src = [json.loads(l) for l in open(args.sources)]
    if only is not None:
        src = [r for r in src if r["doc_id"] in only or r.get("repo") in only]
    outp = QNA_DIR / "repo_scoped_qa.jsonl"
    done = set()
    if outp.exists():
        done = {json.loads(l)["doc_id"] for l in open(outp)}
    todo = [r for r in src if r["doc_id"] not in done]
    print(f"enhanced QA: {len(todo)} repos to do (of {len(src)} selected)", flush=True)
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
                    "target": " " + qa["answer"], "lang": row.get("lang", "unknown"),
                    "source": "batch2_novel"}) + "\n")
                n_qa[0] += 1
            n_docs[0] += 1
            if n_docs[0] % 25 == 0:
                print(f"  {n_docs[0]}/{len(todo)} repos, {n_qa[0]} QA "
                      f"({n_docs[0]/max(1e-9,(time.time()-t0)/60):.1f}/min)", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for _ in as_completed([ex.submit(work, r) for r in todo]):
            pass
    out_f.close()
    print(f"\nDone. {n_docs[0]} novel repos -> {n_qa[0]} enhanced (debug/security/perf/"
          f"concurrency/migration) QA.", flush=True)

if __name__ == "__main__":
    main()
