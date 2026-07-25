#!/usr/bin/env python3
"""Breadth generator: judgment QA from CommitPackFT commits across THOUSANDS
of DISTINCT repos (25k distinct repos in the python shard alone).

Complements generate_techlead_qa.py (SWE-bench), which is deep but spans only
~12 repos. For a HYPERNETWORK, distinct-repo count is the currency of
generalization -- so this takes ~1 commit per NEW repo to maximize breadth,
not many commits of the same repo.

Commit-scope judgment aspects (a single small commit rarely shows whole-system
architecture, so we skip that): why / conventions / contracts / impact.
Same Tier-A/B discipline: short judgment answers, no exact file/line lists.
gemini-3.6-flash reasoning is mandatory -> generous max_tokens.

Usage:
  python scripts/generate_commitpack_qa.py --limit 3
  python scripts/generate_commitpack_qa.py --max-repos 2500 --per-repo 1
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

MODEL = "google/gemini-3.6-flash"
BASE_URL = "https://openrouter.ai/api/v1"
SHARD = REPO_ROOT / "data" / "commitpack" / "multilang_commits.jsonl"

SYSTEM = (
 "You are a staff engineer writing onboarding Q&A about a single real commit. "
 "Given the commit message and the before/after file contents, produce a JSON "
 "array of 3-4 objects with keys 'aspect','question','answer'. aspect in: "
 "why, conventions, contracts, impact. Rules: (1) test JUDGMENT/understanding, "
 "not trivia. (2) answers SHORT -- one phrase/sentence, no file paths or line "
 "numbers. (3) 'why' = rationale/trade-off; 'impact' = the KIND of behavior/"
 "contract affected, not a file list; 'conventions' = the idiom/style this "
 "change follows. (4) ground answers in the actual change; invent nothing. "
 "Output ONLY the JSON array."
)

def cache_key(*p): return hashlib.sha256("||".join(p).encode()).hexdigest()[:24]

def gen_qa(client, ctx, cache_dir, retries=4) -> List[Dict]:
    cf = cache_dir / f"cpqa_{cache_key(MODEL, ctx)}.json"
    if cf.exists():
        raw = cf.read_text()
    else:
        last = None
        for a in range(retries):
            try:
                r = client.chat.completions.create(model=MODEL,
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": ctx}],
                    max_tokens=3000, temperature=0.6)
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

def build_ctx(d) -> str:
    def clip(s, n): return (s or "")[:n]
    return (f"REPO: {d['repos'].split(',')[0]}  FILE: {d.get('new_file','')}\n"
            f"COMMIT: {clip(d.get('subject'),120)}\n{clip(d.get('message'),500)}\n\n"
            f"BEFORE:\n{clip(d.get('old_contents'),1800)}\n\n"
            f"AFTER:\n{clip(d.get('new_contents'),1800)}")

def main():
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-repos", type=int, default=2500)
    ap.add_argument("--per-repo", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--workers", type=int, default=10,
                     help="Concurrent OpenRouter requests. Sequential gen was "
                          "~10x too slow for 3k+ repos; the API handles "
                          "concurrency fine and the per-prompt cache keeps it "
                          "idempotent.")
    args = ap.parse_args()
    globals()["MODEL"] = args.model
    ensure_dirs()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key: raise SystemExit("OPENROUTER_API_KEY not set")
    client = OpenAI(base_url=BASE_URL, api_key=key)

    target = args.limit if args.limit else args.max_repos
    # select one commit per distinct repo (breadth), up to target
    seen: Dict[str, int] = {}
    tasks = []
    for line in open(SHARD):
        if len(tasks) >= target: break
        try: d = json.loads(line)
        except json.JSONDecodeError: continue
        repo = d["repos"].split(",")[0]
        if seen.get(repo, 0) >= args.per_repo: continue
        seen[repo] = seen.get(repo, 0) + 1
        tasks.append(d)

    qna_f = (QNA_DIR / "techlead_qa_commitpack.jsonl").open("a")
    src_f = (DOCS_DIR / "techlead_sources_commitpack.jsonl").open("a")
    lock = threading.Lock()
    n_docs = [0]; n_qa = [0]; t0 = time.time()

    def work(d):
        repo = d["repos"].split(",")[0]
        qas = gen_qa(client, build_ctx(d), CACHE_DIR)
        if not qas: return
        doc_id = f"{repo}@{d['commit'][:10]}"
        with lock:
            src_f.write(json.dumps({"doc_id": doc_id, "repo": repo, "base_commit": d["commit"],
                        "context": build_ctx(d), "source": "commitpackft", "lang": d.get("lang","python")}) + "\n")
            for qa in qas:
                qna_f.write(json.dumps({"doc_id": doc_id, "doc_version": d["commit"],
                    "split": "train", "qna_split": "train", "aspect": qa["aspect"],
                    "question": qa["question"], "prefix": f"Q: {qa['question']}\nA:",
                    "target": " " + qa["answer"], "lang": d.get("lang","python")}) + "\n")
                n_qa[0] += 1
            n_docs[0] += 1
            if n_docs[0] % 50 == 0:
                rate = n_docs[0] / max(1e-9, (time.time() - t0) / 60)
                print(f"  {n_docs[0]} repos, {n_qa[0]} QA ({rate:.1f} repo/min)", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for _ in as_completed([ex.submit(work, d) for d in tasks]):
            pass
    qna_f.close(); src_f.close()
    print(f"\nDone. {n_docs[0]} DISTINCT repos -> {n_qa[0]} judgment-QA pairs.", flush=True)

if __name__ == "__main__":
    main()
