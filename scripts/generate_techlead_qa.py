#!/usr/bin/env python3
"""Generate high-quality, judgment-shaped "Tech Lead" QA from REAL GitHub
data (SWE-bench: real issues + fix patches + test patches + PR discussion),
via OpenRouter google/gemini-3.6-flash.

Design decisions baked in from this session's analysis:
- Targets are TIER-A/B *judgment* QA only (why / architecture / data-flow /
  contracts / conventions / impact-NARRATIVE), never Tier-C exact-recall
  ("list the precise files+lines") which a LoRA can only hallucinate.
- Answers are SHORT (a phrase/sentence) so they're compressible into a
  weight-delta and gradeable.
- Each example is keyed to doc_id = "{repo}@{base_commit[:10]}" so it can
  later be tied to a repo-state embedding for hypernetwork training.
- gemini-3.6-flash is a REASONING model: reasoning is mandatory and cannot
  be disabled, and it consumes completion tokens BEFORE content -- so
  max_tokens must be generous (default 3500) or content comes back empty.

Usage:
  python scripts/generate_techlead_qa.py --limit 3          # smoke test
  python scripts/generate_techlead_qa.py --max-instances 1200
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

SYSTEM = (
 "You are a staff engineer writing onboarding Q&A about a real code change. "
 "Given an issue, its fix patch, the test patch, and any discussion, produce "
 "a JSON array of 5-8 objects with keys 'aspect', 'question', 'answer'. "
 "aspect must be one of: architecture, data_flow, why, contracts, conventions, "
 "impact. Rules: (1) Questions test JUDGMENT and UNDERSTANDING, not memorized "
 "trivia. (2) Answers are SHORT -- one phrase or one sentence, copyable, no "
 "lists of file paths or line numbers. (3) 'why' questions explain the "
 "rationale/trade-off. 'impact' questions describe the KIND of thing affected "
 "(a behavior, a contract), NOT an exact file enumeration. (4) Ground every "
 "answer in the provided change; do not invent APIs. Output ONLY the JSON array."
)

def cache_key(*parts) -> str:
    return hashlib.sha256("||".join(parts).encode()).hexdigest()[:24]

def gen_qa(client, ctx: str, cache_dir: Path, retries=4) -> List[Dict]:
    key = cache_key(MODEL, ctx)
    cf = cache_dir / f"tlqa_{key}.json"
    if cf.exists():
        raw = cf.read_text()
    else:
        last = None
        for a in range(retries):
            try:
                r = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": ctx}],
                    max_tokens=3500, temperature=0.6)
                raw = r.choices[0].message.content or ""
                cf.write_text(raw); break
            except Exception as e:  # noqa: BLE001
                last = e; time.sleep(2 ** a)
        else:
            print(f"    [warn] failed: {last}", flush=True); return []
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m: return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for it in items:
        q = (it.get("question") or "").strip()
        a = (it.get("answer") or "").strip()
        asp = (it.get("aspect") or "general").strip()
        if q and a and len(a) < 240:
            out.append({"aspect": asp, "question": q, "answer": a})
    return out

def build_ctx(row) -> str:
    def clip(s, n): return (s or "")[:n]
    return (f"REPO: {row['repo']}\n"
            f"ISSUE:\n{clip(row.get('problem_statement'),1500)}\n\n"
            f"DISCUSSION:\n{clip(row.get('hints_text'),800)}\n\n"
            f"FIX PATCH:\n{clip(row.get('patch'),2500)}\n\n"
            f"TEST PATCH:\n{clip(row.get('test_patch'),1200)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-instances", type=int, default=1200)
    ap.add_argument("--limit", type=int, default=0, help="smoke-test cap")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    ensure_dirs()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key: raise SystemExit("OPENROUTER_API_KEY not set")
    client = OpenAI(base_url=BASE_URL, api_key=api_key)

    from datasets import load_dataset
    ds = load_dataset("princeton-nlp/SWE-bench", split=args.split, streaming=True)

    n_max = args.limit if args.limit else args.max_instances
    qna_f = (QNA_DIR / "techlead_qa.jsonl").open("a")
    src_f = (DOCS_DIR / "techlead_sources.jsonl").open("a")
    n_docs = n_qa = 0
    t0 = time.time()
    for row in ds:
        if n_docs >= n_max: break
        doc_id = f"{row['repo']}@{row['base_commit'][:10]}"
        ctx = build_ctx(row)
        qas = gen_qa(client, ctx, CACHE_DIR)
        if not qas: continue
        # store the source context so a repo-state embedding can be built later
        src_f.write(json.dumps({"doc_id": doc_id, "repo": row["repo"],
                                "base_commit": row["base_commit"], "context": ctx}) + "\n")
        for qa in qas:
            qna_f.write(json.dumps({
                "doc_id": doc_id, "doc_version": row["base_commit"],
                "split": "train", "qna_split": "train",
                "aspect": qa["aspect"], "question": qa["question"],
                "prefix": f"Q: {qa['question']}\nA:", "target": " " + qa["answer"],
            }) + "\n")
            n_qa += 1
        n_docs += 1
        if n_docs % 10 == 0:
            rate = n_docs / max(1e-9, (time.time() - t0) / 60)
            print(f"  {n_docs} instances, {n_qa} QA ({rate:.1f} inst/min) latest={doc_id}", flush=True)
    qna_f.close(); src_f.close()
    print(f"\nDone. {n_docs} real instances -> {n_qa} judgment-QA pairs.", flush=True)
    print(f"  {QNA_DIR/'techlead_qa.jsonl'}\n  {DOCS_DIR/'techlead_sources.jsonl'}", flush=True)

if __name__ == "__main__":
    main()
