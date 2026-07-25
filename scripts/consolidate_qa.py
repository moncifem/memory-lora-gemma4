#!/usr/bin/env python3
"""Consolidate all tech-lead QA sources into one balanced training file with
a PER-REPO CAP, so no single repo/domain dominates. This is what fixes the
Django problem: SWE-bench inherently has ~5 Python repos (django-dominated)
with thousands of QA already generated; capping per repo collapses django
from ~2200 to <=CAP while keeping the 700+ distinct repos' diversity.

Reads: data/qna/techlead_qa.jsonl, techlead_qa_commitpack.jsonl,
       repo_scoped_qa.jsonl  (+ optional multilang tags)
Writes: data/qna/techlead_consolidated.jsonl
"""
from __future__ import annotations
import argparse, json, glob, random, sys
from collections import Counter, defaultdict
from pathlib import Path
HERE = Path(__file__).resolve().parent; REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))
from memory_lora.data_paths import QNA_DIR

def repo_of(doc_id): return doc_id.split("@")[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-repo-cap", type=int, default=15)
    ap.add_argument("--out", default=str(QNA_DIR / "techlead_consolidated.jsonl"))
    args = ap.parse_args()
    rng = random.Random(3407)

    files = [QNA_DIR / "techlead_qa.jsonl", QNA_DIR / "techlead_qa_commitpack.jsonl",
             QNA_DIR / "repo_scoped_qa.jsonl"]
    by_repo = defaultdict(list)
    for f in files:
        if not f.exists(): continue
        for l in open(f):
            try: d = json.loads(l)
            except json.JSONDecodeError: continue
            by_repo[repo_of(d["doc_id"])].append(d)

    kept = []
    for repo, rows in by_repo.items():
        if len(rows) > args.per_repo_cap:
            rows = rng.sample(rows, args.per_repo_cap)
        kept.extend(rows)
    rng.shuffle(kept)

    with open(args.out, "w") as fo:
        for d in kept: fo.write(json.dumps(d) + "\n")

    langs = Counter(d.get("lang", "python?") for d in kept)
    repos = Counter(repo_of(d["doc_id"]) for d in kept)
    django = sum(v for k, v in repos.items() if "django" in k.lower())
    print(f"consolidated: {len(kept)} QA | {len(repos)} distinct repos | cap={args.per_repo_cap}/repo")
    print(f"  django share: {django} ({100*django/max(1,len(kept)):.1f}%)  <- was 46%")
    print(f"  langs: {dict(langs)}")
    print(f"  top repos: {repos.most_common(5)}")
    print(f"  -> {args.out}")

if __name__ == "__main__":
    main()
