#!/usr/bin/env python3
"""Assemble the final ALIGNED 6-view training dataset by joining the
multi-view embeddings (inputs, doc_id = 'owner/repo') to ALL tech-lead QA
(targets) matched by repo name (commitpack/swe QA carry 'owner/repo@commit'
-> stripped to 'owner/repo'). This turns 132 repo-scoped-only aligned repos
into 515 aligned repos (~4000 QA) -- the best use of what we built.

Splits (per repo, deterministic): 80% train / 10% cr_val / 10% cr_test, so
the hypernetwork is evaluated on held-out repos it never trained on. Within
train repos, ~15% of QA -> qna_split=held_out (feeds ir_test).

Per-repo QA cap keeps any single repo (e.g. django) from dominating.

Output: data/embeddings/aligned6_embeddings.parquet
        data/qna/aligned6_qna.jsonl
"""
from __future__ import annotations
import json, hashlib, random, sys
from collections import defaultdict
from pathlib import Path
import pyarrow as pa, pyarrow.parquet as pq
HERE = Path(__file__).resolve().parent; REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))
from memory_lora.data_paths import EMBEDDINGS_DIR, QNA_DIR

PER_REPO_CAP = 20

def split_of(repo: str) -> str:
    h = int(hashlib.md5(repo.encode()).hexdigest(), 16) % 100
    if h < 10: return "cr_test"
    if h < 20: return "cr_val"
    return "train"

def main():
    rng = random.Random(3407)
    mv = pq.read_table(EMBEDDINGS_DIR / "multiview_embeddings.parquet").to_pylist()
    mv_by_repo = {}
    for r in mv:                      # dedupe by repo (keep first)
        mv_by_repo.setdefault(r["doc_id"], r)
    mv_repos = set(mv_by_repo)

    qa_by_repo = defaultdict(list)
    for f in ["repo_scoped_qa.jsonl", "techlead_qa_commitpack.jsonl", "techlead_qa.jsonl"]:
        p = QNA_DIR / f
        if not p.exists(): continue
        for l in open(p):
            d = json.loads(l); repo = d["doc_id"].split("@")[0]
            if repo in mv_repos:
                qa_by_repo[repo].append(d)

    # embeddings with splits (only repos that have >=1 QA are trainable/evaluable)
    emb_rows = []
    for repo, r in mv_by_repo.items():
        if repo not in qa_by_repo: continue
        emb_rows.append({"doc_id": repo, "doc_version": "head", "split": split_of(repo),
                         "category": "aligned6", "doc_embedding": r["doc_embedding"]})
    t = pa.table({k: [e[k] for e in emb_rows] for k in
                  ["doc_id", "doc_version", "split", "category", "doc_embedding"]})
    pq.write_table(t, EMBEDDINGS_DIR / "aligned6_embeddings.parquet")

    # qna with qna_split
    n_qa = 0
    from collections import Counter
    split_ct = Counter()
    with (QNA_DIR / "aligned6_qna.jsonl").open("w") as out:
        for repo, rows in qa_by_repo.items():
            sp = split_of(repo)
            if len(rows) > PER_REPO_CAP:
                rows = rng.sample(rows, PER_REPO_CAP)
            for i, d in enumerate(rows):
                if sp == "train":
                    qsplit = "held_out" if rng.random() < 0.15 else "train"
                else:
                    qsplit = "held_out"   # cr_val/cr_test: all held out
                out.write(json.dumps({"doc_id": repo, "doc_version": "head",
                    "split": sp, "qna_split": qsplit, "aspect": d.get("aspect", "?"),
                    "question": d.get("question", ""), "prefix": d["prefix"],
                    "target": d["target"], "lang": d.get("lang", "unknown")}) + "\n")
                n_qa += 1
                split_ct[sp] += 1

    print(f"aligned6 dataset: {len(emb_rows)} repos (embeddings) | {n_qa} QA")
    print(f"  repo splits: {Counter(split_of(r) for r in qa_by_repo)}")
    print(f"  QA by split: {dict(split_ct)}")
    print(f"  -> aligned6_embeddings.parquet + aligned6_qna.jsonl")

if __name__ == "__main__":
    main()
