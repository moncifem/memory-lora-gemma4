#!/usr/bin/env python3
"""Merge RepoPeftBench (Code2LoRA's own benchmark) into the aligned6 corpus.

RepoPeftBench contributes ~500K *assertion-completion* items over 512 repos --
short, exact code targets. aligned6 contributes ~27K *prose* QA over 2066 repos
-- conventions and architecture. They are complementary: the benchmark run
showed the head learns a repo's stack and conventions but not what it actually
does, and exact-recall data is what addresses that.

Split integrity is the thing to get right. A repo must never appear in both a
training split and an eval split, or cross-repo evaluation becomes meaningless.
RepoPeftBench already partitions BY REPO (cr_val / cr_test hold out whole
repositories), so we carry its partition through unchanged and only ever add
repos to `train` when the benchmark itself calls them training repos.

    doc split   : which repos the hypernetwork trains on (train/cr_val/cr_test)
    qna_split   : within a train repo, held-out QA for in-repo eval (ir_*)

Usage:
    python scripts/merge_repopeft_corpus.py \
        --repopeft-emb data/embeddings/repopeft_6view.parquet \
        --out-emb data/embeddings/all_lora_embeddings.parquet \
        --out-qna data/qna/all_lora_qna.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))

# file stem -> (doc split contributed, qna_split)
# cr_* hold out whole repos; ir_* are held-out QA of repos that stay in train.
SPLIT_MAP = {
    "train": ("train", "train"),
    "cr_val": ("cr_val", "train"),
    "cr_test": ("cr_test", "train"),
    "ir_val": ("train", "held_out"),
    "ir_test": ("train", "held_out"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repopeft-emb", default="data/embeddings/repopeft_6view.parquet")
    ap.add_argument("--aligned-emb", default="data/embeddings/aligned6_embeddings.parquet")
    ap.add_argument("--aligned-qna", default="data/qna/aligned6_qna.jsonl")
    ap.add_argument("--repopeft-glob", default="data/real_code2lora/*/qna/*.parquet")
    ap.add_argument("--out-emb", default="data/embeddings/all_lora_embeddings.parquet")
    ap.add_argument("--out-qna", default="data/qna/all_lora_qna.jsonl")
    ap.add_argument("--max-qna-per-repo", type=int, default=400,
                    help="cap per repo: evo alone has ~1000/repo, which would "
                         "swamp the prose QA and bias the head toward one task")
    ap.add_argument("--max-target-chars", type=int, default=400)
    args = ap.parse_args()

    import glob as _glob
    import pyarrow as pa

    # ---------------- embeddings ----------------
    ali = pq.read_table(args.aligned_emb)
    ali_dim = len(ali.column("doc_embedding")[0].as_py())
    print(f"aligned6:  {ali.num_rows} repos, dim {ali_dim}")

    rp_path = Path(args.repopeft_emb)
    if not rp_path.exists():
        print(f"!! missing {rp_path} -- run build_repo_multiview.py first",
              file=sys.stderr)
        sys.exit(1)
    rp = pq.read_table(rp_path)
    rp_dim = len(rp.column("doc_embedding")[0].as_py())
    print(f"repopeft:  {rp.num_rows} repos, dim {rp_dim}")
    if rp_dim != ali_dim:
        print(f"!! dim mismatch {rp_dim} != {ali_dim}; the head cannot consume both",
              file=sys.stderr)
        sys.exit(1)

    # Which split does each RepoPeftBench repo belong to? Derived from the QA
    # files it appears in, so we inherit the benchmark's own repo partition.
    repo_split: dict[str, str] = {}
    for f in sorted(_glob.glob(args.repopeft_glob)):
        stem = Path(f).stem
        if stem not in SPLIT_MAP:
            continue
        doc_split, _ = SPLIT_MAP[stem]
        ids = set(pq.read_table(f, columns=["repo_id"]).column("repo_id").to_pylist())
        for r in ids:
            # An eval assignment always wins: if a repo is used to hold out
            # cross-repo performance anywhere, it must never be trained on.
            if repo_split.get(r) in ("cr_val", "cr_test"):
                continue
            repo_split[r] = doc_split

    rp_ids = rp.column("doc_id").to_pylist()
    rp_embs = rp.column("doc_embedding").to_pylist()
    ali_ids = set(ali.column("doc_id").to_pylist())

    out_ids, out_ver, out_split, out_cat, out_emb = [], [], [], [], []
    for c, col in (("doc_id", out_ids), ("doc_version", out_ver),
                   ("split", out_split), ("category", out_cat)):
        if c in ali.column_names:
            col.extend(ali.column(c).to_pylist())
        else:
            col.extend(["v1"] * ali.num_rows if c == "doc_version"
                       else ["aligned6"] * ali.num_rows)
    out_emb.extend(ali.column("doc_embedding").to_pylist())

    added = 0
    for rid, emb in zip(rp_ids, rp_embs):
        if rid in ali_ids:            # already covered by aligned6
            continue
        sp = repo_split.get(rid)
        if sp is None:                # embedded but no QA -> useless
            continue
        out_ids.append(rid)
        out_ver.append("head")
        out_split.append(sp)
        out_cat.append("repopeftbench")
        out_emb.append(emb)
        added += 1
    print(f"merged embeddings: {len(out_ids)} repos (+{added} from RepoPeftBench)")
    print("  split counts:", dict(Counter(out_split)))

    pq.write_table(pa.table({
        "doc_id": out_ids, "doc_version": out_ver, "split": out_split,
        "category": out_cat, "doc_embedding": out_emb,
    }), args.out_emb)

    # ---------------- QA ----------------
    have_emb = set(out_ids)
    split_of = dict(zip(out_ids, out_split))
    per_repo: dict[str, int] = defaultdict(int)
    n_written = 0
    src_counts: Counter = Counter()

    with open(args.out_qna, "w") as out:
        # aligned6 first, verbatim
        with open(args.aligned_qna) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.write(line + "\n")
                n_written += 1
                src_counts["aligned6"] += 1

        for f in sorted(_glob.glob(args.repopeft_glob)):
            stem = Path(f).stem
            if stem not in SPLIT_MAP:
                continue
            _, qna_split = SPLIT_MAP[stem]
            t = pq.read_table(f, columns=["repo_id", "prefix", "target"])
            rid_c = t.column("repo_id").to_pylist()
            pre_c = t.column("prefix").to_pylist()
            tgt_c = t.column("target").to_pylist()
            kept = 0
            for rid, pre, tgt in zip(rid_c, pre_c, tgt_c):
                if rid not in have_emb:
                    continue
                if not pre or not tgt:
                    continue
                if len(tgt) > args.max_target_chars:
                    continue
                if per_repo[rid] >= args.max_qna_per_repo:
                    continue
                per_repo[rid] += 1
                out.write(json.dumps({
                    "doc_id": rid,
                    "doc_version": "head",
                    "split": split_of[rid],
                    "qna_split": qna_split,
                    "question": "",
                    "prefix": pre,
                    "target": tgt,
                }) + "\n")
                kept += 1
                n_written += 1
            src_counts[Path(f).parent.parent.name + "/" + stem] += kept

    print(f"\nmerged QA: {n_written} rows -> {args.out_qna}")
    for k, v in src_counts.most_common():
        print(f"  {k:<34} {v}")
    print(f"\nrepos with QA: {len(per_repo)}")


if __name__ == "__main__":
    main()
