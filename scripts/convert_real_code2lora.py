#!/usr/bin/env python3
"""Convert the REAL Code2LoRA/RepoPeftBench datasets into our local schema.

Sources (downloaded from HF under data/real_code2lora/):
  * code2lora-evo             -- PRIMARY. Full per-commit history for all
    400 train + 49 cr_val + 51 cr_test repos (58,617 commit rows for train
    alone), each with repo_state_embedding (2048-d), diff_embedding
    (2048-d, embeds production_code_diff), and the literal
    production_code_diff text. QnA files (train/ir_val/ir_test/cr_val/
    cr_test) carry assertion_event_type + old_target -- i.e. this is real
    diff/change data, not just static snapshots.
  * code2lora-static-anchor   -- supplementary: qna/train.parquet has the
    static-track anchor-based QnAs (different extraction protocol than
    evo's train QnAs -- both are valid, kept as separate rows).
  * repopeftbench-ood         -- 92-repo temporal holdout, used only for
    held-out evaluation, never trained on.

Output (appended, not overwritten, so re-running is additive-safe against
accidental double-runs is NOT guaranteed -- this script always rewrites
its own output files from scratch):
  data/embeddings/real_code2lora_embeddings.parquet
      doc_id = f"{repo_id}@{commit_sha[:10]}", doc_embedding = repo_state_embedding
  data/qna/real_code2lora_qna.jsonl
      one row per assertion-completion task, joined against the embeddings
      above via the real (repo_id, commit_sha) pair (never guessed).
  data/embeddings/real_code2lora_diffs.parquet
      doc_id = f"{repo_id}@{commit_sha[:10]}", diff_embedding, and the raw
      production_code_diff text -- kept SEPARATE from repo_state so a
      future "what changed at this commit" task can condition on the diff
      specifically rather than the whole-repo snapshot.

Usage:
    python scripts/convert_real_code2lora.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Set, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))
from memory_lora.data_paths import DATA_ROOT, EMBEDDINGS_DIR, QNA_DIR, ensure_dirs  # noqa: E402

REAL_ROOT = DATA_ROOT / "real_code2lora"
EVO_ROOT = REAL_ROOT / "code2lora-evo"
ANCHOR_ROOT = REAL_ROOT / "code2lora-static-anchor"
OOD_ROOT = REAL_ROOT / "repopeftbench-ood"


def _doc_id(repo_id: str, commit_sha: str) -> str:
    return f"{repo_id}@{str(commit_sha)[:10]}"


def convert_embeddings_and_diffs() -> Set[Tuple[str, str]]:
    """evo/commits/{split}.parquet -> repo-state embeddings AND diff
    embeddings (kept in separate output files). Returns the set of
    (repo_id, commit_sha) pairs with a real embedding, for the QnA join."""
    emb_rows, diff_rows = [], []
    valid_keys: Set[Tuple[str, str]] = set()
    seen_ids: Set[str] = set()

    for split_file, split_label in [
        ("train.parquet", "train"), ("cr_val.parquet", "cr_val"), ("cr_test.parquet", "cr_test"),
    ]:
        path = EVO_ROOT / "commits" / split_file
        if not path.exists():
            print(f"  [skip] {path} not found", flush=True)
            continue
        table = pq.read_table(path, columns=[
            "repo_id", "commit_sha", "repo_state_embedding", "diff_embedding", "production_code_diff",
        ])
        n = table.num_rows
        repo_col = table.column("repo_id").to_pylist()
        sha_col = table.column("commit_sha").to_pylist()
        emb_col = table.column("repo_state_embedding").to_pylist()
        diff_emb_col = table.column("diff_embedding").to_pylist()
        diff_text_col = table.column("production_code_diff").to_pylist()
        for i in range(n):
            key = (repo_col[i], sha_col[i])
            valid_keys.add(key)
            doc_id = _doc_id(*key)
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                emb_rows.append({
                    "doc_id": doc_id, "doc_version": sha_col[i],
                    "split": split_label, "category": "real_code_repo",
                    "doc_embedding": emb_col[i],
                })
            if diff_emb_col[i] is not None:
                diff_rows.append({
                    "doc_id": doc_id, "doc_version": sha_col[i], "split": split_label,
                    "diff_embedding": diff_emb_col[i],
                    "diff_text": (diff_text_col[i] or "")[:4000],
                })
        print(f"  evo {split_label}: {n} (repo, commit) rows, {len(set(repo_col))} unique repos", flush=True)

    # supplementary: OOD holdout (from static-anchor, evo has no OOD split)
    ood_file = OOD_ROOT / "ood_test.parquet"
    if ood_file.exists():
        table = pq.read_table(ood_file)
        if "repo_state_embedding" in table.column_names:
            n = table.num_rows
            repo_col = table.column("repo_id").to_pylist()
            sha_col = table.column("commit_sha").to_pylist()
            emb_col = table.column("repo_state_embedding").to_pylist()
            for i in range(n):
                key = (repo_col[i], sha_col[i])
                valid_keys.add(key)
                doc_id = _doc_id(*key)
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    emb_rows.append({
                        "doc_id": doc_id, "doc_version": sha_col[i],
                        "split": "cr_test", "category": "real_code_repo_ood",
                        "doc_embedding": emb_col[i],
                    })
            print(f"  ood: {n} (repo, commit) rows, {len(set(repo_col))} unique repos", flush=True)

    emb_table = pa.table({
        "doc_id": [r["doc_id"] for r in emb_rows],
        "doc_version": [r["doc_version"] for r in emb_rows],
        "split": [r["split"] for r in emb_rows],
        "category": [r["category"] for r in emb_rows],
        "doc_embedding": [r["doc_embedding"] for r in emb_rows],
    })
    emb_path = EMBEDDINGS_DIR / "real_code2lora_embeddings.parquet"
    pq.write_table(emb_table, emb_path)
    print(f"Wrote {len(emb_rows)} real repo embeddings -> {emb_path}", flush=True)

    diff_table = pa.table({
        "doc_id": [r["doc_id"] for r in diff_rows],
        "doc_version": [r["doc_version"] for r in diff_rows],
        "split": [r["split"] for r in diff_rows],
        "diff_embedding": [r["diff_embedding"] for r in diff_rows],
        "diff_text": [r["diff_text"] for r in diff_rows],
    })
    diff_path = EMBEDDINGS_DIR / "real_code2lora_diffs.parquet"
    pq.write_table(diff_table, diff_path)
    print(f"Wrote {len(diff_rows)} real diff embeddings -> {diff_path}", flush=True)

    return valid_keys


def convert_qna(valid_keys: Set[Tuple[str, str]]) -> int:
    """evo/qna/{split}.parquet (primary) + static-anchor/qna/train.parquet
    (supplementary static-track anchors) -> our jsonl rows, joined against
    valid_keys (only keep QnAs whose (repo_id, commit_sha) has a real
    embedding)."""
    out_path = QNA_DIR / "real_code2lora_qna.jsonl"
    n_written, n_dropped = 0, 0

    sources = [
        (EVO_ROOT / "qna" / "train.parquet", "train", "train"),
        (EVO_ROOT / "qna" / "ir_val.parquet", "train", "held_out"),
        (EVO_ROOT / "qna" / "ir_test.parquet", "train", "held_out"),
        (EVO_ROOT / "qna" / "cr_val.parquet", "cr_val", "held_out"),
        (EVO_ROOT / "qna" / "cr_test.parquet", "cr_test", "held_out"),
        (ANCHOR_ROOT / "qna" / "train.parquet", "train", "train"),
    ]
    with out_path.open("w") as f:
        for path, doc_split, qna_split in sources:
            if not path.exists():
                print(f"  [skip] {path} not found", flush=True)
                continue
            table = pq.read_table(path, columns=["repo_id", "commit_sha", "prefix", "target"])
            n = table.num_rows
            repo_col = table.column("repo_id").to_pylist()
            sha_col = table.column("commit_sha").to_pylist()
            prefix_col = table.column("prefix").to_pylist()
            target_col = table.column("target").to_pylist()
            kept = 0
            for i in range(n):
                key = (repo_col[i], sha_col[i])
                if key not in valid_keys:
                    n_dropped += 1
                    continue
                f.write(json.dumps({
                    "doc_id": _doc_id(repo_col[i], sha_col[i]), "doc_version": sha_col[i],
                    "split": doc_split, "qna_split": qna_split,
                    "question": "", "prefix": prefix_col[i], "target": target_col[i],
                }) + "\n")
                n_written += 1
                kept += 1
            print(f"  {path.parent.parent.name}/{path.name}: {kept}/{n} QnAs matched "
                  f"-> split={doc_split} qna_split={qna_split}", flush=True)
    print(f"Wrote {n_written} real QnA pairs ({n_dropped} dropped, no matching "
          f"embedding) -> {out_path}", flush=True)
    return n_written


def main() -> None:
    ensure_dirs()
    print("Converting real repo + diff embeddings (from code2lora-evo)...", flush=True)
    valid_keys = convert_embeddings_and_diffs()
    print(f"\n{len(valid_keys)} valid (repo, commit) embedding keys found.\n", flush=True)
    print("Converting real QnA pairs (joined against real embeddings)...", flush=True)
    n_qna = convert_qna(valid_keys)
    print(f"\nDone: {len(valid_keys)} real repo-commit docs, {n_qna} real QnA pairs.", flush=True)


if __name__ == "__main__":
    main()
