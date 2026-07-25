#!/usr/bin/env python3
"""Augment a document's QA pairs with paraphrased variants of the SAME facts.

Why: the direct-LoRA experiment on the Code2LoRA paper doc (runs/direct_paper)
converged to train_loss=0.0 but only hit 33% exact-match on its OWN training
questions at free-generation time, with several wrong answers collapsing to
the same recycled string (e.g. "47.4%" answering four different percentage
questions). With ~1 phrasing per fact and ~18 facts total, the model found a
shortcut (a few frequently-reinforced answers) instead of learning to
discriminate between questions. Multiple paraphrases of the SAME fact force
the model to actually condition on question content rather than pattern-
match a training-set shortcut.

This does NOT introduce new facts -- it takes the existing (question,
answer) pairs for a document and asks the model to rephrase the QUESTION
several different ways while keeping the answer identical.

Usage:
    python scripts/augment_paraphrases.py --doc-id code2lora_paper --n-paraphrases 5
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import List

from openai import OpenAI

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))
from memory_lora.data_paths import QNA_DIR, CACHE_DIR, ensure_dirs  # noqa: E402
from generate_synthetic_dataset import (  # noqa: E402
    CachedClient, _load_dotenv, DEFAULT_GEN_MODEL, OPENROUTER_BASE_URL,
)

_load_dotenv(REPO_ROOT / ".env")

PARAPHRASE_SYSTEM = (
    "You rewrite a factual question in N different ways while keeping its "
    "meaning and the correct short answer completely unchanged. Output a "
    "JSON array of N strings, each a differently-phrased question that "
    "still has the exact same answer as the original. Vary sentence "
    "structure, word choice, and question format (e.g. direct question, "
    "'what value/number', fill-in-the-blank style) -- but never change what "
    "is being asked. Output ONLY the JSON array, no prose."
)


def gen_paraphrases(client: CachedClient, model: str, question: str, answer: str, n: int) -> List[str]:
    prompt = f"Original question: {question}\nCorrect answer: {answer.strip()}\nGenerate {n} paraphrases."
    raw = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": PARAPHRASE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.8,
        max_tokens=800,
    )
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return []
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return [q.strip() for q in items if isinstance(q, str) and q.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--n-paraphrases", type=int, default=5)
    ap.add_argument("--gen-model", default=DEFAULT_GEN_MODEL)
    ap.add_argument("--held-out-fraction", type=float, default=0.15,
                     help="Fraction of NEW paraphrases assigned to "
                          "qna_split=held_out (rest go to train, since the "
                          "goal here is more train-time exposure per fact).")
    args = ap.parse_args()

    ensure_dirs()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set (expected in .env)")
    client = CachedClient(OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key), cache_dir=CACHE_DIR)

    qna_path = QNA_DIR / "qna.jsonl"
    rows = [json.loads(l) for l in qna_path.open()]
    doc_rows = [r for r in rows if r["doc_id"] == args.doc_id and r["qna_split"] == "train"]
    print(f"Found {len(doc_rows)} existing train QA pairs for {args.doc_id}", flush=True)

    import random
    rng = random.Random(3407)
    new_rows = []
    for r in doc_rows:
        question = r["question"] if r.get("question") else r["prefix"].removeprefix("Q: ").removesuffix("\nA:")
        paraphrases = gen_paraphrases(client, args.gen_model, question, r["target"], args.n_paraphrases)
        for pq in paraphrases:
            qna_split = "held_out" if rng.random() < args.held_out_fraction else "train"
            new_rows.append({
                "doc_id": r["doc_id"], "doc_version": r["doc_version"], "split": r["split"],
                "qna_split": qna_split, "question": pq,
                "prefix": f"Q: {pq}\nA:", "target": r["target"],
            })
        print(f"  {question[:60]!r} -> {len(paraphrases)} paraphrases", flush=True)

    with qna_path.open("a") as f:
        for row in new_rows:
            f.write(json.dumps(row) + "\n")
    print(f"\nAppended {len(new_rows)} paraphrased QA pairs to {qna_path}", flush=True)


if __name__ == "__main__":
    main()
