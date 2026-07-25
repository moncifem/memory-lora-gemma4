#!/usr/bin/env python3
"""Train a standalone, directly-parameterized LoRA adapter on priority
document(s) -- NOT hypernetwork-generated.

Why this exists (see runs/full1, runs/full2, runs/full3_priority logs):
a single hypernetwork trunk shared across 165+ documents cannot
simultaneously (a) generalize broadly across the corpus and (b) reliably
memorize any one document's specific facts -- three separate interventions
(bigger head, oversampling) each improved (b) only by making (a) collapse
faster, because they all perturb the SAME shared trunk weights.

This script sidesteps the tension entirely for content we don't need
zero-shot generalization on (documents we already have and specifically
want memorized, e.g. the Code2LoRA paper): instead of a hypernetwork
mapping embedding->weights, the LoRA A/B matrices are ordinary
``nn.Parameter`` tensors trained directly via backprop on exactly the
QA pairs for the selected doc_id(s) -- standard PEFT-style fine-tuning,
with the same LoRA wrapper class (``memory_lora.core.LoRA``) and the same
frozen Gemma-4-E2B target modules as the hypernetwork path, so results are
directly comparable and the diagnostic single-doc runs (which proved this
converges to correct multi-fact recall) apply unchanged.

Usage:
    python scripts/train_direct_lora.py --output-dir direct_paper \\
        --doc-ids code2lora_paper --epochs 300
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoTokenizer, get_cosine_schedule_with_warmup

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))
from memory_lora.data_paths import QNA_DIR, RUNS_DIR, ensure_dirs  # noqa: E402
from memory_lora.core import (  # noqa: E402
    DEFAULT_ROOT_PREFIX,
    discover_module_types_and_dims,
    get_module_specs,
    load_qna_rows,
    replace_with_lora,
)

DEFAULT_MODEL = "google/gemma-4-E2B"
DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "up_proj", "gate_proj", "down_proj",
]


def _tokenize_lm_batch(tokenizer, prefixes: List[str], targets: List[str],
                        max_seq_len: int = 384) -> Dict[str, torch.Tensor]:
    eos = tokenizer.eos_token or ""
    input_ids_list, labels_list = [], []
    for p, t in zip(prefixes, targets):
        t_ids = tokenizer(t + eos, add_special_tokens=False)["input_ids"]
        if not t_ids:
            continue
        prefix_budget = max(8, max_seq_len - len(t_ids))
        p_ids_full = tokenizer(p, add_special_tokens=False)["input_ids"]
        p_ids = p_ids_full[-prefix_budget:] if len(p_ids_full) > prefix_budget else p_ids_full
        ids = p_ids + t_ids
        labels = ([-100] * len(p_ids)) + list(t_ids)
        input_ids_list.append(torch.tensor(ids, dtype=torch.long))
        labels_list.append(torch.tensor(labels, dtype=torch.long))
    if not input_ids_list:
        return {}
    L = max(t.size(0) for t in input_ids_list)
    pad_id = tokenizer.pad_token_id or 0

    def _lpad(x, val):
        return F.pad(x, (L - x.size(0), 0), value=val)

    input_ids = torch.stack([_lpad(t, pad_id) for t in input_ids_list], 0)
    labels = torch.stack([_lpad(t, -100) for t in labels_list], 0)
    attn_list = [torch.ones(t.size(0), dtype=torch.long) for t in input_ids_list]
    attn = torch.stack([_lpad(t, 0) for t in attn_list], 0)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}


def init_direct_lora_params(model: nn.Module, specs, rank: int, device, dtype) -> Dict[str, Dict[str, nn.Parameter]]:
    """Give each LoRA wrapper its OWN trainable (A, B), instead of an
    externally-injected tensor from a hypernetwork. B initialized to zero
    (standard LoRA init) so the adapter starts as a no-op."""
    named = dict(model.named_modules())
    params: Dict[str, Dict[str, nn.Parameter]] = {}
    for sp in specs:
        lora_mod = named[sp.full_name]
        A = nn.Parameter(torch.randn(rank, sp.in_features, device=device, dtype=torch.float32) * 0.01)
        B = nn.Parameter(torch.zeros(sp.out_features, rank, device=device, dtype=torch.float32))
        lora_mod.set_lora_weights(A, B)
        params[sp.full_name] = {"A": A, "B": B}
    return params


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qna-path", default=str(QNA_DIR / "qna.jsonl"))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-name", default=DEFAULT_MODEL)
    ap.add_argument("--target-modules", nargs="+", default=DEFAULT_TARGET_MODULES)
    ap.add_argument("--root-prefix", default=DEFAULT_ROOT_PREFIX)
    ap.add_argument("--doc-ids", nargs="+", default=[],
                     help="Train a single direct LoRA jointly on the union "
                          "of these documents' QnAs (use one doc_id for a "
                          "dedicated per-document adapter).")
    ap.add_argument("--doc-ids-file", default="",
                     help="Alternative to --doc-ids: path to a file with "
                          "whitespace-separated doc_ids. Avoids shell "
                          "word-splitting pitfalls (e.g. zsh does not "
                          "word-split unquoted $VAR by default) for large "
                          "doc-id lists.")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=64.0)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--max-seq-len", type=int, default=384)
    ap.add_argument("--lm-micro-batch", type=int, default=4)
    ap.add_argument("--eval-every-epochs", type=int, default=10)
    ap.add_argument("--epoch-ckpt-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--attn-implementation", default="sdpa", choices=["sdpa", "eager"])
    ap.add_argument("--gradient-checkpointing", action="store_true", default=True)
    ap.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = RUNS_DIR / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_dirs()

    device = torch.device(args.device if (args.device != "mps" or torch.backends.mps.is_available()) else "cpu")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.doc_ids_file:
        file_ids = Path(args.doc_ids_file).read_text().split()
        doc_ids = set(args.doc_ids) | set(file_ids)
    else:
        doc_ids = set(args.doc_ids)
    if not doc_ids:
        raise SystemExit("No --doc-ids or --doc-ids-file provided.")
    print(f"Training a direct (non-hypernetwork) LoRA jointly on: {sorted(doc_ids)}", flush=True)
    all_qnas = load_qna_rows(Path(args.qna_path))
    train_qnas = [q for q in all_qnas if q.doc_id in doc_ids and q.qna_split == "train"]
    held_out_qnas = [q for q in all_qnas if q.doc_id in doc_ids and q.qna_split == "held_out"]
    print(f"  {len(train_qnas)} train QAs, {len(held_out_qnas)} held-out QAs", flush=True)
    if not train_qnas:
        raise SystemExit("No train QnAs found for the given --doc-ids.")

    print(f"Loading {args.model_name} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForImageTextToText.from_pretrained(
        args.model_name, torch_dtype=dtype, attn_implementation=args.attn_implementation,
    ).to(device)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False
    if args.gradient_checkpointing:
        base_model.config.use_cache = False
        base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        print("  gradient checkpointing: ON", flush=True)

    specs = get_module_specs(base_model, args.target_modules, root_prefix=args.root_prefix)
    type_dims = discover_module_types_and_dims(specs)
    print(f"  discovered {len(specs)} target modules, {len(type_dims)} shape-types", flush=True)
    replace_with_lora(base_model, specs, rank=args.rank, alpha=args.alpha)
    lora_params = init_direct_lora_params(base_model, specs, args.rank, device, dtype)
    all_params = [p for pair in lora_params.values() for p in pair.values()]
    n_params = sum(p.numel() for p in all_params)
    print(f"  direct LoRA trainable params: {n_params / 1e6:.2f}M "
          f"(rank={args.rank}, {len(specs)} modules)", flush=True)

    optim = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    sched = get_cosine_schedule_with_warmup(optim, warmup_steps, total_steps)

    def _save(name: str) -> Path:
        out = out_dir / f"lora.{name}.pt"
        state = {full_name: {"A": pair["A"].detach().cpu(), "B": pair["B"].detach().cpu()}
                  for full_name, pair in lora_params.items()}
        torch.save({"lora": state, "rank": args.rank, "alpha": args.alpha,
                     "doc_ids": sorted(doc_ids), "target_modules": args.target_modules,
                     "root_prefix": args.root_prefix}, out)
        return out

    @torch.no_grad()
    def _eval_held_out() -> float:
        if not held_out_qnas:
            return float("nan")
        base_model.eval()
        total_loss, total_tok = 0.0, 0
        prefixes = [q.prefix for q in held_out_qnas]
        targets = [q.target for q in held_out_qnas]
        for i in range(0, len(prefixes), args.lm_micro_batch):
            j = min(i + args.lm_micro_batch, len(prefixes))
            batch = _tokenize_lm_batch(tokenizer, prefixes[i:j], targets[i:j], max_seq_len=args.max_seq_len)
            if not batch:
                continue
            batch = {k: v.to(device) for k, v in batch.items()}
            out = base_model(**batch)
            ntok = (batch["labels"] != -100).sum().item()
            total_loss += out.loss.item() * ntok
            total_tok += ntok
        return total_loss / max(total_tok, 1)

    prefixes_all = [q.prefix for q in train_qnas]
    targets_all = [q.target for q in train_qnas]
    metrics_log: List[Dict[str, Any]] = []
    t0 = time.time()
    for epoch in range(args.epochs):
        order = list(range(len(prefixes_all)))
        random.shuffle(order)
        prefixes = [prefixes_all[i] for i in order]
        targets = [targets_all[i] for i in order]
        loss_acc, n_tok_seen = 0.0, 0
        for i in range(0, len(prefixes), args.lm_micro_batch):
            j = min(i + args.lm_micro_batch, len(prefixes))
            batch = _tokenize_lm_batch(tokenizer, prefixes[i:j], targets[i:j], max_seq_len=args.max_seq_len)
            if not batch:
                continue
            batch = {k: v.to(device) for k, v in batch.items()}
            out = base_model(**batch)
            ntok = (batch["labels"] != -100).sum().item()
            loss = out.loss * ntok
            loss.backward()
            loss_acc += loss.detach().item()
            n_tok_seen += ntok
        torch.nn.utils.clip_grad_norm_(all_params, args.max_grad_norm)
        optim.step()
        sched.step()
        optim.zero_grad(set_to_none=True)

        avg = loss_acc / max(n_tok_seen, 1)
        elapsed = (time.time() - t0) / 60
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"[ep{epoch}] train_loss={avg:.4f} lr={sched.get_last_lr()[0]:.2e} elapsed={elapsed:.1f}m", flush=True)

        if epoch % max(1, args.eval_every_epochs) == 0 or epoch == args.epochs - 1:
            held_out_loss = _eval_held_out()
            print(f"  [eval] ep{epoch} held_out_loss={held_out_loss:.4f}", flush=True)
            metrics_log.append({"epoch": epoch, "train_loss": avg, "held_out_loss": held_out_loss})
            (out_dir / "metrics.jsonl").open("a").write(json.dumps(metrics_log[-1]) + "\n")

        if epoch % max(1, args.epoch_ckpt_every) == 0 or epoch == args.epochs - 1:
            p = _save(f"ep{epoch}")
            print(f"  [ckpt] -> {p}", flush=True)
        _save("latest")

    print(f"\nDirect LoRA training done. Final train_loss={avg:.4f}", flush=True)


if __name__ == "__main__":
    main()
