#!/usr/bin/env python3
"""Train the Memory-LoRA hypernetwork on google/gemma-4-E2B.

Forked from Code2LoRA's ``hypernetwork/train_code2lora_static_v2.py``
(direct-projection trainer), retargeted:

  * repo embedding  -> doc embedding   (memory_lora/encoder.py output)
  * Qwen2.5-Coder    -> google/gemma-4-E2B (memory_lora/core.py target modules)
  * cuda + flash_attn2 -> mps + sdpa (falls back to eager)
  * no wandb/TRL      -> plain PyTorch loop + TensorBoard (SummaryWriter)

Same core trick as the paper: only the hypernetwork head is trained; the
base LLM is frozen (gradient-checkpointed for memory headroom); LoRA A/B
tensors are non-detached so the causal-LM loss's backward graph flows
straight into the head's parameters.

Usage:
    python scripts/train_memory_lora.py --output-dir runs/pilot1 \\
        --limit-train-docs 5 --epochs 1        # smoke test

    python scripts/train_memory_lora.py --output-dir runs/full1
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForImageTextToText,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))
from memory_lora.data_paths import EMBEDDINGS_DIR, QNA_DIR, RUNS_DIR, ensure_dirs  # noqa: E402
from memory_lora.core import (  # noqa: E402
    MemoryLoRAHead,
    DEFAULT_ROOT_PREFIX,
    discover_module_types_and_dims,
    get_module_specs,
    inject_lora_weights,
    load_doc_rows,
    load_qna_rows,
    replace_with_lora,
)

DEFAULT_MODEL = "google/gemma-4-E2B"
DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "up_proj", "gate_proj", "down_proj",
]


# ---------------------------------------------------------------------------
# Dataset & batching
# ---------------------------------------------------------------------------

class DocDataset:
    """One example = one document with its train-split QnAs."""

    def __init__(
        self,
        docs_by_id: Dict[str, Dict[str, Any]],
        qnas_by_doc: Dict[str, List[Dict[str, str]]],
        doc_ids: List[str],
        max_qna_per_doc: int = 32,
        seed: int = 3407,
    ):
        self.doc_ids = list(doc_ids)
        self.docs = docs_by_id
        self.qnas = qnas_by_doc
        self.max_qna = max_qna_per_doc
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.doc_ids)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        d = self.doc_ids[idx]
        pairs = list(self.qnas.get(d, []))
        if not pairs:
            return None
        if len(pairs) > self.max_qna:
            pairs = self.rng.sample(pairs, self.max_qna)
        return {"doc_id": d, "embedding": self.docs[d]["emb"], "qnas": pairs}


def _tokenize_lm_batch(tokenizer, prefixes: List[str], targets: List[str],
                        max_seq_len: int = 2048,
                        fixed_len: bool = False) -> Dict[str, torch.Tensor]:
    """Causal-LM batch with the loss masked on prefix tokens. Keeps the
    rightmost prefix tokens on overflow; targets are never truncated.

    fixed_len: pad every batch to EXACTLY max_seq_len instead of the
    batch's own local max length. Real code prefixes vary widely (100-1024+
    tokens across different repos), so per-batch padding produces a new
    tensor shape almost every document. On MPS this repeatedly triggered
    unbounded memory growth (observed: a run that stayed under 13GB on
    homogeneous-length synthetic docs hit 70+GB and got OS-killed within
    ~10 documents of real, variable-length code) -- MPS's caching allocator
    does not appear to reliably reclaim/reuse blocks across many distinct
    shapes the way CUDA's does. Fixing every batch to one shape avoids the
    allocator ever seeing a new size after the first batch. Slightly wastes
    compute on padding for short sequences; that trade is worth it for
    system stability.
    """
    eos = tokenizer.eos_token or ""
    input_ids_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
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
    local_max = max(t.size(0) for t in input_ids_list)
    L = max(max_seq_len, local_max) if fixed_len else local_max
    pad_id = tokenizer.pad_token_id or 0

    def _lpad(x, val):
        return F.pad(x, (L - x.size(0), 0), value=val)

    input_ids = torch.stack([_lpad(t, pad_id) for t in input_ids_list], 0)
    labels = torch.stack([_lpad(t, -100) for t in labels_list], 0)
    attn_list = [torch.ones(t.size(0), dtype=torch.long) for t in input_ids_list]
    attn = torch.stack([_lpad(t, 0) for t in attn_list], 0)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_suite(
    base_model: nn.Module, head: MemoryLoRAHead, specs, tokenizer,
    doc_rows: List[Any], qnas_by_doc: Dict[str, List[Dict[str, str]]],
    *, device: torch.device, max_seq_len: int = 512,
    lm_micro_batch: int = 4, max_qna_per_doc: int = 32,
    fixed_len: bool = False, with_baseline: bool = False,
) -> Dict[str, float]:
    """Evaluate the adapted model, and (with ``with_baseline``) the SAME model
    with no adapter injected.

    The baseline is the metric that actually matters: an eval loss of 2.6 says
    nothing on its own, because it does not reveal whether the generated
    adapter is helping, doing nothing, or actively hurting. Tracking only the
    adapted loss is how a head that had collapsed to emitting one constant,
    worse-than-random adapter for every repo went unnoticed. ``delta`` below is
    the number to watch: it must go NEGATIVE and stay there.
    """
    base_model.eval()
    head.eval()
    total_loss = 0.0
    total_tokens = 0
    n_docs = 0
    base_loss_total = 0.0
    base_tokens = 0
    for dr in doc_rows:
        pairs = qnas_by_doc.get(dr.doc_id)
        if not pairs:
            continue
        if len(pairs) > max_qna_per_doc:
            pairs = pairs[:max_qna_per_doc]
        ctx = torch.from_numpy(dr.doc_embedding).to(device).unsqueeze(0)
        head_out = head(ctx)

        if with_baseline:
            # Detach the adapter (A=B=None -> LoRA.forward returns base output)
            # and score the identical batches through the frozen model.
            for sp in specs:
                m = dict(base_model.named_modules())[sp.full_name]
                m.A, m.B = None, None
            prefixes_b = [p["prefix"] for p in pairs]
            targets_b = [p["target"] for p in pairs]
            for i in range(0, len(prefixes_b), lm_micro_batch):
                j = min(i + lm_micro_batch, len(prefixes_b))
                b = _tokenize_lm_batch(tokenizer, prefixes_b[i:j], targets_b[i:j],
                                       max_seq_len=max_seq_len, fixed_len=fixed_len)
                if not b:
                    continue
                b = {k: v.to(device) for k, v in b.items()}
                o = base_model(**b)
                nt = (b["labels"] != -100).sum().item()
                base_loss_total += o.loss.item() * nt
                base_tokens += nt

        inject_lora_weights(base_model, specs, head_out, batch_index=0)
        prefixes = [p["prefix"] for p in pairs]
        targets = [p["target"] for p in pairs]
        for i in range(0, len(prefixes), lm_micro_batch):
            j = min(i + lm_micro_batch, len(prefixes))
            batch = _tokenize_lm_batch(tokenizer, prefixes[i:j], targets[i:j],
                                        max_seq_len=max_seq_len, fixed_len=fixed_len)
            if not batch:
                continue
            batch = {k: v.to(device) for k, v in batch.items()}
            out = base_model(**batch)
            loss = out.loss
            ntok = (batch["labels"] != -100).sum().item()
            total_loss += loss.item() * ntok
            total_tokens += ntok
        n_docs += 1
    avg = total_loss / max(total_tokens, 1)
    out: Dict[str, float] = {"eval_loss": avg, "n_docs": n_docs,
                             "n_tokens": total_tokens}
    if with_baseline and base_tokens:
        base_avg = base_loss_total / base_tokens
        out["baseline_loss"] = base_avg
        out["delta_vs_baseline"] = avg - base_avg  # negative == adapter helps
    return out


@torch.no_grad()
def adapter_input_sensitivity(head: MemoryLoRAHead, doc_rows, device,
                              n: int = 16) -> Dict[str, float]:
    """Mean pairwise cosine between the adapters generated for different repos.

    ~1.0 means the head ignores its input and emits one constant adapter (the
    failure mode that made a trained head score worse than random noise); low
    values mean the emitted adapter is genuinely repo-conditional.
    """
    rows = doc_rows[:n]
    if len(rows) < 2:
        return {}
    ctx = torch.from_numpy(
        np.stack([r.doc_embedding for r in rows])).to(device)
    o = head(ctx)
    t = sorted(o["A"].keys())[0]
    D = torch.einsum("nor,nri->noi", o["B"][t].float(), o["A"][t].float()).flatten(1)
    Dn = F.normalize(D, dim=1)
    C = Dn @ Dn.T
    iu = torch.triu_indices(len(D), len(D), 1)
    return {"adapter_cosine": float(C[iu[0], iu[1]].mean()),
            "adapter_delta_fro": float(D.norm(dim=1).mean())}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _docs_by_id(doc_rows) -> Dict[str, Dict[str, Any]]:
    return {dr.doc_id: {"emb": dr.doc_embedding} for dr in doc_rows}


def _group_qnas_by_doc(rows) -> Dict[str, List[Dict[str, str]]]:
    out: Dict[str, List[Dict[str, str]]] = {}
    for qr in rows:
        out.setdefault(qr.doc_id, []).append({"prefix": qr.prefix, "target": qr.target})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embeddings-path", default=str(EMBEDDINGS_DIR / "doc_embeddings.parquet"))
    ap.add_argument("--qna-path", default=str(QNA_DIR / "qna.jsonl"))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-name", default=DEFAULT_MODEL)
    ap.add_argument("--target-modules", nargs="+", default=DEFAULT_TARGET_MODULES)
    ap.add_argument("--root-prefix", default=DEFAULT_ROOT_PREFIX)

    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--head-hidden-dim", type=int, default=128,
                     help="Kept small deliberately -- with ~165 training "
                          "docs, the paper's 512-1024 hidden dim overfits "
                          "within ~2 epochs (see memory_lora/core.py docstring).")
    ap.add_argument("--head-dropout", type=float, default=0.1)

    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--lr-total-steps", type=int, default=0,
                     help="Override the cosine LR schedule's total-step "
                          "target with a realistic estimate of what "
                          "--max-hours will actually cover, instead of "
                          "steps_per_epoch * epochs (which assumes the run "
                          "finishes a full epoch -- unrealistic at "
                          "tens-of-thousands-of-docs scale). 0 = use the "
                          "epoch-based calculation.")
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--early-stop-patience", type=int, default=8,
                     help="Stop after this many consecutive evals with no "
                          "improvement on --primary-eval-suite. 0 = disabled.")

    ap.add_argument("--max-qna-per-doc", type=int, default=32)
    ap.add_argument("--lm-micro-batch", type=int, default=4)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--fixed-seq-len", action="store_true", default=True,
                     help="Pad every batch to exactly --max-seq-len instead "
                          "of each batch's own local max length. See "
                          "_tokenize_lm_batch docstring: on MPS, varying "
                          "tensor shapes across many real-code documents "
                          "of wildly different lengths caused unbounded "
                          "memory growth (a run went from healthy to "
                          "OS-killed within ~10 documents). Costs some "
                          "wasted padding compute; worth it for stability.")
    ap.add_argument("--no-fixed-seq-len", dest="fixed_seq_len", action="store_false")

    ap.add_argument("--eval-every-steps", type=int, default=50)
    ap.add_argument("--eval-suites", nargs="+", default=["cr_val", "cr_test", "ir_test"])
    ap.add_argument("--limit-eval-docs", type=int, default=200,
                     help="Cap docs per eval suite (random sample, fixed "
                          "seed) for speed at real-corpus scale -- e.g. "
                          "cr_val alone can be 8,600+ real repo-commit "
                          "docs; evaluating all of them every eval cycle "
                          "would dominate wall-clock time. 0 = no cap "
                          "(matches the paper's own --limit-eval-snapshots).")
    ap.add_argument("--primary-eval-suite", default="cr_val")
    ap.add_argument("--log-every-iters", type=int, default=10)

    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--attn-implementation", default="sdpa", choices=["sdpa", "eager"])
    ap.add_argument("--limit-train-docs", type=int, default=0)
    ap.add_argument("--priority-doc-ids", nargs="+", default=[],
                     help="doc_ids to oversample within the shared "
                          "multi-document head (--priority-oversample "
                          "extra passes per epoch), instead of raising "
                          "global head capacity -- raising rank/hidden_dim "
                          "fixes within-document fact interference but "
                          "makes the whole corpus overfit faster (see "
                          "runs/full2 diagnosis); oversampling gives "
                          "specific documents more gradient signal without "
                          "changing capacity or hurting the rest.")
    ap.add_argument("--priority-oversample", type=int, default=5,
                     help="How many times to repeat each --priority-doc-ids "
                          "entry per epoch's shuffled training order.")
    ap.add_argument("--only-doc-ids", nargs="+", default=[],
                     help="Restrict training (and, if present in this set, "
                          "ir_test eval) to exactly these doc_ids. Used for "
                          "single-document capacity diagnostics -- e.g. can "
                          "the architecture memorize ONE document's facts "
                          "when not sharing hypernetwork capacity across "
                          "165 others?")
    ap.add_argument("--gradient-checkpointing", action="store_true", default=True)
    ap.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    ap.add_argument("--max-hours", type=float, default=0.0,
                     help="Wall-clock training budget in hours. 0 = unlimited "
                          "(stop only after --epochs). Checked once per doc "
                          "iteration; when exceeded, saves a final checkpoint "
                          "and stops cleanly (does not just get killed mid-write).")
    ap.add_argument("--min-available-gb", type=float, default=15.0,
                     help="Hard safety floor: stop (with a final "
                          "checkpoint) if SYSTEM-WIDE available memory "
                          "(psutil.virtual_memory().available -- NOT this "
                          "process's own RSS, which undercounts MPS "
                          "memory on Apple Silicon) drops below this many "
                          "GB. Checked every 2 iterations. 0 = disabled.")
    ap.add_argument("--checkpoint-every-steps", type=int, default=50,
                     help="Overwrite head.latest.pt every N optimizer steps "
                          "so a crash/kill never loses more than N steps of "
                          "progress. Overwrites (doesn't accumulate files), "
                          "so it's disk-safe even for a 3GB head. 0=disabled.")
    ap.add_argument("--checkpoint-every-minutes", type=float, default=30.0,
                     help="Save a timestamped checkpoint every N minutes of "
                          "wall-clock time, independent of eval/epoch "
                          "boundaries. 0 = disabled (epoch-end saves only).")
    ap.add_argument("--epoch-ckpt-every", type=int, default=10,
                     help="Only write a NEW numbered head.epN.pt every N "
                          "epochs (head.latest.pt still updates every "
                          "epoch). Each checkpoint is a full head save "
                          "(hundreds of MB) -- with many small/fast epochs "
                          "(e.g. a tiny single-document run), saving one "
                          "per epoch can fill the disk in minutes.")
    ap.add_argument("--no-eval-baseline", action="store_true",
                    help="skip the no-adapter baseline during eval (faster, but "
                         "you lose the only signal that says whether the "
                         "adapter is actually helping)")
    ap.add_argument("--resume-from", default="",
                     help="Path to a head.*.pt checkpoint to load weights "
                          "from before training starts (optimizer/scheduler "
                          "restart fresh; only head weights are resumed).")
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

    tb = SummaryWriter(log_dir=str(out_dir / "tb"))

    # ---- Load embeddings + QnAs ----
    print("Loading document embeddings ...", flush=True)
    all_docs = load_doc_rows(Path(args.embeddings_path))
    only_ids = set(args.only_doc_ids) if args.only_doc_ids else None
    train_docs = [d for d in all_docs if d.split == "train"]
    if only_ids:
        train_docs = [d for d in train_docs if d.doc_id in only_ids]
    if args.limit_train_docs:
        train_docs = train_docs[: args.limit_train_docs]
    print(f"  {len(train_docs)} train docs (of {len(all_docs)} total)", flush=True)

    print("Loading QnAs ...", flush=True)
    all_qnas = load_qna_rows(Path(args.qna_path))
    if only_ids:
        all_qnas = [q for q in all_qnas if q.doc_id in only_ids]
    train_qnas = [q for q in all_qnas if q.qna_split == "train"]
    qnas_train = _group_qnas_by_doc(train_qnas)
    docs_by_id = _docs_by_id(train_docs)
    doc_ids = [d for d in docs_by_id if d in qnas_train]
    print(f"  {sum(len(v) for v in qnas_train.values())} train QA pairs across {len(doc_ids)} docs", flush=True)

    if args.priority_doc_ids and args.priority_oversample > 1:
        extra = []
        for pid in args.priority_doc_ids:
            if pid in doc_ids:
                extra.extend([pid] * (args.priority_oversample - 1))
            else:
                print(f"  [warn] --priority-doc-ids {pid!r} not in training set, skipping", flush=True)
        doc_ids = doc_ids + extra
        print(f"  oversampled {args.priority_doc_ids} x{args.priority_oversample} "
              f"-> {len(doc_ids)} entries/epoch", flush=True)

    ds = DocDataset(docs_by_id, qnas_train, doc_ids,
                     max_qna_per_doc=args.max_qna_per_doc, seed=args.seed)

    # ---- Build LLM, discover modules, wrap them ----
    print(f"Loading {args.model_name} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForImageTextToText.from_pretrained(
        args.model_name, torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
    ).to(device)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False
    if args.gradient_checkpointing:
        base_model.config.use_cache = False
        try:
            base_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            print("  gradient checkpointing: ON", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] gradient checkpointing unavailable: {e}", flush=True)

    specs = get_module_specs(base_model, args.target_modules, root_prefix=args.root_prefix)
    type_dims = discover_module_types_and_dims(specs)
    print(f"  discovered {len(specs)} target modules, {len(type_dims)} types: {sorted(type_dims)}", flush=True)
    if not specs:
        raise SystemExit(
            f"No modules matched root_prefix={args.root_prefix!r} + "
            f"{args.target_modules}. Inspect base_model.named_modules() and "
            f"pass --root-prefix explicitly."
        )
    replace_with_lora(base_model, specs, rank=args.rank, alpha=args.alpha)

    head = MemoryLoRAHead(
        input_dim=train_docs[0].doc_embedding.shape[0],
        type_dims=type_dims,
        hidden_dim=args.head_hidden_dim,
        rank=args.rank,
        dropout=args.head_dropout,
    ).to(device)
    # Standardize the conditioning input using TRAIN docs only. Without this the
    # ~64% DC component shared by all repo embeddings dominates the trunk and it
    # emits a near-identical adapter for every repo (see MemoryLoRAHead docs).
    head.fit_input_stats(torch.from_numpy(
        np.stack([d.doc_embedding for d in train_docs])).to(device))
    print(f"  fitted input standardization over {len(train_docs)} train docs",
          flush=True)
    if args.resume_from:
        # weights_only=False: these checkpoints carry the run's config/args dicts,
    # not just tensors, and torch>=2.6 defaults the strict unpickler on --
    # which rejects them ("Unsupported operand"). They are produced by this
    # project's own training script, so loading them fully is intended.
    ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        # strict=False: checkpoints written before input standardization
        # existed carry no input_mean/input_std, so the stats fitted above are
        # kept. A checkpoint that does carry them overwrites the fresh fit,
        # which is what a resumed run wants -- the transform must not change
        # mid-training.
        missing, _ = head.load_state_dict(ckpt["state_dict"], strict=False)
        if missing:
            print(f"  (new buffers not in checkpoint: {missing})", flush=True)
        print(f"  resumed head weights from {args.resume_from}", flush=True)
    n_head_params = sum(p.numel() for p in head.parameters())
    print(f"  head params: {n_head_params / 1e6:.1f}M", flush=True)

    optim = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(ds))
    if args.lr_total_steps:
        # At real-corpus scale (tens of thousands of docs), --max-hours will
        # cut training off long before steps_per_epoch * epochs is reached,
        # so a schedule calibrated to full-epoch coverage would barely start
        # annealing from its LR peak. Calibrate to the realistically
        # achievable step count instead.
        total_steps = args.lr_total_steps
    else:
        total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    sched = get_cosine_schedule_with_warmup(optim, warmup_steps, total_steps)

    # ---- Eval suites ----
    eval_suites: Dict[str, Dict[str, Any]] = {}
    print("Loading eval suites ...", flush=True)
    qnas_by_doc_all = _group_qnas_by_doc(all_qnas)
    qnas_held_out_by_doc = _group_qnas_by_doc([q for q in all_qnas if q.qna_split == "held_out"])
    eval_rng = random.Random(args.seed)
    for suite in args.eval_suites:
        if suite in ("cr_val", "cr_test"):
            rows = [d for d in all_docs if d.split == suite]
            q_by_doc = qnas_by_doc_all
        elif suite == "ir_test":
            rows = train_docs
            q_by_doc = qnas_held_out_by_doc
        else:
            continue
        if args.limit_eval_docs and len(rows) > args.limit_eval_docs:
            rows = eval_rng.sample(rows, args.limit_eval_docs)
        eval_suites[suite] = {"doc_rows": rows, "qnas_by_doc": q_by_doc}
        n_q = sum(len(q_by_doc.get(d.doc_id, [])) for d in rows)
        print(f"  {suite}: {len(rows)} docs, {n_q} qnas", flush=True)

    # ---- Train ----
    metrics_log: List[Dict[str, Any]] = []
    best_eval = float("inf")
    global_step = 0
    t0 = time.time()
    last_ckpt_wall = t0
    budget_seconds = args.max_hours * 3600.0 if args.max_hours > 0 else float("inf")
    ckpt_interval_seconds = args.checkpoint_every_minutes * 60.0 if args.checkpoint_every_minutes > 0 else float("inf")
    stop_training = False
    patience_counter = 0
    for epoch in range(args.epochs):
        if stop_training:
            break
        order = list(range(len(ds)))
        random.shuffle(order)
        head.train()
        running_loss, running_n = 0.0, 0
        for it, di in enumerate(order):
            now = time.time()
            if now - t0 >= budget_seconds:
                print(f"  [budget] {args.max_hours:.2f}h training budget reached "
                      f"(epoch {epoch}, it {it}/{len(order)}) -- stopping.", flush=True)
                stop_training = True
                break
            if now - last_ckpt_wall >= ckpt_interval_seconds:
                mins = int((now - t0) / 60)
                p = _save_ckpt(out_dir, head, type_dims, args, name=f"t{mins:04d}m")
                _save_ckpt(out_dir, head, type_dims, args, name="latest")
                print(f"  [ckpt] periodic ({args.checkpoint_every_minutes:.0f}min interval) -> {p}", flush=True)
                last_ckpt_wall = now
            if args.min_available_gb > 0 and it % 2 == 0:
                # IMPORTANT: this checks SYSTEM-WIDE available memory
                # (psutil.virtual_memory), not this process's own RSS.
                # psutil.Process().memory_info().rss -- like `ps -o rss` --
                # does NOT reliably capture MPS/GPU-resident allocations
                # on Apple Silicon: a run was observed at 55-83GB actual
                # usage (per `top`'s MEM column, corroborated by system
                # vm_stat showing genuine memory exhaustion) while RSS
                # reported under 1GB the whole time. System-wide available
                # memory is the metric that's actually reliable here.
                available_gb = psutil.virtual_memory().available / 1e9
                if available_gb < args.min_available_gb:
                    print(f"  [safety] system available memory {available_gb:.1f}GB "
                          f"below --min-available-gb {args.min_available_gb:.1f}GB "
                          f"(epoch {epoch}, it {it}) -- saving and stopping to "
                          f"protect system stability.", flush=True)
                    _save_ckpt(out_dir, head, type_dims, args, name="latest")
                    stop_training = True
                    break
            sample = ds[di]
            if sample is None:
                continue
            ctx = torch.from_numpy(sample["embedding"]).to(device).unsqueeze(0)
            qnas = sample["qnas"]
            prefixes = [q["prefix"] for q in qnas]
            targets = [q["target"] for q in qnas]
            micro_batches = []
            for i in range(0, len(prefixes), args.lm_micro_batch):
                j = min(i + args.lm_micro_batch, len(prefixes))
                b = _tokenize_lm_batch(tokenizer, prefixes[i:j], targets[i:j],
                                        max_seq_len=args.max_seq_len, fixed_len=args.fixed_seq_len)
                if b:
                    micro_batches.append({k: v.to(device) for k, v in b.items()})
            if not micro_batches:
                continue
            n_tok_seen, loss_acc = 0, 0.0
            for mb_idx, batch in enumerate(micro_batches):
                if args.min_available_gb > 0 and mb_idx % 3 == 0:
                    # Same system-wide check as the per-document one below,
                    # but INSIDE the micro-batch loop too: observed runaway
                    # growth can blow past a safe threshold within a
                    # single document's micro-batches, before the
                    # per-document check would ever fire.
                    available_gb = psutil.virtual_memory().available / 1e9
                    if available_gb < args.min_available_gb:
                        print(f"  [safety] system available memory {available_gb:.1f}GB "
                              f"below --min-available-gb {args.min_available_gb:.1f}GB "
                              f"mid-document (epoch {epoch}, it {it}, micro-batch "
                              f"{mb_idx}) -- saving and stopping immediately.", flush=True)
                        _save_ckpt(out_dir, head, type_dims, args, name="latest")
                        stop_training = True
                        break
                head_out = head(ctx)
                inject_lora_weights(base_model, specs, head_out, batch_index=0)
                out = base_model(**batch)
                ntok = (batch["labels"] != -100).sum().item()
                loss = out.loss * ntok
                loss.backward()
                loss_acc += loss.detach().item()
                n_tok_seen += ntok
                del head_out, out, loss
            if stop_training:
                break
            if n_tok_seen == 0:
                continue
            if device.type == "mps" and it % 5 == 0:
                # MPS's caching allocator is markedly less aggressive about
                # returning freed blocks to the OS than CUDA's -- on a
                # unified-memory Mac (CPU and GPU share physical RAM,
                # unlike a discrete-GPU box with isolated VRAM) that cache
                # growth directly threatens the whole system, not just this
                # process. Without this, a real-corpus run OOM'd the OS
                # itself (83GB RSS, process state "stuck", heavy swapping)
                # within the first ~10 minutes.
                torch.mps.empty_cache()

            torch.nn.utils.clip_grad_norm_(head.parameters(), args.max_grad_norm)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            global_step += 1

            if args.checkpoint_every_steps > 0 and global_step % args.checkpoint_every_steps == 0:
                _save_ckpt(out_dir, head, type_dims, args, name="latest")

            running_loss += loss_acc
            running_n += n_tok_seen
            if it % max(1, args.log_every_iters) == 0:
                avg = running_loss / max(running_n, 1)
                elapsed = (time.time() - t0) / 60
                print(f"[ep{epoch} it{it}/{len(order)} step{global_step}] "
                      f"loss={avg:.4f} lr={sched.get_last_lr()[0]:.2e} elapsed={elapsed:.1f}m", flush=True)
                tb.add_scalar("train/loss", avg, global_step)
                tb.add_scalar("train/lr", sched.get_last_lr()[0], global_step)
                running_loss, running_n = 0.0, 0

            if (args.eval_every_steps > 0 and global_step > 0
                    and global_step % args.eval_every_steps == 0
                    and it + 1 != len(order)):
                # The `it + 1 != len(order)` guard skips a redundant eval when
                # eval_every_steps happens to equal (a multiple of) steps-per-
                # epoch -- the unconditional end-of-epoch eval below would
                # otherwise double-count this exact step in the early-stop
                # patience counter every epoch.
                prev_best = best_eval
                best_eval = _do_eval(args, base_model, head, specs, tokenizer, eval_suites,
                                      device, out_dir, metrics_log, best_eval, global_step, epoch, tb)
                patience_counter = 0 if best_eval < prev_best else patience_counter + 1
                if args.early_stop_patience > 0 and patience_counter >= args.early_stop_patience:
                    print(f"  [early-stop] no improvement on {args.primary_eval_suite} for "
                          f"{patience_counter} evals -- stopping.", flush=True)
                    stop_training = True
                    break

        _save_ckpt(out_dir, head, type_dims, args, name="latest")
        if epoch % max(1, args.epoch_ckpt_every) == 0:
            ep_path = _save_ckpt(out_dir, head, type_dims, args, name=f"ep{epoch}")
            print(f"  [ckpt] end-of-epoch ep{epoch} -> {ep_path}", flush=True)
        else:
            print(f"  [ckpt] end-of-epoch ep{epoch} -> (latest.pt only)", flush=True)
        prev_best = best_eval
        best_eval = _do_eval(args, base_model, head, specs, tokenizer, eval_suites,
                              device, out_dir, metrics_log, best_eval, global_step, epoch, tb,
                              end_of_epoch=True)
        patience_counter = 0 if best_eval < prev_best else patience_counter + 1
        if args.early_stop_patience > 0 and patience_counter >= args.early_stop_patience:
            print(f"  [early-stop] no improvement on {args.primary_eval_suite} for "
                  f"{patience_counter} evals -- stopping.", flush=True)
            stop_training = True

    tb.close()
    print(f"\nTraining done. Best primary eval = {best_eval:.4f}", flush=True)


def _save_ckpt(out_dir: Path, head: MemoryLoRAHead, type_dims, args, name: str = "latest") -> Path:
    out = out_dir / f"head.{name}.pt"
    torch.save({
        "state_dict": head.state_dict(),
        "config": head.config_dict(),
        "type_dims": type_dims,
        "args": vars(args),
    }, out)
    return out


def _do_eval(args, base_model, head, specs, tokenizer, eval_suites, device, out_dir,
             metrics_log, best_eval, global_step, epoch, tb, end_of_epoch: bool = False) -> float:
    suite_metrics: Dict[str, Dict[str, float]] = {}
    for name, suite in eval_suites.items():
        m = evaluate_suite(base_model, head, specs, tokenizer,
                            suite["doc_rows"], suite["qnas_by_doc"],
                            device=device, max_seq_len=args.max_seq_len,
                            lm_micro_batch=args.lm_micro_batch,
                            max_qna_per_doc=args.max_qna_per_doc,
                            fixed_len=args.fixed_seq_len,
                            with_baseline=not args.no_eval_baseline)
        suite_metrics[name] = m
        delta_s = ""
        if "delta_vs_baseline" in m:
            verdict = "HELPS" if m["delta_vs_baseline"] < 0 else "HURTS"
            delta_s = (f" | base={m['baseline_loss']:.4f} "
                       f"delta={m['delta_vs_baseline']:+.4f} {verdict}")
        print(f"  [eval {name}] step={global_step} loss={m['eval_loss']:.4f} "
              f"docs={m['n_docs']} tok={m['n_tokens']}{delta_s}", flush=True)
        tb.add_scalar(f"eval/{name}_loss", m["eval_loss"], global_step)
        if "baseline_loss" in m:
            tb.add_scalar(f"eval/{name}_baseline_loss", m["baseline_loss"], global_step)
            # THE metric to watch: must be negative for the adapter to be useful.
            tb.add_scalar(f"eval/{name}_delta_vs_baseline",
                          m["delta_vs_baseline"], global_step)

    # Input-sensitivity diagnostic: is the head emitting repo-conditional
    # adapters, or one constant adapter regardless of input?
    any_suite = next(iter(eval_suites.values()), None)
    if any_suite:
        diag = adapter_input_sensitivity(head, any_suite["doc_rows"], device)
        for k, v in diag.items():
            tb.add_scalar(f"diag/{k}", v, global_step)
        if diag:
            print(f"  [diag] adapter_cosine={diag['adapter_cosine']:.4f} "
                  f"(1.0 = same adapter for every repo) "
                  f"delta_fro={diag['adapter_delta_fro']:.3f}", flush=True)
        suite_metrics["_diag"] = diag
    primary = suite_metrics.get(args.primary_eval_suite)
    primary_loss = primary["eval_loss"] if primary else float("inf")
    row = {"step": global_step, "epoch": epoch, "end_of_epoch": end_of_epoch,
           "eval_loss": primary_loss, "suites": suite_metrics}
    metrics_log.append(row)
    (out_dir / "metrics.jsonl").open("a").write(json.dumps(row) + "\n")
    if primary_loss < best_eval:
        best_eval = primary_loss
        p = _save_ckpt(out_dir, head, head.type_dims, args, name="best")
        print(f"  [ckpt] best updated -> {p} (loss={primary_loss:.4f})", flush=True)
    _save_ckpt(out_dir, head, head.type_dims, args, name="latest")
    head.train()
    return best_eval


if __name__ == "__main__":
    main()
