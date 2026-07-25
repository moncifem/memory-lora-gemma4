#!/usr/bin/env python3
"""Shared building blocks for the Memory-LoRA hypernetwork.

Forked from Code2LoRA's ``hypernetwork/code2lora_core.py`` (Hotsko et al.,
"Code2LoRA: Hypernetwork-Generated Adapters for Code Language Models under
Software Evolution", MIT-licensed code release). Same core trick, different
target model and conditioning input:

* Code2LoRA:   repository embedding -> LoRA adapter for Qwen2.5-Coder-1.5B
* Memory-LoRA: document embedding   -> LoRA adapter for google/gemma-4-E2B

The ``MemoryLoRAHead`` (renamed from ``Code2LoRAHead``, architecture
unchanged) outputs ONE (A, B) pair per LoRA module *type* (q_proj, k_proj,
v_proj, o_proj, gate_proj, up_proj, down_proj), shared across every target
transformer layer -- not per-layer. This keeps the head's parameter count
tractable for local (MPS) training.

Gemma-4-E2B specifics (verified against the model's actual safetensors
header, not guessed):

* Decoder is nested at ``model.language_model.layers.{i}.*`` -- NOT
  ``model.layers.{i}.*`` like Qwen2.5-Coder. ``get_module_specs`` below
  matches on ``language_model\\.layers\\.(\\d+)\\.``, not ``model\\.layers``.
* Module type names are identical to Code2LoRA's defaults:
  q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj.
* Only ``model.language_model.*`` is ever touched. ``vision_tower`` and
  ``audio_tower`` are left completely alone -- irrelevant to text recall and
  risky to perturb.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pads
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# LoRA module + injection (unchanged from Code2LoRA -- architecture-agnostic)
# ---------------------------------------------------------------------------

class LoRA(nn.Module):
    """Wraps an ``nn.Linear`` with an additive low-rank update.

    Forward: ``y = base(x) + scaling * (x @ A^T) @ B^T``, where per-batch
    A: ``[rank, in_features]`` and B: ``[out_features, rank]`` come from an
    external hypernet via :meth:`set_lora_weights`.

    IMPORTANT autograd contract: A and B are kept as **plain attributes**,
    not buffers, and stored **without detaching**, so the LM loss's backward
    graph flows through them straight into the hypernet parameters that
    produced them. The base ``nn.Linear`` is frozen and its forward sees a
    detached copy of the input to avoid building an autograd graph through
    the (much larger) frozen LLM weights.
    """

    def __init__(self, base: nn.Linear, in_features: int, out_features: int,
                 rank: int, alpha: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = float(alpha) / float(max(1, rank))
        self.A: Optional[torch.Tensor] = None  # [rank, in_features]
        self.B: Optional[torch.Tensor] = None  # [out_features, rank]

    def set_lora_weights(self, A: torch.Tensor, B: torch.Tensor) -> None:
        self.A = A
        self.B = B

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.A is None or self.B is None:
            return y
        x_f32 = x.detach().to(torch.float32)
        A = self.A.to(torch.float32)
        B = self.B.to(torch.float32)
        delta = F.linear(F.linear(x_f32, A), B) * self.scaling
        return y + delta.to(dtype=y.dtype)


@dataclass
class ModuleSpec:
    full_name: str   # e.g. 'model.language_model.layers.5.self_attn.q_proj'
    layer_idx: int
    type: str        # e.g. 'q_proj'
    in_features: int
    out_features: int


# Gemma-4-E2B nests its text decoder here (verified via safetensors header).
DEFAULT_ROOT_PREFIX = "model.language_model."
_LAYER_IDX_RE = re.compile(r"\blanguage_model\.layers\.(\d+)\.")


def get_module_specs(model: nn.Module, target_module_types: List[str],
                      root_prefix: str = DEFAULT_ROOT_PREFIX
                      ) -> List[ModuleSpec]:
    """Discover every nn.Linear under ``root_prefix`` whose name contains one
    of ``target_module_types`` and return one :class:`ModuleSpec` per match,
    sorted by (layer_idx, full_name).

    Restricting to ``root_prefix`` is what keeps ``vision_tower`` /
    ``audio_tower`` untouched even though they also contain q_proj/k_proj/
    v_proj/o_proj-named linears.

    Gemma-4-E2B is architecturally heterogeneous across layers (unlike
    Qwen2.5-Coder, which Code2LoRA was built for): every 5th layer is a
    wider "full_attention" layer (q_proj/o_proj 2x the width of the
    "sliding_attention" layers), and 20 of the 35 layers have NO k_proj/
    v_proj at all -- they reuse an earlier layer's KV cache
    (``num_kv_shared_layers=20`` in the model config). A LoRA (A, B) pair
    can only be shared across modules of IDENTICAL shape, so ``.type`` here
    is ``"{module_name}_{in}x{out}"`` (shape-qualified), not just the raw
    module name -- e.g. ``"q_proj_1536x2048"`` vs ``"q_proj_1536x4096"``
    end up as distinct hypernetwork output heads. Layers with no matching
    module (e.g. k_proj on a KV-shared layer) simply produce no spec for
    that layer, which is architecturally correct: there is nothing to
    adapt there since that layer never computes its own K/V.
    """
    specs: List[ModuleSpec] = []
    for name, m in model.named_modules():
        if root_prefix and not name.startswith(root_prefix):
            continue
        match_type = next(
            (t for t in target_module_types if t in name), None
        )
        if match_type is None:
            continue
        if not isinstance(m, nn.Linear):
            continue
        m_layer = _LAYER_IDX_RE.search(name)
        layer_idx = int(m_layer.group(1)) if m_layer else -1
        shape_qualified_type = f"{match_type}_{m.in_features}x{m.out_features}"
        specs.append(ModuleSpec(
            full_name=name,
            layer_idx=layer_idx,
            type=shape_qualified_type,
            in_features=int(m.in_features),
            out_features=int(m.out_features),
        ))
    specs.sort(key=lambda s: (s.layer_idx, s.full_name))
    return specs


def replace_with_lora(model: nn.Module, specs: List[ModuleSpec],
                       rank: int, alpha: float) -> None:
    """Replace each target ``nn.Linear`` in ``model`` with a :class:`LoRA`
    wrapper. Idempotent."""
    named = dict(model.named_modules())
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    for sp in specs:
        parent_name, attr = sp.full_name.rsplit(".", 1)
        orig = getattr(named[parent_name], attr)
        if isinstance(orig, LoRA):
            continue
        assert isinstance(orig, nn.Linear), \
            f"{sp.full_name} is not nn.Linear (got {type(orig)})"
        wrapped = LoRA(orig, sp.in_features, sp.out_features,
                        rank, alpha).to(device=device, dtype=dtype)
        setattr(named[parent_name], attr, wrapped)


def inject_lora_weights(model: nn.Module, specs: List[ModuleSpec],
                         head_out: Dict[str, Dict[str, torch.Tensor]],
                         batch_index: int = 0) -> None:
    """Push ``head_out["A"][type]`` and ``head_out["B"][type]`` into the
    wrapper :class:`LoRA` modules for every spec sharing that type."""
    A_by_type = head_out["A"]
    B_by_type = head_out["B"]
    named = dict(model.named_modules())
    for sp in specs:
        named[sp.full_name].set_lora_weights(
            A_by_type[sp.type][batch_index],
            B_by_type[sp.type][batch_index],
        )


def discover_module_types_and_dims(specs: List[ModuleSpec]
                                    ) -> Dict[str, Tuple[int, int]]:
    """Return {type_name: (in_features, out_features)} -- one entry per
    target module type. Assumes all instances of the same type share dims."""
    type_dims: Dict[str, Tuple[int, int]] = {}
    for sp in specs:
        if sp.type in type_dims:
            assert type_dims[sp.type] == (sp.in_features, sp.out_features), \
                f"type {sp.type} appears with inconsistent dims"
            continue
        type_dims[sp.type] = (sp.in_features, sp.out_features)
    return type_dims


# ---------------------------------------------------------------------------
# Shared LoRA generation head (= Code2LoRAHead, renamed; architecture unchanged)
# ---------------------------------------------------------------------------

class MemoryLoRAHead(nn.Module):
    """Maps a document-context embedding to a LoRA adapter in one forward
    pass.

    Input  : ctx ``[B, input_dim]`` -- a single document embedding.
    Output : ``{"A": {type: [B, rank, in_f]}, "B": {type: [B, out_f, rank]}}``,
             one (A, B) pair per LoRA module *type*, shared across all
             target transformer layers.

    Args:
        input_dim   : Context-vector dim (2048, matches the Qwen3-Embedding
                      weighted-mean + max-pool concat from ``encoder.py``).
        type_dims   : ``{type: (in_features, out_features)}`` for each LoRA
                      module type (q_proj, v_proj, gate_proj, ...).
        hidden_dim  : Trunk hidden dimension. Default 128 -- deliberately
                      small: with only ~165 training documents (~3K QA
                      pairs), a 745M-param head (hidden_dim=512, the
                      original default) overfits within ~2 epochs (train
                      loss -> 0.4 while held-out cr_val/cr_test loss rises
                      from ~1.9 to ~2.7). hidden_dim=128 cuts head size
                      roughly 4x; combine with --head-dropout and higher
                      weight decay for further regularization.
        rank        : LoRA rank ``r``.
        init_log_scale : Initial log-scale for tanh squashing. -3.5 gives
                         output magnitudes ~0.03 at init -> tiny LoRA delta.
        dropout     : Dropout applied after each trunk GELU. 0.0 (paper's
                      original setting) had no regularization at all;
                      nonzero here specifically to counter the overfitting
                      observed on this project's much smaller corpus.
    """

    def __init__(
        self,
        input_dim: int,
        type_dims: Dict[str, Tuple[int, int]],
        hidden_dim: int = 128,
        rank: int = 16,
        init_log_scale: float = -3.5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.dropout = dropout
        self.type_dims = dict(type_dims)
        self.types = sorted(type_dims.keys())

        # Input standardization statistics (set via :meth:`fit_input_stats`).
        #
        # WHY THIS EXISTS: the 6-view repo embedding is ~64% a constant vector
        # shared by every repo (the frozen encoder's mean response to "source
        # code"), so raw embeddings have mean pairwise cosine ~0.73 -- and the
        # trunk, dominated by that DC component, mapped them to cosine ~0.98,
        # i.e. it emitted essentially the SAME adapter for every repository
        # (measured: emitted-delta cosine 0.96 across 24 unrelated repos).
        # A near-constant, non-trivially-sized delta is pure damage: it scored
        # WORSE than random noise of matched scale on held-out repo text.
        # Centering removes that DC term (pairwise cosine ~0.00), which is what
        # makes the conditioning signal actually reach the output heads.
        # Buffers (not parameters) so they persist in the checkpoint and are
        # applied identically at training and inference time.
        self.register_buffer("input_mean", torch.zeros(input_dim))
        self.register_buffer("input_std", torch.ones(input_dim))

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.heads_A = nn.ModuleDict({
            t: nn.Linear(hidden_dim, rank * type_dims[t][0])
            for t in self.types
        })
        self.heads_B = nn.ModuleDict({
            t: nn.Linear(hidden_dim, type_dims[t][1] * rank)
            for t in self.types
        })
        self.log_scale_A = nn.ParameterDict({
            t: nn.Parameter(torch.tensor(init_log_scale)) for t in self.types
        })
        self.log_scale_B = nn.ParameterDict({
            t: nn.Parameter(torch.tensor(init_log_scale)) for t in self.types
        })

    @torch.no_grad()
    def fit_input_stats(self, embeddings: torch.Tensor) -> None:
        """Set the standardization buffers from the TRAINING embeddings only.

        Must be called once before training and never refit afterwards --
        inference has to apply exactly the same transform, which is why the
        stats travel inside the checkpoint.
        """
        e = embeddings.float()
        self.input_mean.copy_(e.mean(0))
        # Guard against near-constant dimensions blowing up when divided.
        self.input_std.copy_(e.std(0).clamp_min(1e-3))

    def forward(self, ctx: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        if ctx.dim() == 3:
            ctx = torch.max(ctx, dim=1).values
        ctx = (ctx.float() - self.input_mean) / self.input_std
        h = self.trunk(ctx.float())
        h = F.normalize(h, p=2, dim=-1) * math.sqrt(self.hidden_dim)

        A_out: Dict[str, torch.Tensor] = {}
        B_out: Dict[str, torch.Tensor] = {}
        for t in self.types:
            in_f, out_f = self.type_dims[t]
            A_raw = self.heads_A[t](h).view(-1, self.rank, in_f)
            B_raw = self.heads_B[t](h).view(-1, out_f, self.rank)
            scale_A = torch.exp(self.log_scale_A[t]).clamp(1e-5, 0.3)
            scale_B = torch.exp(self.log_scale_B[t]).clamp(1e-5, 0.3)
            A_out[t] = torch.tanh(A_raw) * scale_A
            B_out[t] = torch.tanh(B_raw) * scale_B
        return {"A": A_out, "B": B_out}

    def config_dict(self) -> Dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "rank": self.rank,
            "dropout": self.dropout,
            "types": self.types,
            "type_dims": {t: list(v) for t, v in self.type_dims.items()},
        }


# ---------------------------------------------------------------------------
# Parquet loaders -- documents + recall QA pairs
# ---------------------------------------------------------------------------

def _list_to_f32_array(col, dim: int) -> np.ndarray:
    """Vectorized fixed-width-list -> ndarray. The naive per-row Python
    loop (``for i, v in enumerate(col.to_pylist()): out[i] = v``) does
    dim * n_rows individual scalar assignments in pure Python -- fine at
    ~200 rows, but at real-corpus scale (74K rows x 2048 dims = 151M
    scalar ops) it single-handedly took 5+ minutes just to load
    embeddings before training could even start. pyarrow's own flatten()
    + numpy reshape does the same conversion in C.
    """
    if len(col) == 0:
        return np.zeros((0, dim), dtype=np.float32)
    flat = col.combine_chunks().flatten() if hasattr(col, "combine_chunks") else col.flatten()
    arr = flat.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    return arr.reshape(len(col), dim)


@dataclass
class DocRow:
    doc_id: str
    doc_version: str          # constant "v1" for static (non-evolving) docs
    split: str                # "train" | "cr_val" | "cr_test" (cross-corpus)
    doc_embedding: np.ndarray  # fp32 [2048]


@dataclass
class QnaRow:
    doc_id: str
    doc_version: str
    split: str                 # cross-corpus split, inherited from DocRow
    qna_split: str              # "train" | "held_out" (in-corpus split)
    question: str
    prefix: str
    target: str


def load_doc_rows(parquet_path: Path,
                   splits: Optional[List[str]] = None,
                   embedding_col: str = "doc_embedding",
                   ) -> List[DocRow]:
    needed = ["doc_id", "doc_version", "split", embedding_col]
    ds = pads.dataset(str(parquet_path), format="parquet")
    flt = None
    if splits:
        flt = pc.is_in(pads.field("split"),
                        value_set=pa.array(splits, type=pa.string()))
    table = ds.to_table(columns=needed, filter=flt)
    n = table.num_rows
    if n == 0:
        return []
    dim = len(table.column(embedding_col)[0].as_py())
    embs = _list_to_f32_array(table.column(embedding_col), dim)
    doc_col = table.column("doc_id").to_pylist()
    ver_col = table.column("doc_version").to_pylist()
    split_col = table.column("split").to_pylist()
    rows: List[DocRow] = []
    for i in range(n):
        rows.append(DocRow(
            doc_id=doc_col[i], doc_version=ver_col[i],
            split=split_col[i] or "",
            doc_embedding=embs[i],
        ))
    return rows


def load_qna_rows(jsonl_path: Path,
                   splits: Optional[List[str]] = None,
                   qna_splits: Optional[List[str]] = None,
                   doc_ids: Optional[List[str]] = None,
                   ) -> List[QnaRow]:
    """QnA pairs are written as JSONL by ``generate_synthetic_dataset.py``
    (one row per line, cheap to append incrementally during generation) --
    unlike doc embeddings, which are batch-written parquet. Filters are
    applied in Python; at the scale of this project (low thousands of rows)
    that's simpler and fast enough."""
    import json as _json

    splits_set = set(splits) if splits else None
    qna_splits_set = set(qna_splits) if qna_splits else None
    doc_ids_set = set(doc_ids) if doc_ids else None

    rows: List[QnaRow] = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = _json.loads(line)
            if splits_set and d.get("split") not in splits_set:
                continue
            if qna_splits_set and d.get("qna_split") not in qna_splits_set:
                continue
            if doc_ids_set and d.get("doc_id") not in doc_ids_set:
                continue
            rows.append(QnaRow(
                doc_id=d.get("doc_id", ""),
                doc_version=d.get("doc_version", "v1"),
                split=d.get("split", ""),
                qna_split=d.get("qna_split", ""),
                question=d.get("question", ""),
                prefix=d.get("prefix", ""),
                target=d.get("target", ""),
            ))
    return rows


__all__ = [
    "LoRA",
    "ModuleSpec",
    "DEFAULT_ROOT_PREFIX",
    "get_module_specs",
    "replace_with_lora",
    "inject_lora_weights",
    "discover_module_types_and_dims",
    "MemoryLoRAHead",
    "DocRow",
    "QnaRow",
    "load_doc_rows",
    "load_qna_rows",
]
