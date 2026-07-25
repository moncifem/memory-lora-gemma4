#!/usr/bin/env python3
"""Frozen document encoder for the Memory-LoRA hypernetwork.

Forked from Code2LoRA's ``create_dataset/embed_repos.py`` chunk/pool
pipeline (mean-pool chunks -> file vector -> weighted-mean+max repo vector),
retargeted from "repo files" to "document sections":

* Code2LoRA:   file_i        -> chunks -> mean-pool -> file vector
               repo          -> weighted-mean+max over file vectors
* Memory-LoRA: doc_section_i -> chunks -> mean-pool -> section vector
               doc           -> weighted-mean+max over section vectors

Same frozen encoder as the paper (Qwen3-Embedding-0.6B), same reasoning for
the weighting (content-distinctiveness via cosine-distance-from-mean +
log-size normalization) -- multi-section documents (e.g. the Code2LoRA paper
chunked into abstract/method/results/limitations) benefit from it exactly
the way multi-file repos did. Single-section synthetic fact-sheets degenerate
gracefully to a near-uniform weighting over their own chunks.

No gradient ever flows through this encoder; embeddings are precomputed once
and cached to parquet by ``scripts/build_doc_embeddings.py``.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

DEFAULT_EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"

# Section-name heuristics (loose analogue of Code2LoRA's path up/down-weight
# lists). Neutral by default for synthetic single-section documents; the
# upweighted names matter for the multi-section Code2LoRA-paper document.
SECTION_UPWEIGHT = [
    r"abstract", r"result", r"conclusion", r"contribution",
]
SECTION_DOWNWEIGHT = [
    r"acknowledg", r"reference", r"appendix",
]

MIN_CHARS_FOR_FULL_WEIGHT = 200  # sections shorter than this are downweighted


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_token_ids(token_ids: List[int], chunk_tokens: int, overlap: int) -> List[List[int]]:
    """Produce overlapping token windows (identical to Code2LoRA's version)."""
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be > 0")
    if overlap >= chunk_tokens:
        raise ValueError("chunk_overlap must be < chunk_tokens")
    chunks: List[List[int]] = []
    step = chunk_tokens - overlap
    n = len(token_ids)
    if n == 0:
        return chunks
    for start in range(0, n, step):
        end = min(start + chunk_tokens, n)
        window = token_ids[start:end]
        if len(window) < 16:
            continue
        chunks.append(window)
        if end >= n:
            break
    return chunks


# ---------------------------------------------------------------------------
# Embedding model wrapper
# ---------------------------------------------------------------------------

@torch.inference_mode()
def embed_texts(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    texts: List[str],
    device: str,
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    """Return embeddings [N, D] using mean pooling over last_hidden_state."""
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        last = out.last_hidden_state  # [B, T, H]
        mask = enc["attention_mask"].unsqueeze(-1)  # [B, T, 1]
        mean = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        all_vecs.append(mean.detach().cpu())
    if not all_vecs:
        return torch.empty((0, model.config.hidden_size))
    return torch.cat(all_vecs, dim=0)


# ---------------------------------------------------------------------------
# Pooling: chunks -> section -> document
# ---------------------------------------------------------------------------

def pool_section_embeddings(chunk_embs: torch.Tensor) -> Optional[torch.Tensor]:
    """chunk_embs [K, D] -> section_emb [D]"""
    if chunk_embs.numel() == 0:
        return None
    return chunk_embs.mean(dim=0)


def _section_name_bonus(section_name: str) -> float:
    s = section_name.lower()
    bonus = 0.0
    for pat in SECTION_DOWNWEIGHT:
        if re.search(pat, s):
            bonus -= 0.25
            break
    for pat in SECTION_UPWEIGHT:
        if re.search(pat, s):
            bonus += 0.15
            break
    return bonus


def compute_section_weights(
    section_embs: torch.Tensor,          # [S, D]
    section_char_counts: torch.Tensor,   # [S]
    section_names: List[str],
    a_distinct: float,
    b_size: float,
    tau: float,
) -> torch.Tensor:
    """
    Whole-doc, all-sections weighting:
      distinct_i = 1 - cos(s_i, mean_s)
      size_i     = normalized log(1+chars)
      score_i    = a_distinct * distinct_i + b_size * size_i + name_bonus_i + tiny_section_penalty
      w          = softmax(score / tau)
    Returns: w [S] sum=1
    """
    f_norm = F.normalize(section_embs, p=2, dim=-1)
    mean_f = F.normalize(f_norm.mean(dim=0, keepdim=True), p=2, dim=-1)
    cos = (f_norm * mean_f).sum(dim=-1).clamp(-1, 1)
    distinct = 1.0 - cos

    chars = section_char_counts.float().clamp(min=1)
    log_chars = torch.log1p(chars)
    if log_chars.numel() > 1:
        lo, hi = log_chars.min(), log_chars.max()
        size01 = (log_chars - lo) / (hi - lo + 1e-8)
    else:
        size01 = torch.ones_like(log_chars)

    name_bonus = torch.tensor([_section_name_bonus(n) for n in section_names],
                               dtype=torch.float32)
    tiny_scale = (chars / float(MIN_CHARS_FOR_FULL_WEIGHT)).clamp(max=1.0)
    tiny_bonus = torch.log(tiny_scale + 1e-6)

    score = (a_distinct * distinct.cpu() + b_size * size01.cpu()
             + name_bonus + 0.15 * tiny_bonus.cpu())
    return torch.softmax(score / max(tau, 1e-6), dim=0)


def pool_doc_embedding_weighted(
    section_embs: torch.Tensor,          # [S, D]
    section_char_counts: torch.Tensor,   # [S]
    section_names: List[str],
    a_distinct: float = 1.0,
    b_size: float = 0.5,
    tau: float = 0.5,
    alpha_mean: float = 1.0,
    beta_max: float = 1.0,
) -> Optional[torch.Tensor]:
    """Aggregate section embeddings into one document vector:
    concat(alpha_mean * weighted_mean, beta_max * max) -> [2D]. No final
    L2 normalization (matches Code2LoRA's repo-vector convention)."""
    if section_embs.numel() == 0:
        return None
    w = compute_section_weights(
        section_embs, section_char_counts, section_names,
        a_distinct, b_size, tau,
    ).to(section_embs.dtype)
    wmean = (section_embs * w.unsqueeze(-1)).sum(dim=0)
    vmax = section_embs.max(dim=0).values
    return torch.cat([alpha_mean * wmean, beta_max * vmax], dim=0)


# ---------------------------------------------------------------------------
# Main pipeline per document
# ---------------------------------------------------------------------------

def embed_document(
    sections: List[Tuple[str, str]],   # [(section_name, section_text), ...]
    model: AutoModel,
    tokenizer: AutoTokenizer,
    device: str,
    chunk_tokens: int = 4096,
    chunk_overlap: int = 512,
    batch_size: int = 4,
    a_distinct: float = 1.0,
    b_size: float = 0.5,
    tau: float = 0.5,
    alpha_mean: float = 1.0,
    beta_max: float = 1.0,
) -> Optional[torch.Tensor]:
    """One document = list of (name, text) sections (single-element for a
    plain synthetic fact-sheet; multi-element for the chunked paper).
    Returns a [2D] embedding, or None if the document had no usable text."""
    section_vectors: List[torch.Tensor] = []
    section_names: List[str] = []
    section_char_counts: List[int] = []

    for name, text in sections:
        text = (text or "").strip()
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        windows = chunk_token_ids(ids, chunk_tokens=chunk_tokens, overlap=chunk_overlap)
        if not windows:
            continue
        chunks = [tokenizer.decode(w, skip_special_tokens=True) for w in windows]
        chunk_embs = embed_texts(
            model=model, tokenizer=tokenizer, texts=chunks,
            device=device, batch_size=batch_size, max_length=chunk_tokens,
        )
        svec = pool_section_embeddings(chunk_embs)
        if svec is None:
            continue
        section_vectors.append(svec)
        section_names.append(name)
        section_char_counts.append(len(text))

    if not section_vectors:
        return None

    section_embs = torch.stack(section_vectors, dim=0)
    char_t = torch.tensor(section_char_counts, dtype=torch.int64)
    return pool_doc_embedding_weighted(
        section_embs, char_t, section_names,
        a_distinct=a_distinct, b_size=b_size, tau=tau,
        alpha_mean=alpha_mean, beta_max=beta_max,
    )


def load_encoder(model_name: str = DEFAULT_EMBED_MODEL, device: str = "mps"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
    model.to(device)
    model.eval()
    return model, tokenizer


__all__ = [
    "DEFAULT_EMBED_MODEL",
    "chunk_token_ids",
    "embed_texts",
    "pool_section_embeddings",
    "compute_section_weights",
    "pool_doc_embedding_weighted",
    "embed_document",
    "load_encoder",
]
