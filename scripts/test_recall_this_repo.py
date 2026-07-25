#!/usr/bin/env python3
"""Step B of the side recall-test. Loads base Gemma-4-E2B on CPU, generates
a LoRA for THIS repo from its precomputed embedding (Step A) via the
currently-training hypernetwork's latest checkpoint, and compares base vs
adapted completions on prompts drawn from this repo's own code.

CPU-only (never touches the live MPS training). Peak memory ~= just the
base model (~10GB) since the encoder already ran and exited in Step A.

Note on interpretation: the hypernetwork was trained on pytest
ASSERTION-COMPLETION over real repos, and THIS repo is unseen (true
cross-repo test). Prompts are shaped to probe whether the generated
adapter biases the frozen model toward this repo's specific identifiers/
values vs the base model's generic guess.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))
from memory_lora.core import (MemoryLoRAHead, get_module_specs, replace_with_lora,
                               inject_lora_weights, DEFAULT_ROOT_PREFIX)
from transformers import AutoModelForImageTextToText, AutoTokenizer

TARGET_MODULES = ["q_proj","k_proj","v_proj","o_proj","up_proj","gate_proj","down_proj"]

# Prompts drawn from THIS repo's actual code / conventions. Each is cut right
# before a repo-specific value the adapter should help recall.
PROMPTS = [
    ("from memory_lora.core import MemoryLoRAHead\n"
     "def test_default_rank():\n"
     "    head = MemoryLoRAHead(input_dim=2048, type_dims={})\n"
     "    assert head.rank ==", "16"),
    ("# memory_lora/core.py sets the LoRA injection root for Gemma-4-E2B\n"
     "DEFAULT_ROOT_PREFIX =", '"model.language_model."'),
    ("# memory_lora/encoder.py default frozen embedding model\n"
     "DEFAULT_EMBED_MODEL =", '"Qwen/Qwen3-Embedding-0.6B"'),
    ("# scripts/train_memory_lora.py base model being adapted\n"
     "DEFAULT_MODEL =", '"google/gemma-4-E2B"'),
    ("from memory_lora.core import MemoryLoRAHead\n"
     "def test_hidden_dim():\n"
     "    h = MemoryLoRAHead(input_dim=2048, type_dims={})\n"
     "    assert h.hidden_dim ==", "128"),
    ("# the seven attention/MLP projection types Memory-LoRA targets\n"
     "DEFAULT_TARGET_MODULES = [\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\",", '"up_proj"'),
]


@torch.no_grad()
def gen(model, tok, prefix, max_new=14):
    enc = tok(prefix, return_tensors="pt")
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                          pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).split("\n")[0]


def main():
    print("Loading base google/gemma-4-E2B on CPU (bounded ~10GB)...", flush=True)
    tok = AutoTokenizer.from_pretrained("google/gemma-4-E2B")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForImageTextToText.from_pretrained(
        "google/gemma-4-E2B", torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    print("  loaded.", flush=True)

    ckpt = torch.load(REPO_ROOT / "runs" / "test_ckpt.pt", map_location="cpu")
    specs = get_module_specs(model, TARGET_MODULES, root_prefix=DEFAULT_ROOT_PREFIX)
    replace_with_lora(model, specs, rank=ckpt["config"]["rank"], alpha=ckpt["args"].get("alpha", 32.0))
    head = MemoryLoRAHead(input_dim=ckpt["config"]["input_dim"],
                          type_dims={k: tuple(v) for k, v in ckpt["config"]["type_dims"].items()},
                          hidden_dim=ckpt["config"]["hidden_dim"], rank=ckpt["config"]["rank"])
    head.load_state_dict(ckpt["state_dict"]); head.eval()

    repo_emb = torch.from_numpy(np.load(REPO_ROOT / "runs" / "this_repo_emb.npy")).unsqueeze(0)
    head_out = head(repo_emb)
    print(f"  generated adapter for THIS repo (unseen). "
          f"training step in checkpoint: {ckpt['args'].get('epochs','?')} epochs cfg\n", flush=True)

    named = dict(model.named_modules())
    for prefix, gold in PROMPTS:
        # base: no adapter
        for sp in specs:
            named[sp.full_name].A = None; named[sp.full_name].B = None
        base = gen(model, tok, prefix)
        # adapted: inject this repo's generated LoRA
        inject_lora_weights(model, specs, head_out, batch_index=0)
        adapted = gen(model, tok, prefix)
        tag = prefix.strip().split("\n")[-1][:55]
        print(f"PROMPT: ...{tag}", flush=True)
        print(f"  gold:    {gold}", flush=True)
        print(f"  base:    {base!r}", flush=True)
        print(f"  adapted: {adapted!r}", flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
