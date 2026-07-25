#!/usr/bin/env python3
"""Measure whether a generated adapter actually injects repo knowledge.

Loads the base model ONCE, attaches the job's generated LoRA adapter, and
compares next-token loss on text drawn from the repo itself with the adapter
enabled vs. disabled (``PeftModel.disable_adapter()``). One model in memory,
so the two conditions are exactly comparable.

Interpretation:
  adapted loss < base loss   -> the adapter made the repo's own text more
                                predictable, i.e. it carries real information
                                about this repo.
  adapted loss ~= base loss  -> the adapter is inert on this repo.
  adapted loss > base loss   -> the adapter is actively hurting (undertrained
                                hypernetwork emitting noise).

Usage:
    python ab_test_adapter.py --job <jobId> [--max-chars 6000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))
import config  # noqa: E402


def repo_texts(ws: Path, max_chars: int) -> list[tuple[str, str]]:
    """Pull real text from the repo clone + the captured 6 views."""
    out: list[tuple[str, str]] = []
    views_path = ws / "views.json"
    if views_path.exists():
        views = json.loads(views_path.read_text())
        for name, text in views.items():
            if text and len(text.strip()) > 200:
                out.append((name, text[:max_chars]))
    repo = ws / "repo"
    if repo.exists():
        for pat in ("README.md", "README.rst", "pyproject.toml", "setup.py"):
            for p in list(repo.glob(pat))[:1]:
                try:
                    t = p.read_text(errors="ignore")
                except OSError:
                    continue
                if len(t.strip()) > 200:
                    out.append((f"file:{p.name}", t[:max_chars]))
    return out


@torch.no_grad()
def loss_on(model, tok, text: str, device: str) -> float:
    enc = tok(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    out = model(**enc, labels=enc["input_ids"])
    return float(out.loss)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--max-chars", type=int, default=6000)
    ap.add_argument("--device", default=config.DEVICE)
    args = ap.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    ws = config.workspace(args.job)
    adapter = ws / "adapter"
    if not adapter.exists():
        print(f"no adapter at {adapter}", file=sys.stderr)
        sys.exit(1)

    texts = repo_texts(ws, args.max_chars)
    if not texts:
        print("no repo text found to evaluate", file=sys.stderr)
        sys.exit(1)

    device = config.resolve_device(args.device)
    print(f"loading {config.BASE_MODEL} on {device} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForImageTextToText.from_pretrained(
        config.BASE_MODEL, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, str(adapter))
    model.to(device)
    model.eval()
    print("ready.\n", flush=True)

    print(f"{'section':<24} {'base':>9} {'adapted':>9} {'delta':>9}")
    print("-" * 54)
    n_better = 0
    sum_base = sum_adapted = 0.0
    for name, text in texts:
        with model.disable_adapter():
            lb = loss_on(model, tok, text, device)
        la = loss_on(model, tok, text, device)
        d = la - lb
        n_better += d < 0
        sum_base += lb
        sum_adapted += la
        flag = "better" if d < -1e-4 else ("worse" if d > 1e-4 else "same")
        print(f"{name[:24]:<24} {lb:9.4f} {la:9.4f} {d:+9.4f}  {flag}")

    n = len(texts)
    print("-" * 54)
    print(f"{'MEAN':<24} {sum_base/n:9.4f} {sum_adapted/n:9.4f} "
          f"{(sum_adapted-sum_base)/n:+9.4f}")
    print(f"\nsections improved by the adapter: {n_better}/{n}")
    verdict = ("adapter carries repo information"
               if sum_adapted < sum_base - 1e-3
               else "adapter is inert or harmful on this repo")
    print(f"verdict: {verdict}")


if __name__ == "__main__":
    main()
