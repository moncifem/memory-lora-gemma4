#!/usr/bin/env python3
"""Benchmark a generated adapter against the frozen base model on ONE repo.

Builds the question set from the repo's ACTUAL contents (deps, test framework,
packaging, layout, license, entry points) rather than hand-written guesses, so
the gold answers are ground truth rather than opinion. Then scores base vs
adapted on the same model instance, adapter toggled with `disable_adapter()`.

Three task families, because an adapter can help one and hurt another -- which
is exactly what happened here (QA improved hugely while raw-text modelling
regressed), and a single number would have hidden it:

  FACT   - "Q: <question>\\nA:" -> short factual answer. The trained format.
  CODE   - completion of real lines taken from the repo's own source.
  TEXT   - plain continuation of repo prose (README/docstrings).

Metrics per family:
  loss      teacher-forced cross-entropy on the gold answer (lower better)
  win rate  fraction of items where adapted loss < base loss
  keyword   fraction of greedy generations containing the gold keyword

Usage:
    python benchmark_repo.py --job <jobId> --checkpoint ../../runs/h200_run/head.best.pt
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "engine"))
import config  # noqa: E402
from generate_and_merge import (TARGET_MODULES, load_head,  # noqa: E402
                                _lora_modules_by_spec_name)
from memory_lora.core import DEFAULT_ROOT_PREFIX, get_module_specs  # noqa: E402

SKIP = {".git", "__pycache__", ".venv", "node_modules", "build", "dist", ".tox"}


# --------------------------------------------------------------------------
# Build the benchmark from repo ground truth
# --------------------------------------------------------------------------

def _read(p: Path, n: int = 20000) -> str:
    try:
        return p.read_text(errors="ignore")[:n]
    except OSError:
        return ""


def build_fact_items(repo: Path) -> list[dict]:
    """Derive Q/A pairs whose answers are verifiable from the repo itself."""
    items: list[dict] = []
    pyproject = _read(repo / "pyproject.toml")
    setup_py = _read(repo / "setup.py")
    setup_cfg = _read(repo / "setup.cfg")
    build = pyproject + setup_py + setup_cfg

    # packaging backend
    if "setuptools" in build:
        items.append(dict(q="What packaging tool does this project use?",
                          a=" setuptools", kw="setuptools"))
    elif "poetry" in build.lower():
        items.append(dict(q="What packaging tool does this project use?",
                          a=" poetry", kw="poetry"))
    if "hatchling" in build:
        items.append(dict(q="What build backend does this project declare?",
                          a=" hatchling", kw="hatchling"))

    # test framework
    test_files = [p for p in repo.rglob("*.py")
                  if not any(s in p.parts for s in SKIP)
                  and ("test" in p.name.lower() or "tests" in p.parts)]
    joined = " ".join(_read(p, 4000) for p in test_files[:12])
    if "pytest" in joined or "pytest" in build:
        items.append(dict(q="What testing framework does this repository use?",
                          a=" pytest", kw="pytest"))
    elif "unittest" in joined:
        items.append(dict(q="What testing framework does this repository use?",
                          a=" unittest", kw="unittest"))

    # license
    lic = _read(repo / "LICENSE") + _read(repo / "LICENSE.txt")
    for name, key in (("Apache", "Apache"), ("MIT", "MIT"),
                      ("BSD", "BSD"), ("GNU", "GPL")):
        if name.lower() in lic.lower()[:400]:
            items.append(dict(q="What license does this project use?",
                              a=f" {key}", kw=key))
            break

    # top-level package
    EXCL = {"tests", "test", "docs", "doc", "examples", "example", "scripts",
            "benchmarks", "ext"}
    pkgs = [d.name for d in repo.iterdir()
            if d.is_dir() and (d / "__init__.py").exists()
            and d.name not in SKIP and d.name.lower() not in EXCL]
    if pkgs:
        # Prefer the package named after the repo (src/<name> layouts included).
        best = next((k for k in pkgs if k.lower() == repo.name.lower()), pkgs[0])
        items.append(dict(q="What is the name of the main Python package in this repository?",
                          a=f" {best}", kw=best))

    # dependencies
    # Parse only INSIDE a dependency list, otherwise setup.py keywords such as
    # `install_requires=` / `python_requires=` get matched as package names.
    dep_block = ""
    for pat in (r"install_requires\s*=\s*\[(.*?)\]",
                r"dependencies\s*=\s*\[(.*?)\]",
                r"\[project\.dependencies\](.*?)(?:\n\[|\Z)"):
        m = re.search(pat, build, re.S)
        if m:
            dep_block = m.group(1)
            break
    NOT_PKG = {"python", "name", "version", "requires", "install", "extras",
               "setup", "packages", "classifiers"}
    deps = re.findall(r"['\"]([A-Za-z][A-Za-z0-9_.-]{2,})\s*[><=~!\[]", dep_block)
    deps = [d for d in deps if d.lower() not in NOT_PKG]
    if deps:
        items.append(dict(q="Name a runtime dependency of this project.",
                          a=f" {deps[0]}", kw=deps[0]))

    # CI
    ci = list((repo / ".github" / "workflows").glob("*.y*ml")) if (repo / ".github" / "workflows").exists() else []
    if ci:
        items.append(dict(q="What CI system does this repository use?",
                          a=" GitHub Actions", kw="GitHub Actions"))

    # docs
    if (repo / "docs").is_dir():
        conf = _read(repo / "docs" / "conf.py")
        if "sphinx" in conf.lower() or (repo / "docs" / "conf.py").exists():
            items.append(dict(q="What documentation tool does this project use?",
                              a=" Sphinx", kw="Sphinx"))
    return items


def build_code_items(repo: Path, n: int = 12) -> list[dict]:
    """Split real source lines: prefix -> the rest of the line."""
    items: list[dict] = []
    srcs = [p for p in repo.rglob("*.py")
            if not any(s in p.parts for s in SKIP)
            and "test" not in p.name.lower()]
    for p in srcs[:40]:
        text = _read(p, 12000)
        lines = [l for l in text.splitlines()
                 if 30 < len(l) < 110 and not l.strip().startswith("#")
                 and ("(" in l or "=" in l or "import" in l)]
        for l in lines[:2]:
            cut = max(len(l) // 2, l.find("(") + 1 if "(" in l else len(l) // 2)
            prefix, target = l[:cut], l[cut:]
            if len(target.strip()) < 4:
                continue
            items.append(dict(prefix=f"# file: {p.name}\n{prefix}", target=target))
            if len(items) >= n:
                return items
    return items


def build_text_items(repo: Path, n: int = 6) -> list[dict]:
    items: list[dict] = []
    for name in ("README.md", "README.rst", "HISTORY.md", "CHANGELOG.md"):
        t = _read(repo / name, 6000)
        if len(t) < 600:
            continue
        chunks = [c for c in t.split("\n\n") if len(c) > 200][:3]
        for c in chunks:
            half = len(c) // 2
            items.append(dict(prefix=c[:half], target=c[half:half + 300]))
            if len(items) >= n:
                return items
    return items


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

@torch.no_grad()
def item_loss(model, tok, prefix: str, target: str, device: str) -> float:
    pid = tok(prefix, add_special_tokens=False)["input_ids"]
    tid = tok(target, add_special_tokens=False)["input_ids"]
    if not tid or not pid:
        return float("nan")
    ids = torch.tensor([pid + tid], device=device)
    labels = torch.tensor([[-100] * len(pid) + tid], device=device)
    return float(model(input_ids=ids, labels=labels).loss)


@torch.no_grad()
def generate(model, tok, prompt: str, device: str, max_new: int = 24) -> str:
    enc = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:],
                      skip_special_tokens=True).split("\n")[0]


def run_family(model, tok, items, device, name, generate_kw=False):
    rows = []
    for it in items:
        prefix = it.get("prefix") or f"Q: {it['q']}\nA:"
        target = it.get("target") or it["a"]
        with model.disable_adapter():
            lb = item_loss(model, tok, prefix, target, device)
            gb = generate(model, tok, prefix, device) if generate_kw else ""
        la = item_loss(model, tok, prefix, target, device)
        ga = generate(model, tok, prefix, device) if generate_kw else ""
        rows.append(dict(prefix=prefix, target=target, kw=it.get("kw", ""),
                         base=lb, adapted=la, gen_base=gb, gen_adapted=ga))
    valid = [r for r in rows if not (np.isnan(r["base"]) or np.isnan(r["adapted"]))]
    if not valid:
        return None
    mb = float(np.mean([r["base"] for r in valid]))
    ma = float(np.mean([r["adapted"] for r in valid]))
    wins = sum(1 for r in valid if r["adapted"] < r["base"])
    out = dict(family=name, n=len(valid), base=mb, adapted=ma,
               delta=ma - mb, win_rate=wins / len(valid))
    if generate_kw:
        kb = sum(1 for r in valid if r["kw"] and r["kw"].lower() in r["gen_base"].lower())
        ka = sum(1 for r in valid if r["kw"] and r["kw"].lower() in r["gen_adapted"].lower())
        nk = sum(1 for r in valid if r["kw"])
        out["kw_base"] = kb / max(nk, 1)
        out["kw_adapted"] = ka / max(nk, 1)
        out["nk"] = nk
    return out, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--checkpoint", default=str(config.DEFAULT_CHECKPOINT))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--show-generations", action="store_true")
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    ws = config.workspace(args.job)
    repo = ws / "repo"
    if not repo.exists():
        print(f"repo clone missing at {repo}", file=sys.stderr)
        sys.exit(1)
    repo_url = json.loads((ws / "status.json").read_text()).get("repo_url", "?")
    emb = np.load(ws / "embedding.npy").astype("float32")
    device = config.resolve_device(args.device)

    facts = build_fact_items(repo)
    codes = build_code_items(repo)
    texts = build_text_items(repo)
    print(f"repo:       {repo_url}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"benchmark:  {len(facts)} FACT, {len(codes)} CODE, {len(texts)} TEXT items\n",
          flush=True)

    head, cfg, alpha = load_head(Path(args.checkpoint))
    with torch.no_grad():
        head_out = head(torch.from_numpy(emb).unsqueeze(0))

    tok = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForImageTextToText.from_pretrained(
        config.BASE_MODEL, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    specs = get_module_specs(base, TARGET_MODULES, root_prefix=DEFAULT_ROOT_PREFIX)
    type_of = {s.full_name: s.type for s in specs}
    model = get_peft_model(base, LoraConfig(
        r=cfg["rank"], lora_alpha=alpha,
        target_modules=[s.full_name for s in specs], lora_dropout=0.0, bias="none"))
    mods = _lora_modules_by_spec_name(model)
    with torch.no_grad():
        for sp in specs:
            m = mods.get(sp.full_name)
            if m is None:
                continue
            t = type_of[sp.full_name]
            m.lora_A["default"].weight.copy_(head_out["A"][t][0].float())
            m.lora_B["default"].weight.copy_(head_out["B"][t][0].float())
    model.to(device)
    model.eval()

    results = []
    all_rows = {}
    for items, name, gk in ((facts, "FACT", True), (codes, "CODE", False),
                            (texts, "TEXT", False)):
        if not items:
            continue
        r = run_family(model, tok, items, device, name, generate_kw=gk)
        if r:
            res, rows = r
            results.append(res)
            all_rows[name] = rows

    print(f"{'family':<7} {'n':>3} {'base':>8} {'adapted':>8} {'delta':>9} {'win%':>6}")
    print("-" * 46)
    for r in results:
        print(f"{r['family']:<7} {r['n']:>3} {r['base']:>8.4f} {r['adapted']:>8.4f} "
              f"{r['delta']:>+9.4f} {100*r['win_rate']:>5.0f}%")
    for r in results:
        if "kw_base" in r:
            print(f"\nFACT keyword accuracy over {r['nk']} verifiable answers:")
            print(f"  base    {100*r['kw_base']:.0f}%")
            print(f"  adapted {100*r['kw_adapted']:.0f}%")

    if args.show_generations and "FACT" in all_rows:
        print("\n--- FACT generations ---")
        for row in all_rows["FACT"]:
            q = row["prefix"].replace("Q: ", "").replace("\nA:", "")
            print(f"Q: {q}\n  gold:    {row['target'].strip()}")
            print(f"  base:    {row['gen_base'].strip()[:110]}")
            print(f"  adapted: {row['gen_adapted'].strip()[:110]}\n")


if __name__ == "__main__":
    main()
