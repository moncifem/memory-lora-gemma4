#!/usr/bin/env python3
"""Build REAL 6-view repo inputs locally: clone repos, extract the 6
tech-lead views from each full tree, embed each view with the frozen Qwen
encoder, concat -> one multi-view repo vector. Also emit a compact per-view
TEXT summary per repo (so generate_repo_scoped_qa.py can produce
scope-aligned targets from the SAME views the embedding sees).

Views (each -> Qwen mean+max = 2048-d; 6 views -> 12288-d input vector;
head input_dim is set to match, "~8k" was always approximate):
  v_graph      : AST call/import graph (codegraph.py) + dependency edges
  v_arch       : READMEs, top-level docstrings, folder tree
  v_history    : git log --oneline + sampled commit diffs  ("the why")
  v_contracts  : test files + type-annotated signatures
  v_conventions: sampled source files (naming/idioms)
  v_ops        : Dockerfile / CI yaml / pyproject / setup / .env.example

Streaming design: load encoder ONCE, then per repo {clone shallow -> extract
-> embed -> save -> delete clone} so disk and memory stay bounded. Clone
failures / oversized / non-Python repos are skipped gracefully.

Usage:
  python scripts/build_repo_multiview.py --repos-file data/repo_list.txt --limit 3
  python scripts/build_repo_multiview.py --repos-file data/repo_list.txt --max-repos 250
"""
from __future__ import annotations
import argparse, json, os, re, shutil, subprocess, sys, tempfile, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np, torch

HERE = Path(__file__).resolve().parent; REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))
from memory_lora.data_paths import EMBEDDINGS_DIR, DOCS_DIR, ensure_dirs  # noqa: E402
from memory_lora.encoder import load_encoder, embed_document  # noqa: E402
from memory_lora.codegraph import extract_repo_graph_sections, extract_repo_dependency_summary  # noqa: E402

SKIP = {".git", "__pycache__", ".venv", "venv", "node_modules", "build", "dist",
        ".tox", ".mypy_cache", "vendor", "third_party", "target", ".next"}
VIEWS = ["v_graph", "v_arch", "v_history", "v_contracts", "v_conventions", "v_ops"]

# language-agnostic file classification
CODE_EXTS = {".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs", ".c",
             ".h", ".cc", ".cpp", ".hpp", ".php", ".rb", ".kt", ".swift", ".scala", ".cs"}
TEST_HINTS = ("test", "spec", "_test.", ".test.", "tests/")
# import-statement patterns per common language (for the non-Python graph fallback)
IMPORT_RE = re.compile(
    r"^\s*(?:import\s+[^\n;]+|from\s+[^\n]+import[^\n]+|#include\s*[<\"][^>\"]+[>\"]|"
    r"use\s+[^\n;]+|require\s*\(?[^\n)]+\)?|package\s+[^\n;]+)", re.MULTILINE)
DEF_RE = re.compile(
    r"^\s*(?:def\s+\w+|class\s+\w+|func\s+\w+|function\s+\w+|fn\s+\w+|"
    r"(?:public|private|protected|static|\s)+\w[\w<>\[\]]*\s+\w+\s*\(|"
    r"type\s+\w+|struct\s+\w+|interface\s+\w+|export\s+(?:default\s+)?(?:function|class|const)\s+\w+)",
    re.MULTILINE)


def run(cmd: List[str], cwd=None, timeout=120) -> Tuple[int, str]:
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                           errors="ignore", timeout=timeout)
        return r.returncode, r.stdout
    except Exception:  # noqa: BLE001
        return 1, ""


def clone(name: str, dest: Path, depth=80) -> bool:
    url = f"https://github.com/{name}.git"
    code, _ = run(["git", "clone", "--depth", str(depth), "--quiet", url, str(dest)], timeout=180)
    return code == 0 and dest.exists()


def _read(p: Path, n=8000) -> str:
    try: return p.read_text(errors="ignore")[:n]
    except OSError: return ""


def _folder_tree(root: Path, max_lines=120) -> str:
    lines = []
    for p in sorted(root.rglob("*")):
        if any(s in p.parts for s in SKIP): continue
        rel = p.relative_to(root)
        if len(rel.parts) > 3: continue
        lines.append(str(rel) + ("/" if p.is_dir() else ""))
        if len(lines) >= max_lines: break
    return "\n".join(lines)


def extract_views(repo: Path) -> Dict[str, List[Tuple[str, str]]]:
    """Return {view_name: [(section_name, text), ...]} for the 6 views."""
    v: Dict[str, List[Tuple[str, str]]] = {k: [] for k in VIEWS}

    # v_graph: Python AST call/import graph (when present) + language-agnostic
    # regex import/definition summary so non-Python repos still get structure.
    v["v_graph"] = extract_repo_graph_sections(repo, max_files=50, skip_dirs=SKIP)
    dep = extract_repo_dependency_summary(repo, skip_dirs=SKIP)
    if dep: v["v_graph"].append(("dependency_edges", dep))
    if len(v["v_graph"]) < 3:  # non-Python (or thin): regex-extract imports/defs
        code = [p for p in repo.rglob("*")
                if p.suffix.lower() in CODE_EXTS and not any(s in p.parts for s in SKIP)][:60]
        imports, defs = [], []
        for p in code:
            t = _read(p, 6000)
            imports += IMPORT_RE.findall(t)[:8]
            defs += [f"{p.name}: {m.strip()[:80]}" for m in DEF_RE.findall(t)[:6]]
        if imports: v["v_graph"].append(("imports", "\n".join(imports[:120])))
        if defs: v["v_graph"].append(("definitions", "\n".join(defs[:120])))

    # v_arch: READMEs, folder tree, package docstrings
    for pat in ["README*", "ARCHITECTURE*", "docs/*.md", "*/__init__.py"]:
        for p in list(repo.glob(pat))[:4]:
            if p.is_file(): v["v_arch"].append((f"arch:{p.name}", _read(p, 6000)))
    v["v_arch"].append(("folder_tree", _folder_tree(repo)))

    # v_history: git log + a few diffs
    _, log = run(["git", "-C", str(repo), "log", "--oneline", "-80"])
    if log: v["v_history"].append(("git_log", log[:6000]))
    _, shas = run(["git", "-C", str(repo), "log", "--format=%H", "-6"])
    for sha in [s for s in shas.split() if s][:5]:
        _, diff = run(["git", "-C", str(repo), "show", "--stat", "-p", sha])
        if diff: v["v_history"].append((f"diff:{sha[:8]}", diff[:3000]))

    # v_contracts: test files (any language) + typed signatures
    tests = [p for p in repo.rglob("*")
             if p.suffix.lower() in CODE_EXTS and not any(s in p.parts for s in SKIP)
             and any(h in str(p).lower() for h in TEST_HINTS)][:10]
    for p in tests:
        v["v_contracts"].append((f"test:{p.name}", _read(p, 4000)))

    # v_conventions: sampled source files (any language, non-test)
    srcs = [p for p in repo.rglob("*")
            if p.suffix.lower() in CODE_EXTS and not any(s in p.parts for s in SKIP)
            and not any(h in str(p).lower() for h in TEST_HINTS)][:12]
    for p in srcs:
        v["v_conventions"].append((f"src:{p.name}", _read(p, 3500)))

    # v_ops: build/CI/config
    for pat in ["Dockerfile*", ".github/workflows/*.y*ml", "pyproject.toml",
                "setup.py", "setup.cfg", "requirements*.txt", "Makefile", ".env.example",
                "docker-compose*.y*ml", "tox.ini"]:
        for p in list(repo.glob(pat))[:3]:
            if p.is_file(): v["v_ops"].append((f"ops:{p.name}", _read(p, 4000)))
    return v


def summarize_views_text(views: Dict[str, List[Tuple[str, str]]], per_view=1400) -> Dict[str, str]:
    """Compact text per view for QA generation (scope-aligned to embedding)."""
    out = {}
    for k, secs in views.items():
        joined = "\n".join(f"# {n}\n{t}" for n, t in secs)
        out[k] = joined[:per_view]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos-file", required=True)
    ap.add_argument("--max-repos", type=int, default=250)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out-emb", default=str(EMBEDDINGS_DIR / "multiview_embeddings.parquet"))
    ap.add_argument("--out-src", default=str(DOCS_DIR / "multiview_sources.jsonl"))
    args = ap.parse_args()
    ensure_dirs()

    target = args.limit if args.limit else args.max_repos
    # accept "repo" or "repo<TAB>lang" lines; keep the lang tag if present
    name_lang = []
    for l in open(args.repos_file):
        l = l.strip()
        if not l: continue
        parts = l.split("\t")
        name_lang.append((parts[0], parts[1] if len(parts) > 1 else "unknown"))
    name_lang = name_lang[: target * 3]  # oversample for clone failures
    names = [n for n, _ in name_lang]
    lang_of = dict(name_lang)

    device = args.device if (args.device != "mps" or torch.backends.mps.is_available()) else "cpu"
    print(f"Loading Qwen encoder on {device} ...", flush=True)
    enc_model, enc_tok = load_encoder(device=device)

    # resume-safe: skip repos already done
    done = set()
    if Path(args.out_src).exists():
        done = {json.loads(l)["repo"] for l in open(args.out_src)}
    src_f = open(args.out_src, "a")

    import pyarrow as pa, pyarrow.parquet as pq
    emb_rows = []
    # load existing embeddings if resuming
    if Path(args.out_emb).exists():
        t = pq.read_table(args.out_emb).to_pylist()
        emb_rows = t

    n_done = 0; t0 = time.time()
    for name in names:
        if n_done >= target: break
        if name in done: continue
        tmp = Path(tempfile.mkdtemp(prefix="rmv_"))
        try:
            if not clone(name, tmp / "r"):
                continue
            repo = tmp / "r"
            # skip repos with too little code (any language)
            codefiles = [p for p in repo.rglob("*")
                         if p.suffix.lower() in CODE_EXTS and not any(s in p.parts for s in SKIP)]
            if len(codefiles) < 3:
                continue
            views = extract_views(repo)
            # embed each view -> mean+max 2048; concat 6 -> 12288
            vecs = []
            for vk in VIEWS:
                secs = views[vk] or [("empty", "none")]
                vv = embed_document(secs, enc_model, enc_tok, device,
                                    chunk_tokens=2048, chunk_overlap=128, batch_size=2)
                vecs.append(vv.numpy().astype("float32") if vv is not None else np.zeros(2048, "float32"))
            full = np.concatenate(vecs)  # 12288
            doc_id = name  # repo-scoped
            emb_rows.append({"doc_id": doc_id, "doc_version": "head", "split": "train",
                             "category": "real_repo_multiview", "doc_embedding": full.tolist()})
            src_f.write(json.dumps({"repo": name, "doc_id": doc_id,
                                    "lang": lang_of.get(name, "unknown"),
                                    "view_text": summarize_views_text(views)}) + "\n")
            src_f.flush()
            n_done += 1
            if n_done % 10 == 0:
                rate = n_done / max(1e-9, (time.time() - t0) / 60)
                # periodic parquet flush
                _flush(emb_rows, args.out_emb, pa, pq)
                print(f"  {n_done} repos done ({rate:.1f}/min) dim={full.shape[0]} latest={name}", flush=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    _flush(emb_rows, args.out_emb, pa, pq)
    src_f.close()
    print(f"\nDone. {n_done} repos -> {args.out_emb} (dim={len(emb_rows[0]['doc_embedding']) if emb_rows else 0})", flush=True)


def _flush(rows, path, pa, pq):
    if not rows: return
    t = pa.table({k: [r[k] for r in rows] for k in
                  ["doc_id", "doc_version", "split", "category", "doc_embedding"]})
    pq.write_table(t, path)


if __name__ == "__main__":
    main()
