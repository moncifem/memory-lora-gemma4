#!/usr/bin/env python3
"""AST-based codebase graph representation.

New data representation beyond what the Code2LoRA paper used (raw file
chunks, mean+max pooled). This module extracts a STRUCTURAL summary of a
Python codebase -- imports (dependency edges between files), class
hierarchies, function/method signatures, and a best-effort call graph --
and serializes it into compact text sections. Those sections are fed
through the SAME frozen embedding pipeline as raw code
(``memory_lora.encoder.embed_document`` already accepts a list of
``(section_name, section_text)`` tuples), so a repository can be embedded
from BOTH its raw source text and its structural graph, without any
change to the encoder or hypernetwork.

Why this matters for "recall codebases": raw-text chunking captures
surface content (docstrings, comments, literal code) but dilutes the
signal that actually matters for API-level questions -- "what does
function X take", "what inherits from Y", "who imports Z". A compact
graph serialization puts that signal in a small number of dense tokens
instead of burying it across thousands of raw-code tokens.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class FunctionSig:
    name: str
    args: List[str]
    returns: Optional[str]
    is_method: bool = False
    decorators: List[str] = field(default_factory=list)
    calls: List[str] = field(default_factory=list)  # best-effort, unresolved names


@dataclass
class ClassSig:
    name: str
    bases: List[str]
    methods: List[FunctionSig] = field(default_factory=list)


@dataclass
class FileGraph:
    path: str
    imports: List[str]
    import_froms: List[Tuple[str, List[str]]]  # (module, [names])
    classes: List[ClassSig]
    functions: List[FunctionSig]  # module-level only


def _annotation_to_str(node: Optional[ast.AST]) -> Optional[str]:
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:  # noqa: BLE001
        return None


def _arg_to_str(a: ast.arg) -> str:
    ann = _annotation_to_str(a.annotation)
    return f"{a.arg}: {ann}" if ann else a.arg


def _extract_calls(node: ast.AST) -> List[str]:
    calls = []
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name):
                calls.append(f.id)
            elif isinstance(f, ast.Attribute):
                calls.append(f.attr)
    # dedupe, keep order
    seen = set()
    out = []
    for c in calls:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out[:20]  # cap -- this is a signal, not a full trace


def _function_sig(node, is_method: bool = False) -> FunctionSig:
    args = [_arg_to_str(a) for a in node.args.args]
    returns = _annotation_to_str(node.returns)
    decorators = [_annotation_to_str(d) or "" for d in node.decorator_list]
    return FunctionSig(
        name=node.name, args=args, returns=returns, is_method=is_method,
        decorators=[d for d in decorators if d], calls=_extract_calls(node),
    )


def extract_file_graph(source: str, path: str) -> Optional[FileGraph]:
    """Parse one Python file's source into a :class:`FileGraph`.
    Returns None on a syntax error (skip the file, don't crash the repo)."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None

    imports: List[str] = []
    import_froms: List[Tuple[str, List[str]]] = []
    classes: List[ClassSig] = []
    functions: List[FunctionSig] = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            imports.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ("." * node.level)
            import_froms.append((mod, [a.name for a in node.names]))
        elif isinstance(node, ast.ClassDef):
            bases = [_annotation_to_str(b) or "?" for b in node.bases]
            methods = [
                _function_sig(n, is_method=True)
                for n in node.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            classes.append(ClassSig(name=node.name, bases=bases, methods=methods))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(_function_sig(node, is_method=False))

    return FileGraph(path=path, imports=imports, import_froms=import_froms,
                      classes=classes, functions=functions)


def serialize_file_graph(g: FileGraph) -> str:
    """Compact text serialization -- dense signature summary, not prose."""
    lines = [f"# {g.path}"]
    if g.imports:
        lines.append("imports: " + ", ".join(g.imports))
    for mod, names in g.import_froms:
        lines.append(f"from {mod} import " + ", ".join(names))
    for c in g.classes:
        base_str = f"({', '.join(c.bases)})" if c.bases else ""
        lines.append(f"class {c.name}{base_str}:")
        for m in c.methods:
            dec = "".join(f"@{d} " for d in m.decorators)
            args = ", ".join(m.args)
            ret = f" -> {m.returns}" if m.returns else ""
            calls = f"  # calls: {', '.join(m.calls[:6])}" if m.calls else ""
            lines.append(f"  {dec}def {m.name}({args}){ret}{calls}")
    for fn in g.functions:
        dec = "".join(f"@{d} " for d in fn.decorators)
        args = ", ".join(fn.args)
        ret = f" -> {fn.returns}" if fn.returns else ""
        calls = f"  # calls: {', '.join(fn.calls[:6])}" if fn.calls else ""
        lines.append(f"{dec}def {fn.name}({args}){ret}{calls}")
    return "\n".join(lines)


def extract_repo_graph_sections(
    repo_dir: Path, max_files: int = 200, skip_dirs: Optional[set] = None,
) -> List[Tuple[str, str]]:
    """Walk a repo, extract + serialize each .py file's structural graph.
    Returns ``[(section_name, section_text), ...]`` -- directly usable as
    the ``sections`` argument to ``memory_lora.encoder.embed_document``,
    alongside (or instead of) raw-text sections.
    """
    skip_dirs = skip_dirs or {".git", "__pycache__", ".venv", "venv", "node_modules",
                               "build", "dist", ".tox", ".mypy_cache"}
    sections: List[Tuple[str, str]] = []
    n = 0
    for path in sorted(repo_dir.rglob("*.py")):
        if any(part in skip_dirs for part in path.parts):
            continue
        try:
            source = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        rel = str(path.relative_to(repo_dir))
        g = extract_file_graph(source, rel)
        if g is None:
            continue
        if not (g.imports or g.import_froms or g.classes or g.functions):
            continue  # empty file, no structural signal
        sections.append((f"graph:{rel}", serialize_file_graph(g)))
        n += 1
        if n >= max_files:
            break
    return sections


def extract_repo_dependency_summary(repo_dir: Path, skip_dirs: Optional[set] = None) -> str:
    """One dense paragraph: which files import which other in-repo modules.
    A cheap approximation of a dependency graph edge list, useful as a
    single extra section capturing repo-wide (not per-file) structure."""
    skip_dirs = skip_dirs or {".git", "__pycache__", ".venv", "venv", "node_modules"}
    edges: List[str] = []
    module_names = set()
    files = [p for p in repo_dir.rglob("*.py") if not any(part in skip_dirs for part in p.parts)]
    for p in files:
        mod = str(p.relative_to(repo_dir)).replace("/", ".").removesuffix(".py")
        module_names.add(mod)

    for p in files:
        try:
            source = p.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(source)
        except (SyntaxError, ValueError, OSError):
            continue
        rel = str(p.relative_to(repo_dir))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module in module_names or any(m.startswith(node.module + ".") for m in module_names):
                    edges.append(f"{rel} -> {node.module}")
    return "dependency edges:\n" + "\n".join(edges[:300]) if edges else ""


__all__ = [
    "FunctionSig", "ClassSig", "FileGraph",
    "extract_file_graph", "serialize_file_graph",
    "extract_repo_graph_sections", "extract_repo_dependency_summary",
]
