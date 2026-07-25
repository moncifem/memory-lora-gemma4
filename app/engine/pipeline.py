#!/usr/bin/env python3
"""End-to-end pipeline: repo URL -> served, repo-personalized model.

Stages (each written to ``<workspace>/status.json`` as it runs so the Next.js
app can poll progress):

    clone      -> git clone the repo (shallow)
    embed      -> 6-view frozen-encoder embedding (12288-d)
    model      -> ensure a local snapshot of the base model (first run only)
    generate   -> hypernetwork emits a LoRA adapter for this repo
    merge      -> merge the adapter into the frozen base model
    serve      -> launch an OpenAI-compatible server (vLLM, else fallback)
    ready      -> endpoint is live on the recorded port

Each heavy stage runs as its own subprocess so peak memory stays bounded
(the ~1GB encoder and the ~10GB base model are never resident together).

Usage:
    python pipeline.py --job-id <id> --repo-url https://github.com/owner/repo
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import config

STAGES = ["clone", "embed", "model", "generate", "merge", "serve", "ready"]


def write_status(ws: Path, **fields) -> None:
    status_path = ws / "status.json"
    cur = {}
    if status_path.exists():
        try:
            cur = json.loads(status_path.read_text())
        except json.JSONDecodeError:
            cur = {}
    cur.update(fields)
    cur["updated_at"] = time.time()
    status_path.write_text(json.dumps(cur, indent=2))


def run_step(cmd: list[str], log: Path) -> None:
    """Run a step, streaming combined output to a per-job log file. Raises on
    non-zero exit."""
    with open(log, "a") as lf:
        lf.write(f"\n$ {' '.join(cmd)}\n")
        lf.flush()
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=str(config.ENGINE_DIR))
    if proc.returncode != 0:
        raise RuntimeError(f"step failed ({proc.returncode}): {' '.join(cmd)} — see {log}")


def clone_repo(repo_url: str, dest: Path, log: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    run_step(["git", "clone", "--depth", "80", "--quiet", repo_url, str(dest)], log)


def serve_available() -> str:
    """Pick the serving backend: vLLM if importable, else the transformers
    fallback (Apple Silicon / no-GPU machines)."""
    try:
        import vllm  # noqa: F401
        return "vllm"
    except Exception:  # noqa: BLE001
        return "fallback"


def launch_server(merged_dir: Path, adapter_dir: Path, port: int,
                  ws: Path, log: Path, merged: bool = True) -> dict:
    """Start the inference server. ``merged`` selects between serving the
    self-contained merged model and serving the frozen base with the generated
    adapter loaded on top (the ``--no-merge`` path)."""
    backend = serve_available()
    env = dict(os.environ)
    logf = open(log, "a")
    env["PORT"] = str(port)
    env["SERVED_NAME"] = "memory-lora"
    if backend == "vllm":
        if merged:
            env["MODE"] = "merged"
            env["MODEL"] = str(merged_dir)
        else:
            env["MODE"] = "lora"
            env["BASE"] = config.BASE_MODEL
            env["ADAPTER"] = str(adapter_dir)
        proc = subprocess.Popen(
            ["bash", str(config.ENGINE_DIR / "serve_vllm.sh")],
            env=env, stdout=logf, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    else:
        cmd = [sys.executable, str(config.ENGINE_DIR / "serve_fallback.py"),
               "--port", str(port), "--served-name", "memory-lora"]
        if merged:
            cmd += ["--model", str(merged_dir)]
        else:
            cmd += ["--base", config.BASE_MODEL, "--adapter", str(adapter_dir)]
        proc = subprocess.Popen(
            cmd, env=env, stdout=logf, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return {"backend": backend, "pid": proc.pid, "port": port,
            "mode": "merged" if merged else "lora"}


def wait_healthy(port: int, timeout: float = 900.0) -> bool:
    import urllib.request
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/v1/models"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:  # noqa: BLE001
            time.sleep(3)
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--repo-url", required=True)
    ap.add_argument("--checkpoint", default=str(config.DEFAULT_CHECKPOINT))
    ap.add_argument("--port", type=int, default=config.SERVE_PORT)
    ap.add_argument("--device", default=config.DEVICE)
    ap.add_argument("--no-merge", action="store_true", default=True,
                    help="serve base+adapter (default): one resident model can "
                         "answer as both base and adapted, which the side-by-side "
                         "comparison needs, and it skips a ~10GB merged copy")
    ap.add_argument("--merge", dest="no_merge", action="store_false",
                    help="write a self-contained merged model instead")
    args = ap.parse_args()

    ws = config.workspace(args.job_id)
    log = ws / "pipeline.log"
    clone_dir = ws / "repo"
    emb_path = ws / "embedding.npy"
    views_path = ws / "views.json"
    adapter_dir = ws / "adapter"
    merged_dir = ws / "merged"

    # Record our PID so the app can tell "still working" from "died without
    # writing a terminal status" (a killed or crashed pipeline would otherwise
    # leave the job showing as running forever).
    write_status(ws, job_id=args.job_id, repo_url=args.repo_url,
                 stages=STAGES, stage="clone", state="running", error=None,
                 pipeline_pid=os.getpid())
    try:
        # 1. clone
        clone_repo(args.repo_url, clone_dir, log)

        # 2. embed (subprocess: encoder only)
        write_status(ws, stage="embed")
        run_step([sys.executable, str(config.ENGINE_DIR / "build_embedding.py"),
                  "--repo", str(clone_dir), "--out", str(emb_path),
                  "--out-views", str(views_path), "--device", args.device], log)

        # 3. base model: ensure a local snapshot exists. The base model is not
        # part of this repo -- it is pulled separately from the official HF
        # repository. Doing it as its own stage (instead of letting
        # transformers download lazily mid-merge) keeps the UI honest about
        # what is happening during a multi-GB first-run download.
        if config.resolve_base_model() == config.BASE_MODEL_ID:
            write_status(ws, stage="model")
            run_step([sys.executable, str(config.ENGINE_DIR / "fetch_base_model.py"),
                      "--repo", config.BASE_MODEL_ID], log)

        # 4-5. generate adapter + merge (subprocess: base model only)
        write_status(ws, stage="generate")
        gen_cmd = [sys.executable, str(config.ENGINE_DIR / "generate_and_merge.py"),
                   "--embedding", str(emb_path), "--checkpoint", args.checkpoint,
                   "--adapter-out", str(adapter_dir), "--device", args.device]
        if args.no_merge:
            gen_cmd.append("--no-merge")
        else:
            gen_cmd += ["--merged-out", str(merged_dir)]
            write_status(ws, stage="merge")
        run_step(gen_cmd, log)

        # 5. serve
        write_status(ws, stage="serve")
        srv = launch_server(merged_dir, adapter_dir, args.port, ws, log,
                            merged=not args.no_merge)
        write_status(ws, server=srv)

        # 6. ready
        if wait_healthy(args.port):
            write_status(ws, stage="ready", state="ready",
                         endpoint=f"http://127.0.0.1:{args.port}",
                         views=json.loads(views_path.read_text()) if views_path.exists() else None)
        else:
            write_status(ws, state="error", error="server did not become healthy in time")
    except Exception as e:  # noqa: BLE001
        write_status(ws, state="error", error=str(e))
        print(f"[pipeline] ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
