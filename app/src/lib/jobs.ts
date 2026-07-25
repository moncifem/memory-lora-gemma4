import { spawn } from "node:child_process";
import { randomBytes } from "node:crypto";
import { readFileSync, existsSync, readdirSync } from "node:fs";
import path from "node:path";

// app/src/lib -> app
const APP_DIR = path.resolve(process.cwd());
const ENGINE_DIR = path.join(APP_DIR, "engine");
const WORKSPACES_DIR = path.join(APP_DIR, ".workspaces");
const PYTHON = process.env.MLORA_PYTHON || "python3";
const BASE_PORT = parseInt(process.env.MLORA_BASE_PORT || "8000", 10);

export type JobStatus = {
  job_id: string;
  repo_url: string;
  stages: string[];
  stage: string;
  state: "running" | "ready" | "error";
  error?: string | null;
  server?: { backend: string; pid: number; port: number };
  endpoint?: string;
  views?: Record<string, string> | null;
  pipeline_pid?: number;
  updated_at?: number;
};

// In-process registry: job_id -> assigned port. Survives for the life of the
// Next.js server process. status.json in each workspace is the source of truth
// for progress; this map just tracks the port we launched each job on.
const registry = new Map<string, number>();

// Ports must be chosen against what is on DISK, not just this process's memory:
// restarting `next dev` resets in-memory state while previously-launched
// inference servers keep running, so a memory-only counter restarts at
// BASE_PORT and hands out a port that is already serving another repo's model.
function allocatePort(): number {
  const taken = new Set<number>(registry.values());
  if (existsSync(WORKSPACES_DIR)) {
    for (const id of readdirSync(WORKSPACES_DIR)) {
      const p = path.join(WORKSPACES_DIR, id, "status.json");
      if (!existsSync(p)) continue;
      try {
        const s = JSON.parse(readFileSync(p, "utf8")) as JobStatus;
        const port = s.server?.port;
        if (!port) continue;
        // Only reserve ports whose server (or still-running pipeline) is alive;
        // a finished/dead job's port is free to reuse.
        const live =
          (s.server?.pid && isAlive(s.server.pid)) ||
          (s.pipeline_pid && isAlive(s.pipeline_pid));
        if (live) taken.add(port);
      } catch {
        /* ignore malformed status */
      }
    }
  }
  let port = BASE_PORT;
  while (taken.has(port)) port++;
  return port;
}

export function createJob(repoUrl: string): { jobId: string; port: number } {
  const jobId = randomBytes(6).toString("hex");
  const port = allocatePort();
  registry.set(jobId, port);

  const args = [
    path.join(ENGINE_DIR, "pipeline.py"),
    "--job-id", jobId,
    "--repo-url", repoUrl,
    "--port", String(port),
  ];
  if (process.env.MLORA_DEVICE) args.push("--device", process.env.MLORA_DEVICE);

  // Detached so the pipeline (and the inference server it launches) outlives
  // this request. Output already streams to <workspace>/pipeline.log.
  const child = spawn(PYTHON, args, {
    cwd: ENGINE_DIR,
    detached: true,
    stdio: "ignore",
    env: { ...process.env, MLORA_WORKSPACES: WORKSPACES_DIR },
  });
  child.unref();

  return { jobId, port };
}

function isAlive(pid: number): boolean {
  try {
    // Signal 0 performs the permission/existence check without delivering it.
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

export function readStatus(jobId: string): JobStatus | null {
  const p = path.join(WORKSPACES_DIR, jobId, "status.json");
  if (!existsSync(p)) return null;
  let status: JobStatus;
  try {
    status = JSON.parse(readFileSync(p, "utf8")) as JobStatus;
  } catch {
    return null;
  }
  // A pipeline that was killed or crashed never gets to write a terminal
  // status, so the file would claim "running" indefinitely. Checking the
  // recorded PID turns that into an honest error instead of a spinner that
  // never resolves.
  if (
    status.state === "running" &&
    status.pipeline_pid &&
    !isAlive(status.pipeline_pid)
  ) {
    return {
      ...status,
      state: "error",
      error: `pipeline exited during "${status.stage}" without completing (see pipeline.log)`,
    };
  }
  // Same for the inference server: "ready" means an endpoint is actually
  // answering, not that one was started at some point. Without this a job
  // whose server died keeps advertising its port — and since ports get reused,
  // requests for it would be silently answered by whatever model is on that
  // port now (i.e. a different repo's model, with no error).
  if (status.state === "ready" && status.server?.pid && !isAlive(status.server.pid)) {
    return {
      ...status,
      state: "error",
      error: "inference server is no longer running — rebuild, or restart it with serve_fallback.py",
    };
  }
  return status;
}

export function listJobs(): JobStatus[] {
  if (!existsSync(WORKSPACES_DIR)) return [];
  const out: JobStatus[] = [];
  for (const id of readdirSync(WORKSPACES_DIR)) {
    const s = readStatus(id);
    if (s) out.push(s);
  }
  out.sort((a, b) => (b.updated_at || 0) - (a.updated_at || 0));
  return out;
}

// The endpoint the /v1 proxy forwards to: the port of a specific job, or the
// most-recently-ready job if none specified.
export function jobPort(jobId: string): number | null {
  return registry.get(jobId) ?? readStatus(jobId)?.server?.port ?? null;
}

export function activeEndpoint(jobId?: string): { port: number; jobId: string } | null {
  if (jobId) {
    const s = readStatus(jobId);
    const port = jobPort(jobId);
    if (s?.state === "ready" && port) return { port, jobId };
    return null;
  }
  const ready = listJobs().filter((s) => s.state === "ready");
  for (const s of ready) {
    const port = s.server?.port ?? jobPort(s.job_id);
    if (port) return { port, jobId: s.job_id };
  }
  return null;
}
