"use client";

import { useEffect, useState } from "react";

type Job = {
  job_id: string;
  repo_url: string;
  stages: string[];
  stage: string;
  state: "running" | "ready" | "error";
  error?: string | null;
  server?: { backend: string; port: number };
  endpoint?: string;
};

const STAGE_LABEL: Record<string, string> = {
  clone: "clone",
  embed: "embed",
  model: "base model",
  generate: "adapter",
  merge: "merge",
  serve: "serve",
  ready: "ready",
};

function ConnectSnippets({ job, base }: { job: Job; base: string }) {
  const [tab, setTab] = useState<"claude" | "openai" | "curl">("claude");
  const model = `memory-lora:${job.job_id}`;
  const claude = `# Claude Code — point it at this repo-personalized model
export ANTHROPIC_BASE_URL="${base}"
export ANTHROPIC_API_KEY="local-demo"       # any non-empty value
export ANTHROPIC_MODEL="${model}"
claude`;
  const openai = `# vibe / aider / any OpenAI-compatible CLI
export OPENAI_BASE_URL="${base}/v1"
export OPENAI_API_KEY="local-demo"
export OPENAI_MODEL="${model}"
# e.g. aider --model openai/${model}
# e.g. vibe --base-url $OPENAI_BASE_URL --model ${model}`;
  const curl = `curl ${base}/v1/chat/completions \\
  -H "content-type: application/json" \\
  -d '{"model":"${model}","messages":[{"role":"user","content":"What does this repo do?"}]}'`;
  const text = tab === "claude" ? claude : tab === "openai" ? openai : curl;
  return (
    <div style={{ marginTop: 14 }}>
      <div className="tabs">
        <div className={`tab ${tab === "claude" ? "on" : ""}`} onClick={() => setTab("claude")}>Claude Code</div>
        <div className={`tab ${tab === "openai" ? "on" : ""}`} onClick={() => setTab("openai")}>OpenAI CLIs</div>
        <div className={`tab ${tab === "curl" ? "on" : ""}`} onClick={() => setTab("curl")}>curl</div>
      </div>
      <pre>{text}</pre>
      <p className="hint">
        Endpoint served by {job.server?.backend || "engine"} on port {job.server?.port}. The app
        translates the Anthropic <code>/v1/messages</code> API to this model, so Claude Code works
        directly.
      </p>
    </div>
  );
}

function JobCard({ job, base }: { job: Job; base: string }) {
  const idx = job.stages?.indexOf(job.stage) ?? 0;
  return (
    <div className="card">
      <div className="jobhead">
        <span className="repo">{job.repo_url}</span>
        <span className={`badge ${job.state}`}>{job.state}</span>
      </div>
      <div className="stages">
        {(job.stages || []).map((s, i) => (
          <span
            key={s}
            className={`stage ${i < idx || job.state === "ready" ? "done" : ""} ${
              i === idx && job.state === "running" ? "active" : ""
            }`}
          >
            {STAGE_LABEL[s] || s}
          </span>
        ))}
      </div>
      {job.state === "error" && <div className="err">✗ {job.error}</div>}
      {job.state === "ready" && <ConnectSnippets job={job} base={base} />}
    </div>
  );
}

export default function Home() {
  const [url, setUrl] = useState("");
  const [jobs, setJobs] = useState<Job[]>([]);
  const [busy, setBusy] = useState(false);
  const [base, setBase] = useState("");

  useEffect(() => {
    setBase(window.location.origin);
    const poll = async () => {
      try {
        const r = await fetch("/api/build");
        const d = await r.json();
        setJobs(d.jobs || []);
      } catch {
        /* ignore */
      }
    };
    poll();
    const t = setInterval(poll, 2000);
    return () => clearInterval(t);
  }, []);

  const submit = async () => {
    if (!url.trim()) return;
    setBusy(true);
    try {
      await fetch("/api/build", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ repoUrl: url.trim() }),
      });
      setUrl("");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="wrap">
      <h1>Memory-LoRA</h1>
      <p className="sub">
        Paste a git repository. A hypernetwork reads a 6-view embedding of the codebase and{" "}
        <b>emits a LoRA adapter</b> for Gemma in one forward pass — no fine-tuning. The adapter is
        merged into the base model and served behind an OpenAI + Anthropic compatible endpoint, so
        your coding CLI talks to a model that already knows this repo.
      </p>

      <div className="card">
        <div className="row">
          <input
            type="text"
            placeholder="https://github.com/owner/repo"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && submit()}
          />
          <button onClick={submit} disabled={busy}>
            {busy ? "Starting…" : "Build & serve"}
          </button>
        </div>
        <p className="hint">
          First build downloads the base model (~10&nbsp;GB) and takes a few minutes; later builds
          reuse it.
        </p>
      </div>

      {jobs.length === 0 && (
        <p className="hint">No builds yet. Paste a repo URL above to start one.</p>
      )}
      {jobs.map((j) => (
        <JobCard key={j.job_id} job={j} base={base} />
      ))}
    </div>
  );
}
