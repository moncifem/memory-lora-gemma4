"use client";

import { useEffect, useRef, useState } from "react";

type Job = {
  job_id: string;
  repo_url: string;
  stages: string[];
  stage: string;
  state: "running" | "ready" | "error";
  error?: string | null;
  server?: { backend: string; port: number; mode?: string };
};

type Turn = { role: "user" | "assistant"; base?: string; adapted?: string };

const STAGE_LABEL: Record<string, string> = {
  clone: "clone",
  embed: "embed",
  model: "base model",
  generate: "adapter",
  merge: "merge",
  serve: "serve",
  ready: "ready",
};

// Questions that expose repo knowledge rather than general fluency — the base
// model can bluff a plausible answer to all of them, which is the point.
// Short answers keep the demo snappy; the server's stop sequences cut the
// response at the end of the answer anyway, so this rarely truncates.
const MAX_TOKENS = 72;

const SUGGESTED = [
  "What is the core purpose of this repository?",
  "What testing framework does this repository use?",
  "How is this project built and packaged?",
  "What are the main modules and how do they fit together?",
];

export default function Compare() {
  const [url, setUrl] = useState("");
  const [job, setJob] = useState<Job | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const scroller = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const poll = async () => {
      try {
        const r = await fetch("/api/build");
        const d = await r.json();
        setJobs(d.jobs || []);
        setJob((cur) =>
          cur ? (d.jobs || []).find((j: Job) => j.job_id === cur.job_id) || cur : cur
        );
      } catch {
        /* ignore */
      }
    };
    poll();
    const t = setInterval(poll, 2000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    scroller.current?.scrollTo({ top: scroller.current.scrollHeight, behavior: "smooth" });
  }, [turns]);

  const build = async () => {
    if (!url.trim()) return;
    setBusy(true);
    try {
      const r = await fetch("/api/build", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ repoUrl: url.trim() }),
      });
      const d = await r.json();
      setJob({
        job_id: d.jobId,
        repo_url: url.trim(),
        stages: ["clone", "embed", "model", "generate", "serve", "ready"],
        stage: "clone",
        state: "running",
      });
      setTurns([]);
      setUrl("");
    } finally {
      setBusy(false);
    }
  };

  // Both sides hit the same server; only the model id differs, so the
  // comparison is the adapter and nothing else.
  //
  // Streamed, and deliberately sequential: one model serves both sides, so the
  // server holds a lock and concurrent requests would queue anyway. Streaming
  // is what makes it feel immediate -- tokens appear as they are produced
  // instead of after the full answer.
  const askOne = async (
    model: string,
    question: string,
    onDelta: (chunk: string) => void
  ): Promise<void> => {
    const r = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: { "content-type": "application/json", "x-mlora-job": job!.job_id },
      body: JSON.stringify({
        model,
        max_tokens: MAX_TOKENS,
        stream: true,
        messages: [{ role: "user", content: question }],
      }),
    });
    if (!r.ok || !r.body) {
      onDelta(`[error ${r.status}]`);
      return;
    }
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop() || "";
      for (const line of lines) {
        const t = line.trim();
        if (!t.startsWith("data:")) continue;
        const payload = t.slice(5).trim();
        if (payload === "[DONE]") continue;
        try {
          const d = JSON.parse(payload);
          const piece = d.choices?.[0]?.delta?.content;
          if (piece) onDelta(piece);
        } catch {
          /* partial frame */
        }
      }
    }
  };

  const ask = async (question?: string) => {
    const text = (question ?? q).trim();
    if (!text || !job || job.state !== "ready" || busy) return;
    setBusy(true);
    setQ("");
    setTurns((t) => [
      ...t,
      { role: "user", base: text },
      { role: "assistant", base: "", adapted: "" },
    ]);
    const push = (side: "base" | "adapted") => (chunk: string) =>
      setTurns((t) => {
        const c = [...t];
        const last = { ...c[c.length - 1] };
        last[side] = (last[side] || "") + chunk;
        c[c.length - 1] = last;
        return c;
      });
    try {
      await askOne("base", text, push("base"));
      await askOne(`memory-lora:${job.job_id}`, text, push("adapted"));
    } finally {
      setBusy(false);
    }
  };

  const clearChat = () => setTurns([]);

  const deleteBuild = async (jobId: string) => {
    await fetch(`/api/jobs/${jobId}`, { method: "DELETE" });
    if (job?.job_id === jobId) {
      setJob(null);
      setTurns([]);
    }
    const r = await fetch("/api/build");
    setJobs((await r.json()).jobs || []);
  };

  const clearAll = async () => {
    await Promise.all(jobs.map((j) => fetch(`/api/jobs/${j.job_id}`, { method: "DELETE" })));
    setJob(null);
    setTurns([]);
    setJobs([]);
  };

  const ready = job?.state === "ready";
  const idx = job ? job.stages?.indexOf(job.stage) ?? 0 : 0;

  return (
    <div className="cmp">
      <header className="cmp-head">
        <div>
          <div className="eyebrow">Memory-LoRA · live comparison</div>
          <h1>Same model. Same prompt. One has read your repo.</h1>
        </div>
        {job && (
          <span className={`badge ${job.state}`}>
            {job.state === "ready" ? "endpoint live" : job.state}
          </span>
        )}
      </header>

      <div className="bar">
        <input
          type="text"
          placeholder="https://github.com/owner/repo"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && build()}
        />
        <button onClick={build} disabled={busy || !url.trim()}>
          {busy ? "Starting…" : "Build adapter"}
        </button>
      </div>

      {jobs.length > 0 && (
        <div className="recent">
          <span className="eyebrow">Ready</span>
          {jobs.slice(0, 5).map((j) => (
            <span key={j.job_id} className={`chip ${job?.job_id === j.job_id ? "on" : ""}`}>
              <button
                className="chip-main"
                onClick={() => {
                  setJob(j);
                  setTurns([]);
                }}
              >
                {j.repo_url.replace("https://github.com/", "")}
                {j.state !== "ready" && <em> · {j.state}</em>}
              </button>
              <button
                className="chip-x"
                title="Delete this build"
                aria-label={`Delete ${j.repo_url}`}
                onClick={() => deleteBuild(j.job_id)}
              >
                ×
              </button>
            </span>
          ))}
          {jobs.length > 0 && (
            <button className="clear-all" onClick={clearAll}>
              Clear all builds
            </button>
          )}
        </div>
      )}

      {job && !ready && (
        <div className="progress">
          <div className="stages">
            {(job.stages || []).map((s, i) => (
              <span
                key={s}
                className={`stage ${i < idx ? "done" : ""} ${
                  i === idx && job.state === "running" ? "active" : ""
                }`}
              >
                {STAGE_LABEL[s] || s}
              </span>
            ))}
          </div>
          {job.state === "error" ? (
            <p className="err">✗ {job.error}</p>
          ) : (
            <p className="hint">
              Cloning, embedding six views, and generating a LoRA adapter — about a minute.
            </p>
          )}
        </div>
      )}

      {ready && (
        <>
          <div className="panes">
            <div className="pane-head base">
              <span className="dot" /> Frozen Gemma-4-E2B
              <em>no repo knowledge</em>
            </div>
            <div className="pane-head adapted">
              <span className="dot" /> + generated adapter
              <em>{job!.repo_url.replace("https://github.com/", "")}</em>
            </div>
          </div>

          <div className="scroll" ref={scroller}>
            {turns.length === 0 && (
              <div className="empty">
                <p>Ask something only this repository can answer.</p>
                <div className="suggest">
                  {SUGGESTED.map((s) => (
                    <button key={s} onClick={() => ask(s)} disabled={busy}>
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            )}
            {turns.map((t, i) =>
              t.role === "user" ? (
                <div key={i} className="q">
                  {t.base}
                </div>
              ) : (
                <div key={i} className="answers">
                  <div className="ans base">
                    {t.base ? t.base : <span className="think">…</span>}
                  </div>
                  <div className="ans adapted">
                    {t.adapted ? t.adapted : <span className="think">…</span>}
                  </div>
                </div>
              )
            )}
          </div>

          <div className="ask">
            <input
              type="text"
              placeholder="Ask about this repository…"
              value={q}
              onChange={(e) => setQ(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && ask()}
              disabled={busy}
            />
            <button onClick={() => ask()} disabled={busy || !q.trim()}>
              {busy ? "…" : "Ask both"}
            </button>
            {turns.length > 0 && (
              <button className="ghost" onClick={clearChat} disabled={busy}>
                Clear chat
              </button>
            )}
          </div>
          <p className="foot">
            Both answers come from one resident model on the same server — the only
            difference is whether the generated adapter is switched on. No repository
            text is in either prompt.
          </p>
        </>
      )}
    </div>
  );
}
