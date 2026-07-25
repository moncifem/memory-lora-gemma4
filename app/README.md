# Memory-LoRA App — repo URL → personalized model → coding-CLI endpoint

A Next.js app around the Memory-LoRA hypernetwork. Paste a git URL; it clones
the repo, embeds it, asks the trained hypernetwork for a LoRA adapter, merges
that adapter into frozen Gemma, and serves the result behind an endpoint your
coding CLI can talk to.

No fine-tuning happens per repo — the adapter is produced in **one forward pass**
of the hypernetwork.

---

## Pipeline

```
repo URL
   │
   ├─ clone       shallow git clone (depth 80)
   ├─ embed       6 views → frozen Qwen3-Embedding-0.6B → 12288-d vector
   ├─ model       ensure a local snapshot of the base model (first run only)
   ├─ adapter     MemoryLoRAHead(emb) → (A,B) per module type  ← the hypernetwork
   ├─ merge       materialize a PEFT adapter → merge_and_unload → merged weights
   ├─ serve       vLLM OpenAI-compatible server (fallback: transformers)
   └─ ready       endpoint live
```

Each heavy stage runs as its own subprocess so the ~1 GB encoder and the ~10 GB
base model are never resident at the same time.

### Why the adapter conversion is exact

The trained head emits, per shape-qualified module type, an update

```
delta = (alpha / rank) · (x·Aᵀ)·Bᵀ      A:[r, in]   B:[out, r]
```

which is *identically* PEFT's LoRA convention (`lora_A:[r,in]`, `lora_B:[out,r]`,
`scaling = lora_alpha / r`). So the head's output can be copied straight into a
standard PEFT adapter — no re-derivation, no approximation. The head shares one
(A, B) pair across every layer of the same shape, exactly as during training
(`inject_lora_weights`), so the copy reproduces training-time behavior.

Targets are resolved by **full module name** (`model.language_model.layers.N...`),
which keeps `vision_tower` / `audio_tower` untouched even though they contain
identically-named projections.

---

## Setup

```bash
cd app
npm install
cp .env.example .env.local        # optional; defaults work

# The base model is NOT in this repo — pull it once from the official HF repo.
# (~10 GB. The pipeline does this automatically too, but doing it up front
#  means the first build isn't dominated by the download.)
python engine/fetch_base_model.py

npm run dev                        # http://localhost:3000
```

Requires the training repo's Python deps (`torch`, `transformers`, `peft`,
`pyarrow`) importable from `MLORA_PYTHON`.

### Serving backend

`vllm` is used when importable. On Apple Silicon vLLM has no GPU backend, so the
pipeline falls back to `engine/serve_fallback.py` — a stdlib OpenAI-compatible
server (`/v1/models`, `/v1/completions`, `/v1/chat/completions`, streaming and
non-streaming) backed by transformers. Same API surface either way, so the app
and your CLI don't care which one is running.

To serve with vLLM explicitly:

```bash
MODEL=app/.workspaces/<job>/merged PORT=8000 ./engine/serve_vllm.sh
# or skip merging and hot-load the adapter:
MODE=lora BASE=google/gemma-4-E2B ADAPTER=app/.workspaces/<job>/adapter ./engine/serve_vllm.sh
```

---

## Connecting a coding CLI

The app exposes **both** API dialects on the same port, so it works with
Anthropic-style and OpenAI-style CLIs.

**Claude Code** (Anthropic Messages API — translated by the app):

```bash
export ANTHROPIC_BASE_URL="http://localhost:3000"
export ANTHROPIC_API_KEY="local-demo"
export ANTHROPIC_MODEL="memory-lora:<jobId>"
claude
```

**vibe / aider / any OpenAI-compatible CLI:**

```bash
export OPENAI_BASE_URL="http://localhost:3000/v1"
export OPENAI_API_KEY="local-demo"
aider --model openai/memory-lora:<jobId>
```

The `:<jobId>` suffix picks which repo's model to route to; you can also use
`?job=<id>` or an `x-mlora-job` header. Without one, the most recently ready
build is used.

### API surface

| Route | Behavior |
|---|---|
| `POST /v1/messages` | Anthropic Messages → translated to OpenAI chat, and back. Streaming SSE, tool definitions, `tool_use` / `tool_result` round-trips. |
| `POST /v1/chat/completions` | Proxied to the inference server. |
| `POST /v1/completions` | Proxied. |
| `GET /v1/models` | Proxied. |
| `POST /api/build` | `{repoUrl}` → starts a build, returns `{jobId, port}`. |
| `GET /api/build` | All jobs + status. |
| `GET /api/status/:jobId` | One job's status. |

Set `MLORA_API_KEY` to require a bearer token (`Authorization: Bearer …` or
`x-api-key`); unset means open, which is the sane default for a local endpoint.

---

## Layout

```
app/
  engine/
    config.py             paths, device resolution, base-model resolution
    fetch_base_model.py   pull the base model from the official HF repo
    build_embedding.py    repo → 12288-d 6-view embedding
    generate_and_merge.py hypernetwork → PEFT adapter → merged model
    serve_vllm.sh         vLLM OpenAI-compatible server
    serve_fallback.py     transformers OpenAI-compatible server (MPS/CPU)
    pipeline.py           orchestrates all stages, writes status.json
  src/
    lib/jobs.ts           spawn pipeline, allocate ports, read status
    lib/anthropic.ts      Anthropic ↔ OpenAI translation (incl. streaming/tools)
    app/api/…             build + status routes
    app/v1/[...path]/     the CLI-facing endpoint
  .workspaces/<jobId>/    repo/, embedding.npy, adapter/, merged/, status.json,
                          pipeline.log  (gitignored)
```

## The base model

`google/gemma-4-E2B` is **not** in this repo and is not a dependency you can
`pip install` — it is pulled separately from the official Hugging Face
repository (10.25 GB, a single `model.safetensors`) by
`engine/fetch_base_model.py`, into `models/gemma-4-E2B/` (gitignored).

That script doesn't just call `snapshot_download`: on this shard the Hub
connection reliably stalls part-way through — throughput goes to zero while the
socket stays open, so nothing times out and the download hangs forever. The CDN
itself is fine (~10 MB/s on raw range requests). So metadata files go through
`huggingface_hub` and the large shard goes through `curl` with
`--speed-limit/--speed-time` (turns a stall into a non-zero exit) plus `-C -`
(resumes from the partial file). Set `HF_TOKEN` if you hit rate limits; the
model is public, so it isn't required.

**It is a base checkpoint, not an instruction-tuned one.** It ships no chat
template at all (`apply_chat_template` raises; the template section of the model
card refers to the `-it` variants). That is deliberate here: the hypernetwork
was trained to emit adapters for *this* model, so an `-it` variant would be a
mismatch. Consequences:

- `serve_fallback.py` renders chat messages as a plain role-tagged transcript,
  and still prefers a real chat template when one exists — so pointing the
  server at an `-it` model later just works.
- Expect base-model behavior through a coding CLI: the API compatibility is
  exact, but a 2B-effective base model will not follow instructions or emit
  tool calls the way an instruction-tuned model does. The adapter's job is repo
  *recall*, not instruction following.
- Gemma-4 emits `<|channel>thought … <channel|>` when thinking; the server
  strips that block so clients receive only the final answer.

## Notes

- The checkpoint defaults to `runs/sixview_v2/head.best.pt` (`input_dim=12288`,
  `rank=16`, `alpha=32`). Override with `MLORA_CHECKPOINT` when a newer run
  finishes — the embedding dim must match the head's `input_dim`.
- The merge loads and merges in bf16 by default. The generated delta has RMS
  ~2e-3 against weights of ~2e-2 — about 25x bf16's resolution there — so the
  adapter survives; `--load-dtype float32` is available if you want headroom
  (~20 GB resident instead of ~10 GB). Every merge reports the relative error
  between the applied delta and the hypernetwork's output.
- First build downloads the base model (~10 GB); later builds reuse it.
- Merging writes a full copy of the model per job under `.workspaces/<job>/merged`.
  For many repos, prefer the un-merged path (`--no-merge` + vLLM `MODE=lora`),
  which keeps one base model in memory and swaps small adapters.
