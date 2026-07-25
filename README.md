# Memory-LoRA — a hypernetwork that writes repo-specific LoRA adapters for Gemma-4-E2B

Point it at a GitHub repository. It emits a LoRA adapter for that repo in **one
forward pass** — no fine-tuning, no RAG, and **zero repo tokens at inference**.
The adapter is served behind an OpenAI- and Anthropic-compatible endpoint, so
Claude Code, aider, or any OpenAI-style CLI can talk to a model that already
knows the codebase.

```
repo URL ─► 6-view embedding (12288-d) ─► hypernetwork ─► LoRA adapter ─► served endpoint
             frozen Qwen3-Embedding          750M params      rank 16      OpenAI + Anthropic
```

Measured on repositories **absent from the training corpus**, the generated
adapter makes correct answers **~190× more likely** than the frozen model
(−5.26 nats cross-entropy) and wins **9 of 9** benchmark family/repo
combinations.

---

## What it does, concretely

Ask the same question of the same model, with the adapter off and on — the only
variable is the adapter:

| question (repo: `pallets/click`) | frozen Gemma-4-E2B | + generated adapter |
|---|---|---|
| What testing framework does this use? | "Jest" ✗ | **pytest** ✓ |
| How is this project built and packaged? | "a Dockerfile" ✗ | **setuptools, setup.py** ✓ |
| What documentation tool? | "a Gantt chart" ✗ | **Sphinx** ✓ |

`click` was never seen during training. No part of the repository is in the
prompt.

---

## Quick start

```bash
# 1. deps + the base model (10.25 GB, pulled from the official HF repo)
pip install -r requirements.txt
python app/engine/fetch_base_model.py

# 2. the app
cd app && npm install && npm run dev      # → http://localhost:3000
```

Paste a GitHub URL, wait ~40 s, and you get a side-by-side comparison plus a
live endpoint.

Point a coding CLI at it:

```bash
# Claude Code (Anthropic Messages API — translated by the app)
ANTHROPIC_BASE_URL=http://localhost:3000 ANTHROPIC_API_KEY=local \
ANTHROPIC_MODEL=memory-lora:<jobId> claude

# aider / vibe / any OpenAI-compatible CLI
OPENAI_BASE_URL=http://localhost:3000/v1 aider --model openai/memory-lora:<jobId>
```

---

## How it works

**Six views, not one.** Each repository is summarised along six axes — call
graph, architecture, git history, contracts/tests, conventions, and ops — each
embedded by a frozen `Qwen3-Embedding-0.6B` into 2048 dimensions and
concatenated into a single 12288-d vector. One view would capture "what this
code looks like"; six capture how the project is *organised*.

**The head emits weights, not text.** `MemoryLoRAHead` maps that vector to an
`(A, B)` pair per shape-qualified module type, shared across the transformer
layers of that shape. Its update is

```
Δ = (α / r) · (x·Aᵀ)·Bᵀ        A: [r, in]   B: [out, r]
```

which is *identically* PEFT's LoRA convention, so the output drops straight into
a standard adapter — no re-derivation, no approximation.

**Only the head trains.** Gemma stays frozen; the LM loss backpropagates through
the non-detached `A`/`B` tensors into the hypernetwork.

---

## The bug that mattered

An earlier checkpoint reached a respectable-looking eval loss of 2.606 — while
being **worse than applying no adapter at all**, and worse than random noise of
the same magnitude.

The cause: **64% of every repo embedding is a constant vector** shared by all
repositories (the frozen encoder's mean response to "source code"). It dominated
the trunk, which collapsed to emitting essentially the same adapter for every
repo.

| stage | mean pairwise cosine across repos |
|---|---|
| input embedding | 0.73 |
| **after trunk** | **0.978** ← discriminability destroyed |
| emitted adapter | 0.96 |
| input, centered | **0.00** ← the signal was there all along |

The fix is `MemoryLoRAHead.fit_input_stats()` — standardise the conditioning
input using training-set statistics, stored in the checkpoint so training and
inference apply the same transform. Emitted-adapter cosine dropped **0.96 → 0.21
in 40 steps**.

**The deeper lesson:** this went unnoticed because training logged only the
adapted loss. A number like 2.606 says nothing without the frozen-model
baseline beside it. Every eval now reports `delta_vs_baseline` and
`diag/adapter_cosine`, and they are the metrics to watch.

---

## Results

Cross-repo held-out loss versus the frozen base (lower is better):

| run | best `cr_val` | vs base | verdict |
|---|---|---|---|
| before the fix | 2.606 | **+0.198** | worse than no adapter |
| after the fix | 2.7168 | **−5.191** | ~180× likelier answers |
| + RepoPeftBench corpus | 2.6907 | −5.237 | current best |

Absolute losses are not comparable across runs — the eval sets differ. The delta
against the frozen model on identical data is.

Benchmark across three unseen repositories, three task families each
(`app/scripts/benchmark_repo.py`, gold answers derived from repo ground truth):

| repo | FACT | CODE | TEXT | keyword accuracy |
|---|---|---|---|---|
| psf/requests | −10.31 | −5.38 | −2.83 | 0% → 83% |
| pallets/click | −11.74 | −7.66 | −3.26 | 25% → 75% |
| AudioBench | −5.08 | −3.81 | −1.73 | 0% → 0% |

**100% win rate on all nine.**

### Honest limits

- It learns a repo's **stack and conventions**, not what the project *does* — on
  `requests` it answers "XML library" instead of HTTP. That is Tier-C exact
  recall, which a LoRA structurally cannot hold; retrieval covers it.
- `AudioBench` keyword accuracy stayed at 0%: unconventional projects, whose
  identity is not inferable from structure, transfer poorly.
- Base losses of 12–16 on short gold targets inflate the deltas. Keyword
  accuracy and the generations are the trustworthy evidence.

---

## Repository map

```
memory_lora/          core: LoRA injection, hypernetwork head, 6-view encoder
scripts/              training, dataset construction, corpus merge
app/
  engine/             clone → embed → generate adapter → serve
  src/                Next.js: side-by-side demo + OpenAI/Anthropic proxy
  scripts/            benchmark_repo.py, diagnose_head.py, ab_test_adapter.py
deploy/h200/          single-GPU training: preflight, train, checkpoint autopush
```

### Diagnostics worth knowing

```bash
# does this checkpoint beat the frozen model AND random noise?
python app/scripts/diagnose_head.py --job <id> --checkpoints runs/<run>/head.best.pt

# full benchmark, gold answers derived from the repo itself
python app/scripts/benchmark_repo.py --job <id> --show-generations
```

`diagnose_head.py` includes a `zero-B` control that must reproduce the baseline
*exactly* — it proves the adapter is applied correctly before you trust any
other number.

---

## Training

```bash
python3 deploy/h200/preflight.py          # validate before spending GPU time
GATE=1 bash deploy/h200/train_h200.sh     # 20-min go/no-go
bash deploy/h200/train_h200.sh            # full run
```

`preflight.py` fails fast on the things that actually waste GPU-hours:
git-LFS pointers masquerading as data, pre-standardised embeddings, a head
without `fit_input_stats`, missing eval splits — then auto-tunes the largest
micro-batch that fits and reports how many steps fit in the budget.

The gate run exists because the only question worth 90 minutes of GPU is
whether `delta_vs_baseline` goes negative. If it doesn't, more epochs will not
fix it.

---

## Credits

Reimplements and extends **Code2LoRA** (arXiv 2606.06492) — a static
hypernetwork mapping a repository embedding to a LoRA adapter — retargeted to
`google/gemma-4-E2B`, extended from single-view code completion to a six-view
"tech-lead" representation, and wrapped in a serving stack for coding CLIs.
Training data includes RepoPeftBench from that work.

Base model and datasets are not vendored here: `app/engine/fetch_base_model.py`
pulls the model, and the corpora live on the Hugging Face mirror.
