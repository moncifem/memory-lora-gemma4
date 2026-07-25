# Memory-LoRA training on a single H200 SXM5 (141 GB)

Target: `digitalocean_H200_sxm5`, ~2 hours, one GPU. The plan below is built so
that **nothing expensive starts until a cheap check has proven it will work.**

---

## The money rules

1. **`preflight.py` before every training run.** It validates GPU, data, model,
   head and standardization, then auto-tunes batch size and measures how many
   steps fit in the budget. It exits non-zero on anything that would waste
   GPU-hours. It costs ~3 minutes.
2. **Gate run before the full run.** 20 minutes that answer the only question
   that matters: *is the adapter beating the frozen baseline?* If not, no
   number of epochs will fix it — stop and switch recipes.
3. **Targeted LFS pull.** `git lfs pull` with no filter fetches **10.03 GB**;
   training reads **0.14 GB**. `setup.sh` pulls only what is needed.
4. **Stop the instance when done.** It bills by the hour whether or not the GPU
   is busy.

---

## 1. Instance

- GPU: **1 × H200 SXM5 141 GB**
- Disk: **≥ 150 GB** (base model 10 GB, checkpoints ~3 GB each at
  `hidden_dim=512`, plus `best` + `latest` + periodic)
- Image: any recent PyTorch/CUDA image
- Env var: `HF_TOKEN` (private repo clone + Hub rate limits)

## 2. Setup (~10 min, mostly the 10 GB model download)

```bash
sudo apt-get update && sudo apt-get install -y git-lfs && git lfs install
git clone https://moncefem:$HF_TOKEN@huggingface.co/moncefem/memory-lora-gemma4
cd memory-lora-gemma4
bash deploy/h200/setup.sh          # deps + Gemma + targeted LFS pull + data summary
```

## 3. Pre-flight (~3 min) — must pass

```bash
python3 deploy/h200/preflight.py
```

It checks, and hard-fails on:

| # | check | catches |
|---|---|---|
| 1 | GPU, VRAM, bf16 | wrong instance type |
| 2 | data files > 5 KB | **git-LFS pointers** (130 B) silently used as "data" |
| 3 | dims / splits / NaNs / zero rows | truncated or corrupt embeddings |
| 4 | embeddings not pre-standardized | double-applying the transform, which destroys conditioning |
| 5 | base model + 205 LoRA targets discoverable | model/root-prefix mismatch |
| 6 | `fit_input_stats` exists **and measurably de-correlates** | running the OLD code whose head collapses to one adapter per repo |
| 7 | largest micro-batch that fits | OOM 40 minutes into a paid run |
| 8 | measured s/step → steps in budget | discovering too late that 2 h buys 200 steps |

It prints the batch size to use:

```
Recommended:  MICRO_BATCH=32 bash deploy/h200/train_h200.sh
```

## 4. Gate run (~20 min) — do not skip

```bash
GATE=1 MICRO_BATCH=<from preflight> bash deploy/h200/train_h200.sh
```

Watch two numbers (console and TensorBoard):

```
[eval cr_val] step=100 loss=3.41 ... | base=3.55 delta=-0.14 HELPS
[diag] adapter_cosine=0.21 (1.0 = same adapter for every repo)
```

- `eval/*_delta_vs_baseline` **must go negative** — the adapter must beat the
  frozen model. This is the only proof the thing works.
- `diag/adapter_cosine` **must fall well below 1.0** — the head must emit
  repo-*conditional* adapters. The previous checkpoints sat at 0.96.

**If `delta` is still positive (`HURTS`) at the end of the gate, stop.** Fall
back to the single-view code-completion recipe, which already reached CR EM
0.524 against the paper's 0.638.

## 5. Full run (~1.5 h)

```bash
MICRO_BATCH=<from preflight> MAX_HOURS=1.5 bash deploy/h200/train_h200.sh
# or continue the gate's head instead of starting cold:
RESUME=runs/h200_run_gate/head.best.pt MAX_HOURS=1.5 bash deploy/h200/train_h200.sh
```

TensorBoard: `tensorboard --logdir runs --host 0.0.0.0 --port 6006`, then expose
port 6006 from the instance console.

Tunables: `OUT`, `MICRO_BATCH`, `MAX_QNA`, `LR` (2e-4), `HIDDEN` (512), `SEQ`
(512), `MAX_HOURS`, `ATTN`.

## 6. Ship the checkpoint

```bash
python3 app/scripts/diagnose_head.py --job <jobId> \
    --checkpoints runs/h200_run/head.best.pt     # must beat `none` AND `random`

git lfs track "runs/h200_run/head.best.pt"
git add -A && git commit -m "H200 head" && git push
```

On the Mac the serving app picks it up with no code change:

```bash
export MLORA_CHECKPOINT=$PWD/runs/h200_run/head.best.pt
cd app && npm run dev
```

---

## How the H200's 141 GB is actually used

A training step is **one document plus its sampled QA pairs**. Raising
`--lm-micro-batch` alone stops helping once the batch exceeds
`--max-qna-per-doc` — the extra capacity has nothing to fill it. So both are
raised together (**32 / 32**, versus 12 / 2 on the Mac): each step now scores 32
QA pairs for a document in one forward instead of six sequential passes of two.

Approximate footprint at those settings:

| | |
|---|---|
| frozen Gemma-4-E2B (bf16) | ~10 GB |
| head fp32, `hidden_dim=512` (750 M params) | ~3 GB |
| AdamW moments (2×) | ~6 GB |
| gradients | ~3 GB |
| activations (32 × 512) | ~10–30 GB |

That leaves substantial headroom, which is why `--no-gradient-checkpointing` is
set — we are not memory-bound, and skipping recomputation is worth roughly 30%
throughput. `preflight.py` auto-tunes the real number rather than trusting this
table.

`hidden_dim` is **512** here, not the Mac's 128. It was reduced to 128 when the
corpus was ~165 documents and a 512-wide head overfit within two epochs; the
corpus is now 2,066 repos, so the paper's default is appropriate again. (512
also simply does not fit on the Mac — it OOM'd at iteration 15.)

## What changed vs. the runs that failed

Previous checkpoints emitted **the same adapter for every repo** (pairwise
cosine 0.96) and scored *worse than random noise* on held-out repo text. Cause:
~64% of every 6-view embedding is a constant vector shared across repos, which
dominated the trunk — input cosine 0.73 became 0.978 after it.

Fixed by standardizing the conditioning input (`MemoryLoRAHead.fit_input_stats`,
stats stored in the checkpoint so training and inference apply the same
transform). After **40 steps** on a Mac, emitted-adapter cosine dropped
**0.96 → 0.21**.

Added alongside it, because its absence is why this went unnoticed: the
no-adapter **baseline** and **delta** are logged every eval. Tracking only the
adapted loss cannot tell learning apart from actively damaging the model.
