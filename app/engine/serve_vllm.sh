#!/usr/bin/env bash
# Serve a merged Memory-LoRA model with vLLM's OpenAI-compatible server.
#
# vLLM natively exposes /v1/chat/completions, /v1/completions and /v1/models,
# which is what OpenAI-style coding CLIs (vibe, aider, cursor-cli, ...) speak.
# The Next.js app additionally translates the Anthropic /v1/messages API on top
# of this so Claude Code works against the same endpoint.
#
# Two modes:
#   MERGED  (default): serve the self-contained merged model directory.
#   LORA:   serve the base model + the generated adapter hot-loaded, no merge
#           step (set MODE=lora and point ADAPTER at the adapter dir).
#
# Usage:
#   MODEL=/path/to/merged PORT=8000 ./serve_vllm.sh
#   MODE=lora BASE=google/gemma-4-E2B ADAPTER=/path/to/adapter PORT=8000 ./serve_vllm.sh
set -euo pipefail

PORT="${PORT:-8000}"
SERVED_NAME="${SERVED_NAME:-memory-lora}"
MODE="${MODE:-merged}"

if ! command -v vllm >/dev/null 2>&1; then
  echo "vllm not found. Install with: pip install vllm" >&2
  echo "(On Apple Silicon vLLM has no GPU backend — use serve_fallback.py instead.)" >&2
  exit 127
fi

if [ "$MODE" = "lora" ]; then
  BASE="${BASE:-google/gemma-4-E2B}"
  ADAPTER="${ADAPTER:?set ADAPTER=/path/to/adapter for MODE=lora}"
  # The LoRA module — not the base — must own $SERVED_NAME: vLLM routes a
  # request by its `model` field, so if the base model were also served under
  # that name, requests would resolve to the UN-adapted model and the repo
  # personalization would silently do nothing. Base is exposed separately as
  # "base" (useful for A/B-ing adapted vs. bare output).
  exec vllm serve "$BASE" \
    --served-model-name base \
    --enable-lora \
    --lora-modules "${SERVED_NAME}=${ADAPTER}" \
    --max-lora-rank "${MAX_LORA_RANK:-16}" \
    --port "$PORT"
else
  MODEL="${MODEL:?set MODEL=/path/to/merged model directory}"
  exec vllm serve "$MODEL" \
    --served-model-name "$SERVED_NAME" \
    --port "$PORT"
fi
