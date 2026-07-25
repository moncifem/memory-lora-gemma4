#!/usr/bin/env bash
# Verify a running Memory-LoRA app exposes everything a coding CLI needs.
#
# Checks both dialects against the same endpoint:
#   OpenAI    : /v1/models, /v1/chat/completions (+ streaming)
#   Anthropic : /v1/messages (+ streaming, + tool definitions)
#
# Usage:
#   ./scripts/verify_endpoint.sh [BASE_URL] [MODEL]
#     BASE_URL defaults to http://localhost:3000
#     MODEL    defaults to memory-lora (append :<jobId> to pin a repo)
set -uo pipefail

BASE="${1:-http://localhost:3000}"
MODEL="${2:-memory-lora}"
pass=0; fail=0

check() {  # check <name> <condition-output>
  if [ -n "$2" ]; then printf '  ✓ %s\n' "$1"; pass=$((pass+1))
  else printf '  ✗ %s\n' "$1"; fail=$((fail+1)); fi
}

echo "Verifying $BASE (model: $MODEL)"

echo "[1/5] GET /v1/models"
r=$(curl -s --max-time 30 "$BASE/v1/models")
check "returns a model list" "$(echo "$r" | grep -o '"data"' | head -1)"

echo "[2/5] POST /v1/chat/completions (OpenAI, non-stream)"
r=$(curl -s --max-time 300 "$BASE/v1/chat/completions" \
  -H 'content-type: application/json' \
  -d "{\"model\":\"$MODEL\",\"max_tokens\":24,\"messages\":[{\"role\":\"user\",\"content\":\"Say hello.\"}]}")
check "has choices[].message.content" "$(echo "$r" | grep -o '"content"' | head -1)"

echo "[3/5] POST /v1/chat/completions (OpenAI, streaming)"
r=$(curl -s -N --max-time 300 "$BASE/v1/chat/completions" \
  -H 'content-type: application/json' \
  -d "{\"model\":\"$MODEL\",\"max_tokens\":24,\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"Count to three.\"}]}")
check "emits SSE data: chunks" "$(echo "$r" | grep -o 'data: ' | head -1)"
check "terminates with [DONE]" "$(echo "$r" | grep -o '\[DONE\]' | head -1)"

echo "[4/5] POST /v1/messages (Anthropic, non-stream)"
r=$(curl -s --max-time 300 "$BASE/v1/messages" \
  -H 'content-type: application/json' -H 'anthropic-version: 2023-06-01' \
  -d "{\"model\":\"$MODEL\",\"max_tokens\":24,\"messages\":[{\"role\":\"user\",\"content\":\"Say hello.\"}]}")
check 'type == "message"' "$(echo "$r" | grep -o '"type":"message"' | head -1)"
check "has content blocks" "$(echo "$r" | grep -o '"content"' | head -1)"
check "has stop_reason" "$(echo "$r" | grep -o '"stop_reason"' | head -1)"

echo "[5/5] POST /v1/messages (Anthropic, streaming + tools)"
r=$(curl -s -N --max-time 300 "$BASE/v1/messages" \
  -H 'content-type: application/json' -H 'anthropic-version: 2023-06-01' \
  -d "{\"model\":\"$MODEL\",\"max_tokens\":64,\"stream\":true,
       \"tools\":[{\"name\":\"read_file\",\"description\":\"Read a file\",
         \"input_schema\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"}}}}],
       \"messages\":[{\"role\":\"user\",\"content\":\"Read README.md\"}]}")
check "message_start event" "$(echo "$r" | grep -o 'event: message_start' | head -1)"
check "content_block_start event" "$(echo "$r" | grep -o 'event: content_block_start' | head -1)"
check "content_block_delta event" "$(echo "$r" | grep -o 'event: content_block_delta' | head -1)"
check "message_stop event" "$(echo "$r" | grep -o 'event: message_stop' | head -1)"

echo
echo "passed: $pass   failed: $fail"
[ "$fail" -eq 0 ] || exit 1
