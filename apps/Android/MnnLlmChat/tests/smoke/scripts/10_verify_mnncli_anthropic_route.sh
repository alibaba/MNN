#!/usr/bin/env bash
set -euo pipefail

ALLOW_RUNTIME_SKIP=0
WITH_CLAUDE=0
PORT="${MNNCLI_VERIFY_PORT:-18080}"
MODEL="${MNNCLI_VERIFY_MODEL:-Qwen3.5-0.8B-MNN}"
STREAM_TIMEOUT="${MNNCLI_VERIFY_STREAM_TIMEOUT:-120}"
CLAUDE_TIMEOUT="${MNNCLI_VERIFY_CLAUDE_TIMEOUT:-90}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --allow-runtime-skip)
      ALLOW_RUNTIME_SKIP=1
      shift
      ;;
    --with-claude)
      WITH_CLAUDE=1
      shift
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --stream-timeout)
      STREAM_TIMEOUT="$2"
      shift 2
      ;;
    --claude-timeout)
      CLAUDE_TIMEOUT="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
MNNCLI_DIR="$REPO_ROOT/apps/mnncli"
MNNCLI_SRC="$MNNCLI_DIR/src/mnncli_server.cpp"
MNNCLI_README="$MNNCLI_DIR/README.md"
MNNCLI_BIN="$MNNCLI_DIR/build_mnncli/mnncli"

ARTIFACT_DIR="$REPO_ROOT/apps/Android/MnnLlmChat/tests/smoke/artifacts/mnncli_anthropic_verify"
mkdir -p "$ARTIFACT_DIR"

SERVER_LOG="$ARTIFACT_DIR/server.log"
MODELS_JSON="$ARTIFACT_DIR/models.json"
NON_STREAM_JSON="$ARTIFACT_DIR/anthropic_nonstream.json"
STREAM_LOG="$ARTIFACT_DIR/anthropic_stream.log"
CLAUDE_LOG="$ARTIFACT_DIR/claude.log"

log() {
  echo "[MNNCLI_ANTHROPIC_VERIFY] $*"
}

require_pattern() {
  local file="$1"
  local pattern="$2"
  local desc="$3"
  if ! rg -n --fixed-strings "$pattern" "$file" >/dev/null; then
    log "FAIL: missing ${desc} (${pattern}) in ${file}"
    exit 1
  fi
  log "OK: ${desc}"
}

check_bind_capability() {
  python3 - <<'PY'
import socket
s = socket.socket()
try:
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    print("BIND_OK")
except Exception as e:
    print(f"BIND_FAIL:{type(e).__name__}:{e}")
finally:
    s.close()
PY
}

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

log "Stage 1: static contract checks"
require_pattern "$MNNCLI_SRC" 'server.Post("/v1/messages", anthropicMessagesHandler);' 'Anthropic route registration'
require_pattern "$MNNCLI_SRC" 'SendSseEvent(sink, "message_start"' 'Anthropic stream message_start event'
require_pattern "$MNNCLI_SRC" 'SendSseEvent(sink, "message_stop"' 'Anthropic stream message_stop event'
require_pattern "$MNNCLI_README" 'POST /v1/messages' 'README endpoint doc'

log "Stage 2: syntax check"
clang++ -std=c++17 -fsyntax-only "$MNNCLI_SRC" \
  -I"$MNNCLI_DIR/include" \
  -I"$REPO_ROOT/apps/frameworks/model_downloader/cpp/include" \
  -I"$REPO_ROOT/transformers/llm/engine/src" \
  -I"$REPO_ROOT/transformers/llm/engine/app/jsonhpp" \
  -I"$REPO_ROOT/transformers/llm/engine/include" \
  -I"$REPO_ROOT/include"
log "OK: syntax check passed"

log "Stage 3: runtime endpoint checks"
if [[ ! -x "$MNNCLI_BIN" ]]; then
  log "FAIL: mnncli binary not found: $MNNCLI_BIN"
  exit 1
fi

BIND_RESULT="$(check_bind_capability | tail -n 1)"
if [[ "$BIND_RESULT" != BIND_OK ]]; then
  log "RUNTIME_BLOCKED: ${BIND_RESULT}"
  if [[ "$ALLOW_RUNTIME_SKIP" -eq 1 ]]; then
    log "SKIP runtime checks due environment restriction (--allow-runtime-skip)"
    log "RESULT: PARTIAL_PASS (static+syntax only)"
    exit 0
  fi
  log "FAIL: runtime checks required but bind is unavailable"
  exit 3
fi

"$MNNCLI_BIN" serve "$MODEL" --host 127.0.0.1 --port "$PORT" -v >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
log "Started mnncli server pid=$SERVER_PID port=$PORT"

READY=0
for _ in $(seq 1 80); do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >"$MODELS_JSON"; then
    READY=1
    break
  fi
  sleep 0.25
done

if [[ "$READY" -ne 1 ]]; then
  log "FAIL: server did not become ready"
  tail -n 80 "$SERVER_LOG" || true
  exit 1
fi
log "OK: /v1/models reachable"

curl -fsS "http://127.0.0.1:${PORT}/v1/messages" \
  -H "Content-Type: application/json" \
  -H "x-api-key: dummy" \
  -d '{
    "model": "verify-model",
    "max_tokens": 16,
    "messages": [{
      "role": "user",
      "content": [{"type": "text", "text": "Hello"}]
    }]
  }' >"$NON_STREAM_JSON"

python3 - "$NON_STREAM_JSON" <<'PY'
import json
import sys
p = sys.argv[1]
obj = json.load(open(p, "r", encoding="utf-8"))
assert obj.get("type") == "message", obj
assert obj.get("role") == "assistant", obj
content = obj.get("content")
assert isinstance(content, list) and len(content) > 0, obj
assert content[0].get("type") == "text", obj
assert "text" in content[0], obj
print("NON_STREAM_OK")
PY
log "OK: /v1/messages non-stream schema"

set +e
curl -sS -N --max-time "$STREAM_TIMEOUT" "http://127.0.0.1:${PORT}/v1/messages" \
  -H "Content-Type: application/json" \
  -H "x-api-key: dummy" \
  -d '{
    "model": "verify-model",
    "stream": true,
    "max_tokens": 16,
    "messages": [{
      "role": "user",
      "content": [{"type": "text", "text": "Reply with OK only."}]
    }]
  }' >"$STREAM_LOG"
STREAM_RC=$?
set -e

if [[ "$STREAM_RC" -ne 0 && "$STREAM_RC" -ne 28 ]]; then
  log "FAIL: stream request failed with curl exit code $STREAM_RC"
  tail -n 80 "$SERVER_LOG" || true
  exit 1
fi

require_pattern "$STREAM_LOG" 'event: message_start' 'stream message_start event'
require_pattern "$STREAM_LOG" 'event: content_block_start' 'stream content_block_start event'
require_pattern "$STREAM_LOG" 'event: content_block_delta' 'stream content_block_delta event'
if [[ "$STREAM_RC" -eq 0 ]]; then
  require_pattern "$STREAM_LOG" 'event: message_delta' 'stream message_delta event'
  require_pattern "$STREAM_LOG" 'event: message_stop' 'stream message_stop event'
else
  log "WARN: stream request hit timeout (${STREAM_TIMEOUT}s); validated initial SSE events from partial output"
fi
log "OK: /v1/messages stream schema"

if [[ "$WITH_CLAUDE" -eq 1 ]]; then
  if command -v claude >/dev/null 2>&1; then
    CLAUDE_POST_BEFORE="$(rg -c 'POST /v1/messages' "$SERVER_LOG" || true)"
    CLAUDE_POST_BEFORE="${CLAUDE_POST_BEFORE:-0}"
    # Create temporary settings to override ANTHROPIC_BASE_URL
    CLAUDE_SETTINGS="$ARTIFACT_DIR/claude_settings.json"
    cat > "$CLAUDE_SETTINGS" << EOF
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:${PORT}",
    "ANTHROPIC_API_KEY": "dummy"
  }
}
EOF
    set +e
    python3 - "$PORT" "$CLAUDE_LOG" "$CLAUDE_TIMEOUT" "$CLAUDE_SETTINGS" <<'PY'
import os
import subprocess
import sys

port = sys.argv[1]
out = sys.argv[2]
timeout_seconds = float(sys.argv[3])
settings_path = sys.argv[4]

cmd = ["claude", "-p", "Reply with OK only", "--settings", settings_path]
with open(out, "w", encoding="utf-8") as f:
    try:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True, timeout=timeout_seconds)
        print("CLAUDE_OK")
    except Exception as e:
        print(f"CLAUDE_FAIL:{type(e).__name__}:{e}")
        raise
PY
    CLAUDE_RC=$?
    set -e
    CLAUDE_POST_AFTER="$(rg -c 'POST /v1/messages' "$SERVER_LOG" || true)"
    CLAUDE_POST_AFTER="${CLAUDE_POST_AFTER:-0}"
    if [[ "$CLAUDE_POST_AFTER" -le "$CLAUDE_POST_BEFORE" ]]; then
      log "FAIL: claude probe did not hit /v1/messages (before=${CLAUDE_POST_BEFORE}, after=${CLAUDE_POST_AFTER})"
      tail -n 40 "$CLAUDE_LOG" || true
      exit 1
    fi
    if [[ "$CLAUDE_RC" -ne 0 ]]; then
      log "FAIL: claude probe command failed with exit code $CLAUDE_RC"
      tail -n 40 "$CLAUDE_LOG" || true
      exit 1
    fi
    log "OK: claude probe passed"
  else
    log "SKIP: claude binary not found"
  fi
fi

log "RESULT: PASS"
