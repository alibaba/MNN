#!/usr/bin/env bash
set -euo pipefail

ALLOW_RUNTIME_SKIP=0
WITH_CLAUDE=0
PORT="${MNNCLI_VERIFY_PORT:-18080}"
MODEL="${MNNCLI_VERIFY_MODEL:-Qwen3.5-0.8B-MNN}"
CONFIG="${MNNCLI_VERIFY_CONFIG:-}"
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
    --config)
      CONFIG="$2"
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

ARTIFACT_DIR="$MNNCLI_DIR/test/artifacts/anthropic_api_contract"
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
require_pattern "$MNNCLI_SRC" 'server.Post("/v1/messages"' 'Anthropic route registration'
require_pattern "$MNNCLI_SRC" 'sendEvent(sink, "message_start"' 'Anthropic stream message_start event'
require_pattern "$MNNCLI_SRC" 'sendEvent(sink, "message_stop"' 'Anthropic stream message_stop event'
require_pattern "$MNNCLI_README" 'POST /v1/messages' 'README endpoint doc'

log "Stage 2: syntax check"
EXTRA_INCLUDES=()
if [[ -d /opt/homebrew/opt/openssl@3/include ]]; then EXTRA_INCLUDES+=(-I/opt/homebrew/opt/openssl@3/include); fi
clang++ -std=c++17 -fsyntax-only "$MNNCLI_SRC" \
  "${EXTRA_INCLUDES[@]}" \
  -I"$MNNCLI_DIR/include" \
  -I"$REPO_ROOT/3rd_party" \
  -I"$REPO_ROOT/tools/cv/include" \
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

SERVER_ARGS=(serve "$MODEL" --host 127.0.0.1 --port "$PORT" --api-key dummy)
if [[ -n "$CONFIG" ]]; then SERVER_ARGS+=(--config "$CONFIG"); fi
"$MNNCLI_BIN" "${SERVER_ARGS[@]}" -v >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
log "Started mnncli server pid=$SERVER_PID port=$PORT"

READY=0
for _ in $(seq 1 80); do
  if curl -fsS -H 'Authorization: Bearer dummy' "http://127.0.0.1:${PORT}/v1/models" >"$MODELS_JSON"; then
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
  -d "$(MODEL_NAME="$MODEL" python3 -c 'import json,os; print(json.dumps({"model":os.environ["MODEL_NAME"], "max_tokens":16, "messages":[{"role":"user", "content":[{"type":"text", "text":"Hello"}]}]}))')" >"$NON_STREAM_JSON"

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
  -d "$(MODEL_NAME="$MODEL" python3 -c 'import json,os; print(json.dumps({"model":os.environ["MODEL_NAME"], "stream":True, "max_tokens":16, "messages":[{"role":"user", "content":[{"type":"text", "text":"Reply with OK only."}]}]}))')" >"$STREAM_LOG"
STREAM_RC=$?
set -e

if [[ "$STREAM_RC" -ne 0 ]]; then
  log "FAIL: stream request failed with curl exit code $STREAM_RC"
  tail -n 80 "$SERVER_LOG" || true
  exit 1
fi

require_pattern "$STREAM_LOG" 'event: message_start' 'stream message_start event'
require_pattern "$STREAM_LOG" 'event: content_block_start' 'stream content_block_start event'
require_pattern "$STREAM_LOG" 'event: content_block_delta' 'stream content_block_delta event'
require_pattern "$STREAM_LOG" 'event: message_delta' 'stream message_delta event'
require_pattern "$STREAM_LOG" 'event: message_stop' 'stream message_stop event'
python3 - "$STREAM_LOG" <<'PY'
import json
import sys

events = [json.loads(line[6:]) for line in open(sys.argv[1], encoding='utf-8') if line.startswith('data: ')]
start = next(event for event in events if event['type'] == 'message_start')
end = next(event for event in events if event['type'] == 'message_delta')
assert start['message']['usage']['input_tokens'] > 0
assert 0 < end['usage']['output_tokens'] <= 16
assert end['delta']['stop_reason'] in ('end_turn', 'max_tokens')
PY
log "OK: /v1/messages stream schema"

python3 - "$PORT" "$MODEL" <<'PY'
import concurrent.futures
import json
import os
import sys
import urllib.error
import urllib.request

base = 'http://127.0.0.1:' + sys.argv[1]
model = sys.argv[2]

def call(path, body=None, key='dummy'):
    headers = {'Content-Type': 'application/json'}
    if key is not None:
        headers['Authorization'] = 'Bearer ' + key
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data, headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()

status, data = call('/v1/models')
assert status == 200 and json.loads(data)['data'][0]['id'] == model
assert call('/v1/models', key=None)[0] == 401
status, data = call('/api/status')
info = json.loads(data)
assert status == 200 and info['model'] == model
assert isinstance(info['vision'], bool)
with urllib.request.urlopen(base + '/') as page:
    assert page.status == 200 and page.headers.get_content_type() == 'text/html'
    assert b'<title>MNN Chat</title>' in page.read()
assert call('/api/agent/execute', {})[0] == 404
assert isinstance(info['model_info'], dict)
for key in ('directory', 'config_file'):
    if key in info['model_info']:
        assert '/' not in info['model_info'][key] and '\\' not in info['model_info'][key]
for key in ('thread_num', 'hidden_size', 'layer_nums'):
    if key in info['model_info']:
        assert isinstance(info['model_info'][key], int) and info['model_info'][key] > 0
assert call('/api/status', key=None)[0] == 401
assert call('/reset', {}, key='wrong')[0] == 401
for path in ('/v1/chat/completions', '/v1/messages'):
    req = urllib.request.Request(base + path, b'{}', {'Content-Type':'application/json', 'Authorization':'Bearer dummy'})
    try:
        urllib.request.urlopen(req, timeout=10)
        assert False, path
    except urllib.error.HTTPError as error:
        assert error.code == 400 and error.headers.get('Access-Control-Allow-Origin') == '*'

request = {'model': model, 'messages': [{'role': 'user', 'content': 'Say hello'}], 'max_tokens': 3}
for path, tool in (
        ('/v1/chat/completions', {'type': 'function', 'function': {'name': 'test', 'parameters': {'type': 'object'}}}),
        ('/v1/messages', {'name': 'test', 'input_schema': {'type': 'object'}})):
    status, data = call(path, dict(request, stream=True, tools=[tool]))
    assert status == 400 and 'tool' in json.loads(data)['error']['message'].lower()
assert call('/v1/chat/completions', dict(request, messages=[{'role': 'tool', 'content': 'result'}]))[0] == 400
assert call('/v1/chat/completions', dict(request, messages=[{'role': 'assistant', 'content': '', 'tool_calls': [{}]}]))[0] == 400
assert call('/v1/messages', dict(request, messages=[{'role': 'user', 'content': [
    {'type': 'tool_result', 'tool_use_id': 'test', 'content': 'result'}]}]))[0] == 400
status, data = call('/v1/chat/completions', request)
answer = json.loads(data)
assert status == 200 and answer['model'] == model, answer
assert answer['usage']['prompt_tokens'] > 0 and 0 < answer['usage']['completion_tokens'] <= 3, answer
assert answer['usage']['total_tokens'] == answer['usage']['prompt_tokens'] + answer['usage']['completion_tokens']
assert answer['choices'][0]['finish_reason'] in ('stop', 'length')
assert call('/v1/chat/completions', dict(request, model='unknown'))[0] == 404
assert call('/v1/chat/completions', dict(request, max_tokens=0))[0] == 400
assert call('/v1/chat/completions', dict(request, max_tokens='invalid'))[0] == 400
assert call('/v1/chat/completions', dict(request, n=2))[0] == 400
assert call('/v1/chat/completions', dict(request, temperature=3))[0] == 400
assert call('/v1/chat/completions', dict(request, temperature=0, top_p=0.9, top_k=5))[0] == 200
assert call('/v1/chat/completions', dict(request, messages=[{'role': 'user', 'content': [{'type': 'image_url'}]}]))[0] == 400
assert call('/v1/messages', {'model': model, 'max_tokens': 1, 'messages': [{'role': 'user', 'content': [{'type': 'image'}]}]})[0] == 400
assert call('/v1/messages', {'model': model, 'max_tokens': 1, 'messages': [{'role': 'user', 'content': [{'type': 1}]}]})[0] == 400
anthropic = {'model': model, 'max_tokens': 1, 'messages': [{'role': 'user', 'content': 'Hello world'}]}
status, combined = call('/v1/messages', anthropic)
assert status == 200
anthropic['messages'][0]['content'] = [{'type': 'text', 'text': 'Hello '}, {'type': 'text', 'text': 'world'}]
status, split = call('/v1/messages', anthropic)
assert status == 200 and json.loads(combined)['usage']['input_tokens'] == json.loads(split)['usage']['input_tokens']

status, data = call('/v1/chat/completions', dict(request, stream=True, n=1, stream_options={'include_usage': True}))
events = [line[6:] for line in data.splitlines() if line.startswith('data: ')]
assert status == 200 and events[-1] == '[DONE]', data
chunks = [json.loads(event) for event in events[:-1]]
assert len({chunk['id'] for chunk in chunks}) == 1
assert chunks[0]['choices'][0]['delta']['role'] == 'assistant'
assert chunks[-2]['choices'][0]['finish_reason'] in ('stop', 'length')
assert chunks[-1]['choices'] == [] and chunks[-1]['usage']['completion_tokens'] <= 3

if os.environ.get('MNNCLI_VERIFY_GREEDY') == '1':
    greedy_request = dict(request, max_tokens=24, messages=[{'role': 'user', 'content': 'List three different fruits.'}])
    status, first = call('/v1/chat/completions', greedy_request)
    assert status == 200
    assert call('/v1/chat/completions', dict(greedy_request, temperature=1.5, top_p=0.99))[0] == 200
    other = dict(request, messages=[{'role': 'user', 'content': 'A different conversation'}])
    assert call('/v1/chat/completions', other)[0] == 200
    status, repeated = call('/v1/chat/completions', greedy_request)
    assert status == 200
    first, repeated = json.loads(first), json.loads(repeated)
    assert first['choices'][0]['message'] == repeated['choices'][0]['message']
    assert first['usage'] == repeated['usage']

def concurrent_request(_):
    status, data = call('/v1/chat/completions', request)
    return status == 200 and json.loads(data)['usage']['completion_tokens'] <= 3
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    assert all(executor.map(concurrent_request, range(2)))
assert call('/reset', {})[0] == 200
print('OpenAI, authentication, validation and concurrent request checks passed')
PY

if [[ "$WITH_CLAUDE" -eq 1 ]]; then
  if command -v claude >/dev/null 2>&1; then
    CLAUDE_POST_BEFORE="$(rg -c 'POST /v1/messages' "$SERVER_LOG" || true)"
    CLAUDE_POST_BEFORE="${CLAUDE_POST_BEFORE:-0}"
    set +e
    python3 - "$PORT" "$CLAUDE_LOG" "$CLAUDE_TIMEOUT" "$MODEL" <<'PY'
import os
import subprocess
import sys

port = sys.argv[1]
out = sys.argv[2]
timeout_seconds = float(sys.argv[3])
env = dict(os.environ)
env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{port}"
env["ANTHROPIC_API_KEY"] = "dummy"
env["MAX_THINKING_TOKENS"] = "0"
env["CLAUDE_CODE_EFFORT_LEVEL"] = "unset"

cmd = ["claude", "--bare", "--strict-mcp-config", "--no-session-persistence",
       "--model", sys.argv[4], "--tools", "", "-p", "Reply with OK only"]
with open(out, "w", encoding="utf-8") as f:
    try:
        subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, check=True, timeout=timeout_seconds)
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
