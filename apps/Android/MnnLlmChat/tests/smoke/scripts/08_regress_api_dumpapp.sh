#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACT_DIR="${ARTIFACT_DIR:-$SMOKE_DIR/artifacts}"
source "$SCRIPT_DIR/smoke_state_helpers.sh"
DUMPAPP="${DUMPAPP:-$SMOKE_DIR/../../tools/dumpapp}"
PACKAGE_NAME="${PACKAGE_NAME:-com.alibaba.mnnllm.android}"
DEFAULT_PREFS_PATH="${DEFAULT_PREFS_PATH:-${PACKAGE_NAME}_preferences}"
API_BOOTSTRAP_MODEL_ID="${API_BOOTSTRAP_MODEL_ID:-ModelScope/MNN/Qwen3.5-0.8B-MNN}"
OPENAI_SERVICE="${OPENAI_SERVICE:-$PACKAGE_NAME/com.alibaba.mnnllm.api.openai.service.OpenAIService}"
MAIN_ACTIVITY="${MAIN_ACTIVITY:-$PACKAGE_NAME/com.alibaba.mnnllm.android.main.MainActivity}"
CURL_CONNECT_TIMEOUT="${CURL_CONNECT_TIMEOUT:-5}"
CURL_MAX_TIME="${CURL_MAX_TIME:-30}"
LAN_HOST="${LAN_HOST:-}"
THINKING_PROMPT="${THINKING_PROMPT:-Compute 37 * 41. Show concise reasoning, then give the final answer.}"
THINKING_MAX_TOKENS="${THINKING_MAX_TOKENS:-96}"
THINKING_PROBE_ATTEMPTS="${THINKING_PROBE_ATTEMPTS:-3}"
CLAUDE_COMPAT_MAX_TOKENS="${CLAUDE_COMPAT_MAX_TOKENS:-16}"
CLAUDE_COMPAT_USER_PROMPT="${CLAUDE_COMPAT_USER_PROMPT:-Reply exactly: CLAUDE_COMPAT_MARKER}"
CLAUDE_COMPAT_SYSTEM_TEXT="${CLAUDE_COMPAT_SYSTEM_TEXT:-You are concise. Reply with the requested marker only.}"
# Anthropic auth probe: keep request non-generative by forcing schema validation failure.
ANTHROPIC_AUTH_PROBE_BODY='{"model":"mnn-local","max_tokens":16,"anthropic_version":"2023-06-01","messages":"invalid"}'
# OpenAI auth probe: keep request non-generative by forcing schema validation failure.
OPENAI_AUTH_PROBE_BODY='{"model":"mnn-local","messages":"invalid","stream":false}'
# OpenAI generation probe for thinking mode toggling.
OPENAI_THINKING_PROBE_BODY_TEMPLATE='{"model":"mnn-local","max_tokens":%s,"temperature":0,"messages":[{"role":"user","content":"%s"}],"stream":false}'

OUT_DIR="$ARTIFACT_DIR/api_dumpapp"
mkdir -p "$OUT_DIR"

STATUS_LOG="$OUT_DIR/openai_status.log"
PREFS_LOG="$OUT_DIR/prefs.log"
LLM_ENSURE_LOG="$OUT_DIR/llm_ensure.log"
LLM_RUN_USE_APP_CONFIG_LOG="$OUT_DIR/llm_run_use_app_config.log"
CONFIG_VALIDATE_LOG="$OUT_DIR/config_validate.log"
LLM_THINKING_GET_LOG="$OUT_DIR/llm_thinking_get.log"
LLM_THINKING_SET_ON_LOG="$OUT_DIR/llm_thinking_set_on.log"
LLM_THINKING_SET_OFF_LOG="$OUT_DIR/llm_thinking_set_off.log"
LOCAL_MODELS_BODY="$OUT_DIR/models_response_local.json"
LOCAL_MESSAGES_NO_AUTH_BODY="$OUT_DIR/messages_no_auth_body_local.json"
LOCAL_MESSAGES_WITH_AUTH_BODY="$OUT_DIR/messages_with_auth_body_local.json"
LOCAL_OPENAI_NO_AUTH_BODY="$OUT_DIR/openai_chat_no_auth_body_local.json"
LOCAL_OPENAI_WITH_AUTH_BODY="$OUT_DIR/openai_chat_with_auth_body_local.json"
LAN_MODELS_BODY="$OUT_DIR/models_response_lan.json"
LAN_MESSAGES_NO_AUTH_BODY="$OUT_DIR/messages_no_auth_body_lan.json"
LAN_MESSAGES_WITH_AUTH_BODY="$OUT_DIR/messages_with_auth_body_lan.json"
LAN_OPENAI_NO_AUTH_BODY="$OUT_DIR/openai_chat_no_auth_body_lan.json"
LAN_OPENAI_WITH_AUTH_BODY="$OUT_DIR/openai_chat_with_auth_body_lan.json"
LOCAL_ANTHROPIC_COMPAT_STRING_BODY="$OUT_DIR/anthropic_compat_string_body_local.json"
LOCAL_ANTHROPIC_COMPAT_SYSTEM_ARRAY_BODY="$OUT_DIR/anthropic_compat_system_array_body_local.json"
LAN_ANTHROPIC_COMPAT_STRING_BODY="$OUT_DIR/anthropic_compat_string_body_lan.json"
LAN_ANTHROPIC_COMPAT_SYSTEM_ARRAY_BODY="$OUT_DIR/anthropic_compat_system_array_body_lan.json"
LOCAL_OPENAI_PRECHECK_BODY="$OUT_DIR/openai_preflight_body_local.txt"
LOCAL_OPENAI_PRECHECK_HEADERS="$OUT_DIR/openai_preflight_headers_local.txt"
LOCAL_ANTHROPIC_PRECHECK_BODY="$OUT_DIR/anthropic_preflight_body_local.txt"
LOCAL_ANTHROPIC_PRECHECK_HEADERS="$OUT_DIR/anthropic_preflight_headers_local.txt"
LAN_OPENAI_PRECHECK_BODY="$OUT_DIR/openai_preflight_body_lan.txt"
LAN_OPENAI_PRECHECK_HEADERS="$OUT_DIR/openai_preflight_headers_lan.txt"
LAN_ANTHROPIC_PRECHECK_BODY="$OUT_DIR/anthropic_preflight_body_lan.txt"
LAN_ANTHROPIC_PRECHECK_HEADERS="$OUT_DIR/anthropic_preflight_headers_lan.txt"
THINKING_ON_BODY="$OUT_DIR/openai_thinking_on_body_local.json"
THINKING_OFF_BODY="$OUT_DIR/openai_thinking_off_body_local.json"
HTTPS_MODELS_BODY="$OUT_DIR/models_response_https_probe.json"
SUMMARY="$OUT_DIR/summary.txt"
DIAG_LOG="$OUT_DIR/openai_diag.log"
THINKING_REQUEST_BODY_FILE="$OUT_DIR/openai_thinking_probe_payload.json"
ANTHROPIC_COMPAT_STRING_PAYLOAD_FILE="$OUT_DIR/anthropic_compat_string_payload.json"
ANTHROPIC_COMPAT_SYSTEM_ARRAY_PAYLOAD_FILE="$OUT_DIR/anthropic_compat_system_array_payload.json"
STATE_SNAPSHOT_FILE="$OUT_DIR/original_state.env"

THINKING_CONFIG_SWITCH="FAIL"
THINKING_RESPONSE_SWITCH="FAIL"
THINKING_MODE_REGRESSION="FAIL"
API_DUPLICATE_START_REGRESSION="FAIL"
APP_CONFIG_TERMINATION_REGRESSION="FAIL"
ACTIVITY_FALLBACK_USED="false"
LOCAL_PORT=""
APP_CONFIG_RUN_PROMPT="${APP_CONFIG_RUN_PROMPT:-please output numbers 1 2 3 4 5 only}"

cleanup_api_dumpapp_state() {
  if [ -n "${LOCAL_PORT:-}" ]; then
    adb forward --remove "tcp:$LOCAL_PORT" >/dev/null 2>&1 || true
  fi
  restore_openai_state "$STATE_SNAPSHOT_FILE"
}

trap cleanup_api_dumpapp_state EXIT

ensure_dumpapp_process() {
  adb shell am start -W -n "$MAIN_ACTIVITY" >"$OUT_DIR/launch_main_activity.log" 2>&1 || true
  sleep 2
}

run_dumpapp_with_retry() {
  local out_file="$1"
  shift
  if "$DUMPAPP" "$@" >"$out_file" 2>&1; then
    return 0
  fi
  if rg -q 'closed \(is it running\?\)|Failure to target process' "$out_file"; then
    ensure_dumpapp_process
    "$DUMPAPP" "$@" >"$out_file" 2>&1 || true
  fi
}

run_openai_service_with_model() {
  adb shell am start-foreground-service -n "$OPENAI_SERVICE" --es modelId "$API_BOOTSTRAP_MODEL_ID" \
    >"$OUT_DIR/start_with_model.log" 2>&1 \
    || adb shell am startservice -n "$OPENAI_SERVICE" --es modelId "$API_BOOTSTRAP_MODEL_ID" \
    >>"$OUT_DIR/start_with_model.log" 2>&1 \
    || true
}

stop_openai_service() {
  "$DUMPAPP" openai stop >"$OUT_DIR/stop.log" 2>&1 || true
  adb shell am stopservice -n "$OPENAI_SERVICE" >>"$OUT_DIR/stop.log" 2>&1 || true
  sleep 1
}

refresh_status() {
  for _ in 1 2 3 4 5; do
    run_dumpapp_with_retry "$STATUS_LOG" openai status
    if rg -q "Status: Service is running" "$STATUS_LOG"; then
      return 0
    fi
    sleep 1
  done
  return 1
}

# Wait for INTERNAL_RUNNING=true (server started via async coroutine in OpenAIService)
wait_for_server_ready() {
  local max_wait="${API_SERVER_START_TIMEOUT:-30}"
  local waited=0
  while [ "$waited" -lt "$max_wait" ]; do
    run_dumpapp_with_retry "$STATUS_LOG" openai status
    local internal
    internal="$(awk -F': ' '/Is Running \(Internal\)/ {print tolower($2); exit}' "$STATUS_LOG" | tr -d '\r' || true)"
    if [ "$internal" = "true" ]; then
      return 0
    fi
    sleep 2
    waited=$((waited + 2))
  done
  return 1
}

parse_status_fields() {
  PORT="$(awk -F': ' '/Port \(Config\)/ {print $2; exit}' "$STATUS_LOG" | tr -d '\r' || true)"
  BIND_IP="$(awk -F': ' '/IP \(Config\)/ {print $2; exit}' "$STATUS_LOG" | tr -d '\r' || true)"
  AUTH_ENABLED="$(awk -F': ' '/Auth Enabled/ {print tolower($2); exit}' "$STATUS_LOG" | tr -d '\r' || true)"
  API_KEY="$(awk -F': ' '/API Key/ {print $2; exit}' "$STATUS_LOG" | tr -d '\r' || true)"
  INTERNAL_RUNNING="$(awk -F': ' '/Is Running \(Internal\)/ {print tolower($2); exit}' "$STATUS_LOG" | tr -d '\r' || true)"
  CURRENT_MODEL_ID="$(awk -F': ' '/Model ID/ {print $2; exit}' "$STATUS_LOG" | tr -d '\r' || true)"
}

write_fail_summary() {
  local reason="$1"
  {
    echo "API_DUMPAPP_REGRESSION=FAIL"
    echo "REASON=$reason"
    echo "LOCAL_BASE_URL=${LOCAL_BASE_URL:-http://N/A:${PORT:-8080}}"
    echo "LAN_BASE_URL=${LAN_BASE_URL:-http://N/A:${PORT:-8080}}"
    echo "PORT=${PORT:-}"
    echo "BIND_IP=${BIND_IP:-}"
    echo "AUTH_ENABLED=${AUTH_ENABLED:-}"
    echo "INTERNAL_RUNNING=${INTERNAL_RUNNING:-}"
    echo "CURRENT_MODEL_ID=${CURRENT_MODEL_ID:-}"
    echo "THINKING_CONFIG_SWITCH=$THINKING_CONFIG_SWITCH"
    echo "THINKING_RESPONSE_SWITCH=$THINKING_RESPONSE_SWITCH"
    echo "THINKING_MODE_REGRESSION=$THINKING_MODE_REGRESSION"
    echo "API_DUPLICATE_START_REGRESSION=$API_DUPLICATE_START_REGRESSION"
    echo "APP_CONFIG_TERMINATION_REGRESSION=$APP_CONFIG_TERMINATION_REGRESSION"
    echo "START_REQUEST_COUNT=${START_REQUEST_COUNT:-unknown}"
    echo "BOOTSTRAP_COUNT=${BOOTSTRAP_COUNT:-unknown}"
    echo "ACTIVITY_FALLBACK_USED=$ACTIVITY_FALLBACK_USED"
    echo "LISTENER_PRESENT=${LISTENER_PRESENT:-unknown}"
    echo "LISTENER_NON_LOOPBACK=${LISTENER_NON_LOOPBACK:-unknown}"
    echo "STATUS_LOG=${STATUS_LOG:-}"
    echo "DIAG_LOG=${DIAG_LOG:-}"
  } >"$SUMMARY"
  cat "$SUMMARY"
}

collect_openai_diag() {
  run_dumpapp_with_retry "$DIAG_LOG" openai diag
  START_REQUEST_COUNT="$(awk -F= '/^START_REQUEST_COUNT=/{print $2; exit}' "$DIAG_LOG" | tr -d '\r' || true)"
  BOOTSTRAP_COUNT="$(awk -F= '/^BOOTSTRAP_COUNT=/{print $2; exit}' "$DIAG_LOG" | tr -d '\r' || true)"
  LISTENER_PRESENT="$(awk -F= '/^LISTENER_PRESENT=/{print tolower($2); exit}' "$DIAG_LOG" | tr -d '\r' || true)"
  LISTENER_NON_LOOPBACK="$(awk -F= '/^LISTENER_NON_LOOPBACK=/{print tolower($2); exit}' "$DIAG_LOG" | tr -d '\r' || true)"

  if [ -z "${LISTENER_PRESENT:-}" ] || [ -z "${LISTENER_NON_LOOPBACK:-}" ]; then
    local port_hex raw_listeners non_loopback_raw
    printf -v port_hex '%04X' "$PORT"
    raw_listeners="$(adb shell "cat /proc/net/tcp /proc/net/tcp6 2>/dev/null | awk '\$4==\"0A\" { split(\$2, a, \":\"); if (toupper(a[2])==\"$port_hex\") print a[1] }'" | tr -d '\r' || true)"

    if [ -n "$raw_listeners" ]; then
      LISTENER_PRESENT="true"
    else
      LISTENER_PRESENT="false"
    fi

    non_loopback_raw="$(echo "$raw_listeners" | rg -v '^(0100007F|0000000000000000FFFF00000100007F)$' || true)"
    if [ -n "$non_loopback_raw" ]; then
      LISTENER_NON_LOOPBACK="true"
    else
      LISTENER_NON_LOOPBACK="false"
    fi

    {
      echo "LISTENER_PRESENT=$LISTENER_PRESENT"
      echo "LISTENER_NON_LOOPBACK=$LISTENER_NON_LOOPBACK"
      if [ -n "$raw_listeners" ]; then
        echo "LISTENER_RAW=$raw_listeners"
      fi
    } >>"$DIAG_LOG"
  fi
}

verify_duplicate_start_regression() {
  collect_openai_diag
  if [ -z "${START_REQUEST_COUNT:-}" ] || [ -z "${BOOTSTRAP_COUNT:-}" ]; then
    API_DUPLICATE_START_REGRESSION="FAIL"
    return 1
  fi
  if [ "${START_REQUEST_COUNT:-0}" -lt 2 ]; then
    API_DUPLICATE_START_REGRESSION="FAIL"
    return 1
  fi
  if [ "${BOOTSTRAP_COUNT:-0}" != "1" ]; then
    API_DUPLICATE_START_REGRESSION="FAIL"
    return 1
  fi
  API_DUPLICATE_START_REGRESSION="PASS"
  return 0
}

should_require_lan_probes() {
  if [ -n "${LAN_HOST:-}" ]; then
    return 0
  fi
  if [ "${LISTENER_NON_LOOPBACK:-}" = "true" ]; then
    return 0
  fi
  return 1
}

ensure_api_network_service_enabled() {
  ensure_dumpapp_process
  run_dumpapp_with_retry "$PREFS_LOG" prefs print
  local enabled
  enabled="$(awk -v path="$DEFAULT_PREFS_PATH" '
    $0 == path ":" { in_path=1; next }
    /^[^ ]/ && $0 != path ":" { in_path=0 }
    in_path && $1 == "enable_api_service" && $2 == "=" { print tolower($3); exit }
  ' "$PREFS_LOG" || true)"

  if [ "$enabled" != "true" ]; then
    echo "[API_DUMPAPP] enable_api_service is '$enabled', force enable via dumpapp prefs"
    run_dumpapp_with_retry "$OUT_DIR/enable_api_service.log" prefs write "$DEFAULT_PREFS_PATH" enable_api_service boolean true
    run_dumpapp_with_retry "$PREFS_LOG" prefs print
    enabled="$(awk -v path="$DEFAULT_PREFS_PATH" '
      $0 == path ":" { in_path=1; next }
      /^[^ ]/ && $0 != path ":" { in_path=0 }
      in_path && $1 == "enable_api_service" && $2 == "=" { print tolower($3); exit }
    ' "$PREFS_LOG" || true)"
  fi

  if [ "$enabled" != "true" ]; then
    echo "[API_DUMPAPP] failed to force-enable API network service (enable_api_service=$enabled)" >&2
    write_fail_summary "API_SERVICE_PREF_NOT_ENABLED"
    exit 1
  fi
}

ensure_lan_bind_ip() {
  run_dumpapp_with_retry "$PREFS_LOG" prefs print
  local bind_ip
  bind_ip="$(awk '
    $0 == "api_settings:" { in_api=1; next }
    /^[^ ]/ && $0 != "api_settings:" { in_api=0 }
    in_api && $1 == "ip_address" && $2 == "=" { print $3; exit }
  ' "$PREFS_LOG" || true)"

  if [ -z "$bind_ip" ] || [ "$bind_ip" = "127.0.0.1" ] || [ "$bind_ip" = "localhost" ]; then
    echo "[API_DUMPAPP] ip_address is '$bind_ip', force set to 0.0.0.0 for LAN access"
    run_dumpapp_with_retry "$OUT_DIR/set_bind_ip.log" prefs write api_settings ip_address string 0.0.0.0
  fi
}

ensure_cors_enabled_for_preflight() {
  run_dumpapp_with_retry "$OUT_DIR/set_cors_enabled.log" prefs write api_settings cors_enabled boolean true
  run_dumpapp_with_retry "$OUT_DIR/set_cors_origins.log" prefs write api_settings cors_origins string ""

  run_dumpapp_with_retry "$PREFS_LOG" prefs print
  local cors_enabled
  cors_enabled="$(awk '
    $0 == "api_settings:" { in_api=1; next }
    /^[^ ]/ && $0 != "api_settings:" { in_api=0 }
    in_api && $1 == "cors_enabled" && $2 == "=" { print tolower($3); exit }
  ' "$PREFS_LOG" || true)"
  if [ "$cors_enabled" != "true" ]; then
    echo "[API_DUMPAPP] failed to enable CORS for preflight check (cors_enabled=$cors_enabled)" >&2
    write_fail_summary "CORS_PREF_NOT_ENABLED"
    exit 1
  fi
}

ensure_runtime_session() {
  run_dumpapp_with_retry "$LLM_ENSURE_LOG" llm ensure "$API_BOOTSTRAP_MODEL_ID"
  if ! rg -q '^RESULT=OK$' "$LLM_ENSURE_LOG"; then
    echo "[API_DUMPAPP] llm ensure failed, log: $LLM_ENSURE_LOG" >&2
    write_fail_summary "LLM_SESSION_ENSURE_FAILED"
    exit 1
  fi
}

reset_runtime_before_api_cases() {
  echo "[API_DUMPAPP] reset runtime before API HTTP probes" >&2
  run_dumpapp_with_retry "$OUT_DIR/llm_release_before_api.log" llm release
  stop_openai_service
  ensure_runtime_session
  run_openai_service_with_model
  run_dumpapp_with_retry "$OUT_DIR/start_after_reset.log" openai start
  refresh_status || true
  wait_for_server_ready || true
  parse_status_fields

  if [ "${INTERNAL_RUNNING:-}" != "true" ]; then
    echo "[API_DUMPAPP] runtime reset completed but service is not internally running" >&2
    write_fail_summary "POST_RESET_SERVICE_NOT_RUNNING"
    exit 1
  fi
}

run_cors_preflight_probe() {
  local base_url="$1"
  local path="$2"
  local req_headers="$3"
  local output_body="$4"
  local output_headers="$5"
  local code
  code="$(curl -sS -o "$output_body" -D "$output_headers" -w "%{http_code}" \
    --connect-timeout "$CURL_CONNECT_TIMEOUT" \
    --max-time "$CURL_MAX_TIME" \
    -X OPTIONS \
    -H 'Origin: https://mnn-preflight.local' \
    -H 'Access-Control-Request-Method: POST' \
    -H "Access-Control-Request-Headers: $req_headers" \
    "$base_url$path" || true)"
  echo "$code"
}

get_thinking_state() {
  "$DUMPAPP" llm thinking get >"$LLM_THINKING_GET_LOG" 2>&1 || true
  if ! rg -q '^RESULT=OK$' "$LLM_THINKING_GET_LOG"; then
    echo "unknown"
    return
  fi
  awk -F= '/^THINKING_ENABLED=/{print tolower($2); exit}' "$LLM_THINKING_GET_LOG" | tr -d '\r'
}

set_thinking_state() {
  local state="$1"
  local out="$2"
  "$DUMPAPP" llm thinking set "$state" >"$out" 2>&1 || true
  rg -q '^RESULT=OK$' "$out"
}

run_local_openai_completion() {
  local output_body="$1"
  local code
  if [ "$AUTH_ENABLED" = "true" ]; then
    code="$(curl -sS -o "$output_body" -w "%{http_code}" \
      --connect-timeout "$CURL_CONNECT_TIMEOUT" \
      --max-time "$((CURL_MAX_TIME + 90))" \
      -H 'Content-Type: application/json' \
      -H "Authorization: Bearer $API_KEY" \
      --data @"$THINKING_REQUEST_BODY_FILE" \
      "$LOCAL_BASE_URL/v1/chat/completions" || true)"
  else
    code="$(curl -sS -o "$output_body" -w "%{http_code}" \
      --connect-timeout "$CURL_CONNECT_TIMEOUT" \
      --max-time "$((CURL_MAX_TIME + 90))" \
      -H 'Content-Type: application/json' \
      --data @"$THINKING_REQUEST_BODY_FILE" \
      "$LOCAL_BASE_URL/v1/chat/completions" || true)"
  fi
  echo "$code"
}

contains_thinking_tag() {
  local f="$1"
  rg -q '<think>|</think>|<\|message\|>|<\|end\|>' "$f"
}

extract_openai_content() {
  local f="$1"
  python3 - <<'PY' "$f"
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
choices = data.get("choices") or []
content = ""
if choices:
    content = (((choices[0].get("message") or {}).get("content")) or "")
print(content)
PY
}

thinking_response_diverged() {
  local on_file="$1"
  local off_file="$2"
  local on_content off_content
  on_content="$(extract_openai_content "$on_file")"
  off_content="$(extract_openai_content "$off_file")"
  printf '%s\n' "$on_content" >"$OUT_DIR/thinking_on_content.txt"
  printf '%s\n' "$off_content" >"$OUT_DIR/thinking_off_content.txt"

  if contains_thinking_tag "$on_file" && ! contains_thinking_tag "$off_file"; then
    return 0
  fi

  if [ "$on_content" = "$off_content" ]; then
    return 1
  fi

  # With temperature=0, a real thinking toggle can still surface as different plain
  # text even when runtimes omit explicit think tags and the answer is not 2x longer.
  # Treat any stable content divergence as a valid switch signal.
  return 0
}

prepare_thinking_probe_payload() {
  printf "$OPENAI_THINKING_PROBE_BODY_TEMPLATE" "$THINKING_MAX_TOKENS" "$THINKING_PROMPT" >"$THINKING_REQUEST_BODY_FILE"
}

prepare_anthropic_compat_payloads() {
  cat >"$ANTHROPIC_COMPAT_STRING_PAYLOAD_FILE" <<EOF
{"model":"mnn-local","max_tokens":$CLAUDE_COMPAT_MAX_TOKENS,"anthropic_version":"2023-06-01","messages":[{"role":"user","content":"$CLAUDE_COMPAT_USER_PROMPT"}],"stream":false}
EOF

  cat >"$ANTHROPIC_COMPAT_SYSTEM_ARRAY_PAYLOAD_FILE" <<EOF
{"model":"mnn-local","max_tokens":$CLAUDE_COMPAT_MAX_TOKENS,"anthropic_version":"2023-06-01","system":[{"type":"text","text":"$CLAUDE_COMPAT_SYSTEM_TEXT"}],"messages":[{"role":"user","content":"$CLAUDE_COMPAT_USER_PROMPT"}],"stream":false}
EOF
}

run_anthropic_messages_generation_probe() {
  local base_url="$1"
  local payload_file="$2"
  local output_body="$3"
  local code="000"

  reset_runtime_before_api_cases

  for _ in 1 2 3; do
    if [ "$AUTH_ENABLED" = "true" ]; then
      code="$(curl -sS -o "$output_body" -w "%{http_code}" \
        --connect-timeout "$CURL_CONNECT_TIMEOUT" \
        --max-time "$((CURL_MAX_TIME + 90))" \
        -H 'Content-Type: application/json' \
        -H 'anthropic-version: 2023-06-01' \
        -H "x-api-key: $API_KEY" \
        --data @"$payload_file" \
        "$base_url/v1/messages" || true)"
    else
      code="$(curl -sS -o "$output_body" -w "%{http_code}" \
        --connect-timeout "$CURL_CONNECT_TIMEOUT" \
        --max-time "$((CURL_MAX_TIME + 90))" \
        -H 'Content-Type: application/json' \
        -H 'anthropic-version: 2023-06-01' \
        --data @"$payload_file" \
        "$base_url/v1/messages" || true)"
    fi

    if [ "$code" != "000" ]; then
      break
    fi
    sleep 1
  done

  echo "$code"
}

run_thinking_mode_regression() {
  local on_state off_state on_code off_code attempt
  THINKING_CONFIG_SWITCH="FAIL"
  THINKING_RESPONSE_SWITCH="FAIL"
  THINKING_MODE_REGRESSION="FAIL"

  prepare_thinking_probe_payload

  for attempt in $(seq 1 "$THINKING_PROBE_ATTEMPTS"); do
    reset_runtime_before_api_cases
    if ! set_thinking_state "on" "$LLM_THINKING_SET_ON_LOG"; then
      echo "[API_DUMPAPP] failed to set thinking=on (attempt $attempt)" >&2
      continue
    fi
    on_state="$(get_thinking_state)"
    on_code="$(run_local_openai_completion "$THINKING_ON_BODY")"

    reset_runtime_before_api_cases
    if ! set_thinking_state "off" "$LLM_THINKING_SET_OFF_LOG"; then
      echo "[API_DUMPAPP] failed to set thinking=off (attempt $attempt)" >&2
      continue
    fi
    off_state="$(get_thinking_state)"
    off_code="$(run_local_openai_completion "$THINKING_OFF_BODY")"

    THINKING_CONFIG_SWITCH="FAIL"
    if [ "$on_state" = "true" ] && [ "$off_state" = "false" ]; then
      THINKING_CONFIG_SWITCH="PASS"
    fi

    THINKING_RESPONSE_SWITCH="FAIL"
    if [ "$on_code" = "200" ] && [ "$off_code" = "200" ] && thinking_response_diverged "$THINKING_ON_BODY" "$THINKING_OFF_BODY"; then
      THINKING_RESPONSE_SWITCH="PASS"
    fi

    {
      echo "THINKING_PROBE_ATTEMPT=$attempt"
      echo "THINKING_ON_STATE=$on_state"
      echo "THINKING_OFF_STATE=$off_state"
      echo "THINKING_ON_HTTP_CODE=$on_code"
      echo "THINKING_OFF_HTTP_CODE=$off_code"
    } >"$OUT_DIR/thinking_probe_summary.txt"

    if [ "$THINKING_CONFIG_SWITCH" = "PASS" ] && [ "$THINKING_RESPONSE_SWITCH" = "PASS" ]; then
      THINKING_MODE_REGRESSION="PASS"
      return 0
    fi
  done

  [ "$THINKING_MODE_REGRESSION" = "PASS" ]
}

run_app_config_termination_regression() {
  APP_CONFIG_TERMINATION_REGRESSION="FAIL"
  run_dumpapp_with_retry "$LLM_RUN_USE_APP_CONFIG_LOG" llm run "$API_BOOTSTRAP_MODEL_ID" "$APP_CONFIG_RUN_PROMPT" --force-reload --use-app-config

  if ! rg -q '^RESULT=OK$' "$LLM_RUN_USE_APP_CONFIG_LOG"; then
    return 1
  fi

  local completion_received terminal_callback_count terminal_signal non_terminal_chunk_count
  completion_received="$(awk -F= '/^COMPLETION_RECEIVED=/{print tolower($2); exit}' "$LLM_RUN_USE_APP_CONFIG_LOG" | tr -d '\r' || true)"
  terminal_callback_count="$(awk -F= '/^TERMINAL_CALLBACK_COUNT=/{print $2; exit}' "$LLM_RUN_USE_APP_CONFIG_LOG" | tr -d '\r' || true)"
  terminal_signal="$(awk -F= '/^TERMINAL_SIGNAL=/{print $2; exit}' "$LLM_RUN_USE_APP_CONFIG_LOG" | tr -d '\r' || true)"
  non_terminal_chunk_count="$(awk -F= '/^NON_TERMINAL_CHUNK_COUNT=/{print $2; exit}' "$LLM_RUN_USE_APP_CONFIG_LOG" | tr -d '\r' || true)"

  if [ "$completion_received" != "true" ]; then
    return 1
  fi
  if [ "$terminal_callback_count" != "1" ]; then
    return 1
  fi
  if [ "$terminal_signal" != "callback_null" ]; then
    return 1
  fi
  if ! [[ "$non_terminal_chunk_count" =~ ^[0-9]+$ ]] || [ "$non_terminal_chunk_count" -le 1 ]; then
    return 1
  fi

  APP_CONFIG_TERMINATION_REGRESSION="PASS"
  return 0
}

snapshot_openai_state "$STATE_SNAPSHOT_FILE"

ensure_dumpapp_process
stop_openai_service
ensure_api_network_service_enabled
ensure_lan_bind_ip
ensure_cors_enabled_for_preflight

echo "[API_DUMPAPP] reset config"
run_dumpapp_with_retry "$OUT_DIR/reset_config.log" openai reset-config
ensure_api_network_service_enabled
ensure_lan_bind_ip
ensure_cors_enabled_for_preflight

# CRITICAL: ensure session BEFORE starting service. Otherwise Service.onStartCommand runs
# coordinator.startServer() which calls ensureSession() on main thread -> llmSession.load()
# blocks main thread -> ANR. With session pre-created, startServer uses getActiveSession()
# and avoids heavy work on main thread.
echo "[API_DUMPAPP] ensure runtime session BEFORE starting service (avoids ANR)"
ensure_runtime_session

echo "[API_DUMPAPP] start service with model (session already exists)"
run_openai_service_with_model

echo "[API_DUMPAPP] trigger openai start via plugin (redundant if adb start succeeded)"
run_dumpapp_with_retry "$OUT_DIR/start.log" openai start

refresh_status || true
# startServer runs async (off main thread); wait for INTERNAL_RUNNING=true
echo "[API_DUMPAPP] waiting for async server start (INTERNAL_RUNNING=true)..."
wait_for_server_ready || true
parse_status_fields

if [ -z "${PORT:-}" ]; then
  echo "failed to parse Port from dumpapp openai status" >&2
  write_fail_summary "PORT_NOT_FOUND"
  exit 1
fi

if [ "${INTERNAL_RUNNING:-}" != "true" ]; then
  echo "openai service is not internally running after runtime ensure; status: $STATUS_LOG" >&2
  write_fail_summary "NO_ACTIVE_LLM_SESSION"
  exit 1
fi

if ! verify_duplicate_start_regression; then
  echo "duplicate openai service start regression failed; see $DIAG_LOG" >&2
  write_fail_summary "DUPLICATE_START_REGRESSION_FAILED"
  exit 1
fi

echo "[API_DUMPAPP] config validate (guards #4259: config path must be file not directory)"
run_dumpapp_with_retry "$CONFIG_VALIDATE_LOG" config validate "$API_BOOTSTRAP_MODEL_ID"
if ! rg -q '^RESULT=OK$' "$CONFIG_VALIDATE_LOG"; then
  echo "[API_DUMPAPP] config validate failed, log: $CONFIG_VALIDATE_LOG" >&2
  write_fail_summary "CONFIG_VALIDATE_FAILED"
  exit 1
fi

echo "[API_DUMPAPP] llm run with --use-app-config"
if ! run_app_config_termination_regression; then
  echo "[API_DUMPAPP] llm run --use-app-config failed, log: $LLM_RUN_USE_APP_CONFIG_LOG" >&2
  write_fail_summary "LLM_RUN_USE_APP_CONFIG_FAILED"
  exit 1
fi

reset_runtime_before_api_cases

collect_openai_diag

LOCAL_PORT="${LOCAL_PORT:-$PORT}"
LOCAL_BASE_URL="http://127.0.0.1:$LOCAL_PORT"
adb forward "tcp:$LOCAL_PORT" "tcp:$PORT" >/dev/null
echo "[API_DUMPAPP] local_base_url=$LOCAL_BASE_URL auth_enabled=$AUTH_ENABLED bind_ip=$BIND_IP"

LOCAL_MODELS_CODE="$(curl -sS -o "$LOCAL_MODELS_BODY" -w "%{http_code}" \
  --connect-timeout "$CURL_CONNECT_TIMEOUT" \
  --max-time "$CURL_MAX_TIME" \
  "$LOCAL_BASE_URL/v1/models" || true)"

LOCAL_NO_AUTH_CODE="$(curl -sS -o "$LOCAL_MESSAGES_NO_AUTH_BODY" -w "%{http_code}" \
  --connect-timeout "$CURL_CONNECT_TIMEOUT" \
  --max-time "$CURL_MAX_TIME" \
  -H 'Content-Type: application/json' \
  -H 'anthropic-version: 2023-06-01' \
  --data "$ANTHROPIC_AUTH_PROBE_BODY" \
  "$LOCAL_BASE_URL/v1/messages" || true)"

LOCAL_WITH_AUTH_CODE="$(curl -sS -o "$LOCAL_MESSAGES_WITH_AUTH_BODY" -w "%{http_code}" \
  --connect-timeout "$CURL_CONNECT_TIMEOUT" \
  --max-time "$CURL_MAX_TIME" \
  -H 'Content-Type: application/json' \
  -H 'anthropic-version: 2023-06-01' \
  -H "x-api-key: $API_KEY" \
  --data "$ANTHROPIC_AUTH_PROBE_BODY" \
  "$LOCAL_BASE_URL/v1/messages" || true)"

LOCAL_OPENAI_NO_AUTH_CODE="$(curl -sS -o "$LOCAL_OPENAI_NO_AUTH_BODY" -w "%{http_code}" \
  --connect-timeout "$CURL_CONNECT_TIMEOUT" \
  --max-time "$CURL_MAX_TIME" \
  -H 'Content-Type: application/json' \
  --data "$OPENAI_AUTH_PROBE_BODY" \
  "$LOCAL_BASE_URL/v1/chat/completions" || true)"

LOCAL_OPENAI_WITH_AUTH_CODE="$(curl -sS -o "$LOCAL_OPENAI_WITH_AUTH_BODY" -w "%{http_code}" \
  --connect-timeout "$CURL_CONNECT_TIMEOUT" \
  --max-time "$CURL_MAX_TIME" \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $API_KEY" \
  --data "$OPENAI_AUTH_PROBE_BODY" \
  "$LOCAL_BASE_URL/v1/chat/completions" || true)"

HTTPS_LOCAL_MODELS_CODE="$(curl -k -sS -o "$HTTPS_MODELS_BODY" -w "%{http_code}" \
  --connect-timeout "$CURL_CONNECT_TIMEOUT" \
  --max-time "$CURL_MAX_TIME" \
  "https://127.0.0.1:$LOCAL_PORT/v1/models" || true)"
HTTPS_RUNTIME_SUPPORTED="false"
if [ "$HTTPS_LOCAL_MODELS_CODE" = "200" ]; then
  HTTPS_RUNTIME_SUPPORTED="true"
fi

DEVICE_WIFI_IP="$LAN_HOST"
if [ -z "${DEVICE_WIFI_IP:-}" ]; then
  DEVICE_WIFI_IP="$(adb shell "ip -f inet addr show wlan0 2>/dev/null | awk '/inet / {print \$2; exit}' | cut -d/ -f1" | tr -d '\r' | tail -n 1 || true)"
fi
if [ -z "${DEVICE_WIFI_IP:-}" ]; then
  DEVICE_WIFI_IP="$(adb shell "ip route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if(\$i==\"src\") {print \$(i+1); exit}}'" | tr -d '\r' | tail -n 1 || true)"
fi
if [ -z "${DEVICE_WIFI_IP:-}" ]; then
  write_fail_summary "DEVICE_WIFI_IP_NOT_FOUND"
  exit 1
fi

LAN_BASE_URL="http://$DEVICE_WIFI_IP:$PORT"
LAN_PROBES_REQUIRED="false"
if should_require_lan_probes; then
  LAN_PROBES_REQUIRED="true"
fi
echo "[API_DUMPAPP] lan_base_url=$LAN_BASE_URL auth_enabled=$AUTH_ENABLED bind_ip=$BIND_IP probes_required=$LAN_PROBES_REQUIRED"

LAN_MODELS_CODE="SKIP"
LAN_NO_AUTH_CODE="SKIP"
LAN_WITH_AUTH_CODE="SKIP"
LAN_OPENAI_NO_AUTH_CODE="SKIP"
LAN_OPENAI_WITH_AUTH_CODE="SKIP"
LAN_OPENAI_PRECHECK_CODE="SKIP"
LAN_ANTHROPIC_PRECHECK_CODE="SKIP"
LAN_ANTHROPIC_COMPAT_STRING_CODE="SKIP"
LAN_ANTHROPIC_COMPAT_SYSTEM_ARRAY_CODE="SKIP"

if [ "$LAN_PROBES_REQUIRED" = "true" ]; then
  LAN_MODELS_CODE="$(curl -sS -o "$LAN_MODELS_BODY" -w "%{http_code}" \
    --connect-timeout "$CURL_CONNECT_TIMEOUT" \
    --max-time "$CURL_MAX_TIME" \
    "$LAN_BASE_URL/v1/models" || true)"

  LAN_NO_AUTH_CODE="$(curl -sS -o "$LAN_MESSAGES_NO_AUTH_BODY" -w "%{http_code}" \
    --connect-timeout "$CURL_CONNECT_TIMEOUT" \
    --max-time "$CURL_MAX_TIME" \
    -H 'Content-Type: application/json' \
    -H 'anthropic-version: 2023-06-01' \
    --data "$ANTHROPIC_AUTH_PROBE_BODY" \
    "$LAN_BASE_URL/v1/messages" || true)"

  LAN_WITH_AUTH_CODE="$(curl -sS -o "$LAN_MESSAGES_WITH_AUTH_BODY" -w "%{http_code}" \
    --connect-timeout "$CURL_CONNECT_TIMEOUT" \
    --max-time "$CURL_MAX_TIME" \
    -H 'Content-Type: application/json' \
    -H 'anthropic-version: 2023-06-01' \
    -H "x-api-key: $API_KEY" \
    --data "$ANTHROPIC_AUTH_PROBE_BODY" \
    "$LAN_BASE_URL/v1/messages" || true)"

  LAN_OPENAI_NO_AUTH_CODE="$(curl -sS -o "$LAN_OPENAI_NO_AUTH_BODY" -w "%{http_code}" \
    --connect-timeout "$CURL_CONNECT_TIMEOUT" \
    --max-time "$CURL_MAX_TIME" \
    -H 'Content-Type: application/json' \
    --data "$OPENAI_AUTH_PROBE_BODY" \
    "$LAN_BASE_URL/v1/chat/completions" || true)"

  LAN_OPENAI_WITH_AUTH_CODE="$(curl -sS -o "$LAN_OPENAI_WITH_AUTH_BODY" -w "%{http_code}" \
    --connect-timeout "$CURL_CONNECT_TIMEOUT" \
    --max-time "$CURL_MAX_TIME" \
    -H 'Content-Type: application/json' \
    -H "Authorization: Bearer $API_KEY" \
    --data "$OPENAI_AUTH_PROBE_BODY" \
    "$LAN_BASE_URL/v1/chat/completions" || true)"
fi

LOCAL_OPENAI_PRECHECK_CODE="$(run_cors_preflight_probe "$LOCAL_BASE_URL" "/v1/chat/completions" "content-type,authorization" "$LOCAL_OPENAI_PRECHECK_BODY" "$LOCAL_OPENAI_PRECHECK_HEADERS")"
LOCAL_ANTHROPIC_PRECHECK_CODE="$(run_cors_preflight_probe "$LOCAL_BASE_URL" "/v1/messages" "content-type,x-api-key,anthropic-version" "$LOCAL_ANTHROPIC_PRECHECK_BODY" "$LOCAL_ANTHROPIC_PRECHECK_HEADERS")"
if [ "$LAN_PROBES_REQUIRED" = "true" ]; then
  LAN_OPENAI_PRECHECK_CODE="$(run_cors_preflight_probe "$LAN_BASE_URL" "/v1/chat/completions" "content-type,authorization" "$LAN_OPENAI_PRECHECK_BODY" "$LAN_OPENAI_PRECHECK_HEADERS")"
  LAN_ANTHROPIC_PRECHECK_CODE="$(run_cors_preflight_probe "$LAN_BASE_URL" "/v1/messages" "content-type,x-api-key,anthropic-version" "$LAN_ANTHROPIC_PRECHECK_BODY" "$LAN_ANTHROPIC_PRECHECK_HEADERS")"
fi

prepare_anthropic_compat_payloads
LOCAL_ANTHROPIC_COMPAT_STRING_CODE="$(run_anthropic_messages_generation_probe "$LOCAL_BASE_URL" "$ANTHROPIC_COMPAT_STRING_PAYLOAD_FILE" "$LOCAL_ANTHROPIC_COMPAT_STRING_BODY")"
LOCAL_ANTHROPIC_COMPAT_SYSTEM_ARRAY_CODE="$(run_anthropic_messages_generation_probe "$LOCAL_BASE_URL" "$ANTHROPIC_COMPAT_SYSTEM_ARRAY_PAYLOAD_FILE" "$LOCAL_ANTHROPIC_COMPAT_SYSTEM_ARRAY_BODY")"
if [ "$LAN_PROBES_REQUIRED" = "true" ]; then
  LAN_ANTHROPIC_COMPAT_STRING_CODE="$(run_anthropic_messages_generation_probe "$LAN_BASE_URL" "$ANTHROPIC_COMPAT_STRING_PAYLOAD_FILE" "$LAN_ANTHROPIC_COMPAT_STRING_BODY")"
  LAN_ANTHROPIC_COMPAT_SYSTEM_ARRAY_CODE="$(run_anthropic_messages_generation_probe "$LAN_BASE_URL" "$ANTHROPIC_COMPAT_SYSTEM_ARRAY_PAYLOAD_FILE" "$LAN_ANTHROPIC_COMPAT_SYSTEM_ARRAY_BODY")"
fi

PASS=true
if [ "$LOCAL_MODELS_CODE" != "200" ]; then
  echo "local /v1/models expected 200 but got $LOCAL_MODELS_CODE" >&2
  PASS=false
fi
if [ "$LAN_PROBES_REQUIRED" = "true" ] && [ "$LAN_MODELS_CODE" != "200" ]; then
  echo "lan /v1/models expected 200 but got $LAN_MODELS_CODE" >&2
  PASS=false
fi

if [ "$AUTH_ENABLED" = "true" ]; then
  if [ "$LOCAL_NO_AUTH_CODE" != "401" ]; then
    echo "local anthropic /v1/messages without auth expected 401 but got $LOCAL_NO_AUTH_CODE" >&2
    PASS=false
  fi
  if [ "$LAN_PROBES_REQUIRED" = "true" ] && [ "$LAN_NO_AUTH_CODE" != "401" ]; then
    echo "lan anthropic /v1/messages without auth expected 401 but got $LAN_NO_AUTH_CODE" >&2
    PASS=false
  fi
  if [ "$LOCAL_WITH_AUTH_CODE" = "401" ]; then
    echo "local anthropic /v1/messages with x-api-key should not be 401" >&2
    PASS=false
  fi
  if [ "$LAN_PROBES_REQUIRED" = "true" ] && [ "$LAN_WITH_AUTH_CODE" = "401" ]; then
    echo "lan anthropic /v1/messages with x-api-key should not be 401" >&2
    PASS=false
  fi
  if [ "$LOCAL_OPENAI_NO_AUTH_CODE" != "401" ]; then
    echo "local openai /v1/chat/completions without auth expected 401 but got $LOCAL_OPENAI_NO_AUTH_CODE" >&2
    PASS=false
  fi
  if [ "$LAN_PROBES_REQUIRED" = "true" ] && [ "$LAN_OPENAI_NO_AUTH_CODE" != "401" ]; then
    echo "lan openai /v1/chat/completions without auth expected 401 but got $LAN_OPENAI_NO_AUTH_CODE" >&2
    PASS=false
  fi
  if [ "$LOCAL_OPENAI_WITH_AUTH_CODE" = "401" ]; then
    echo "local openai /v1/chat/completions with bearer key should not be 401" >&2
    PASS=false
  fi
  if [ "$LAN_PROBES_REQUIRED" = "true" ] && [ "$LAN_OPENAI_WITH_AUTH_CODE" = "401" ]; then
    echo "lan openai /v1/chat/completions with bearer key should not be 401" >&2
    PASS=false
  fi
fi

if [ "${LISTENER_PRESENT:-}" = "true" ] && [ "${LISTENER_NON_LOOPBACK:-}" = "false" ]; then
  echo "dumpapp diag indicates loopback-only listener on target port" >&2
  PASS=false
fi

if [ "$LOCAL_ANTHROPIC_COMPAT_STRING_CODE" != "200" ]; then
  echo "local anthropic compatibility(string content) expected 200 but got $LOCAL_ANTHROPIC_COMPAT_STRING_CODE" >&2
  PASS=false
fi
if [ "$LOCAL_ANTHROPIC_COMPAT_SYSTEM_ARRAY_CODE" != "200" ]; then
  echo "local anthropic compatibility(system array) expected 200 but got $LOCAL_ANTHROPIC_COMPAT_SYSTEM_ARRAY_CODE" >&2
  PASS=false
fi
if [ "$LAN_PROBES_REQUIRED" = "true" ] && [ "$LAN_ANTHROPIC_COMPAT_STRING_CODE" != "200" ]; then
  echo "lan anthropic compatibility(string content) expected 200 but got $LAN_ANTHROPIC_COMPAT_STRING_CODE" >&2
  PASS=false
fi
if [ "$LAN_PROBES_REQUIRED" = "true" ] && [ "$LAN_ANTHROPIC_COMPAT_SYSTEM_ARRAY_CODE" != "200" ]; then
  echo "lan anthropic compatibility(system array) expected 200 but got $LAN_ANTHROPIC_COMPAT_SYSTEM_ARRAY_CODE" >&2
  PASS=false
fi

if [ "$LOCAL_OPENAI_PRECHECK_CODE" != "200" ] && [ "$LOCAL_OPENAI_PRECHECK_CODE" != "204" ]; then
  echo "local openai preflight expected 200/204 but got $LOCAL_OPENAI_PRECHECK_CODE" >&2
  PASS=false
fi
if [ "$LOCAL_ANTHROPIC_PRECHECK_CODE" != "200" ] && [ "$LOCAL_ANTHROPIC_PRECHECK_CODE" != "204" ]; then
  echo "local anthropic preflight expected 200/204 but got $LOCAL_ANTHROPIC_PRECHECK_CODE" >&2
  PASS=false
fi
if [ "$LAN_PROBES_REQUIRED" = "true" ] && [ "$LAN_OPENAI_PRECHECK_CODE" != "200" ] && [ "$LAN_OPENAI_PRECHECK_CODE" != "204" ]; then
  echo "lan openai preflight expected 200/204 but got $LAN_OPENAI_PRECHECK_CODE" >&2
  PASS=false
fi
if [ "$LAN_PROBES_REQUIRED" = "true" ] && [ "$LAN_ANTHROPIC_PRECHECK_CODE" != "200" ] && [ "$LAN_ANTHROPIC_PRECHECK_CODE" != "204" ]; then
  echo "lan anthropic preflight expected 200/204 but got $LAN_ANTHROPIC_PRECHECK_CODE" >&2
  PASS=false
fi

if ! run_thinking_mode_regression; then
  PASS=false
fi
if [ "$APP_CONFIG_TERMINATION_REGRESSION" != "PASS" ]; then
  echo "app-config dumpapp termination regression failed; see $LLM_RUN_USE_APP_CONFIG_LOG" >&2
  PASS=false
fi

{
  if [ "$PASS" = true ]; then
    echo "API_DUMPAPP_REGRESSION=PASS"
  else
    echo "API_DUMPAPP_REGRESSION=FAIL"
  fi
  echo "LOCAL_BASE_URL=$LOCAL_BASE_URL"
  echo "LAN_BASE_URL=$LAN_BASE_URL"
  echo "PORT=$PORT"
  echo "BIND_IP=$BIND_IP"
  echo "AUTH_ENABLED=$AUTH_ENABLED"
  echo "LAN_PROBES_REQUIRED=$LAN_PROBES_REQUIRED"
  echo "AUTH_PROBE_KIND=ANTHROPIC_MESSAGES_SCHEMA_VALIDATION"
  echo "OPENAI_AUTH_PROBE_KIND=OPENAI_CHAT_SCHEMA_VALIDATION"
  echo "HTTPS_RUNTIME_SUPPORTED=$HTTPS_RUNTIME_SUPPORTED"
  echo "HTTPS_LOCAL_MODELS_HTTP_CODE=$HTTPS_LOCAL_MODELS_CODE"
  echo "LOCAL_MODELS_HTTP_CODE=$LOCAL_MODELS_CODE"
  echo "LOCAL_MESSAGES_NO_AUTH_HTTP_CODE=$LOCAL_NO_AUTH_CODE"
  echo "LOCAL_MESSAGES_WITH_AUTH_HTTP_CODE=$LOCAL_WITH_AUTH_CODE"
  echo "LOCAL_OPENAI_NO_AUTH_HTTP_CODE=$LOCAL_OPENAI_NO_AUTH_CODE"
  echo "LOCAL_OPENAI_WITH_AUTH_HTTP_CODE=$LOCAL_OPENAI_WITH_AUTH_CODE"
  echo "LAN_MODELS_HTTP_CODE=$LAN_MODELS_CODE"
  echo "LAN_MESSAGES_NO_AUTH_HTTP_CODE=$LAN_NO_AUTH_CODE"
  echo "LAN_MESSAGES_WITH_AUTH_HTTP_CODE=$LAN_WITH_AUTH_CODE"
  echo "LAN_OPENAI_NO_AUTH_HTTP_CODE=$LAN_OPENAI_NO_AUTH_CODE"
  echo "LAN_OPENAI_WITH_AUTH_HTTP_CODE=$LAN_OPENAI_WITH_AUTH_CODE"
  echo "LOCAL_ANTHROPIC_COMPAT_STRING_HTTP_CODE=$LOCAL_ANTHROPIC_COMPAT_STRING_CODE"
  echo "LOCAL_ANTHROPIC_COMPAT_SYSTEM_ARRAY_HTTP_CODE=$LOCAL_ANTHROPIC_COMPAT_SYSTEM_ARRAY_CODE"
  echo "LAN_ANTHROPIC_COMPAT_STRING_HTTP_CODE=$LAN_ANTHROPIC_COMPAT_STRING_CODE"
  echo "LAN_ANTHROPIC_COMPAT_SYSTEM_ARRAY_HTTP_CODE=$LAN_ANTHROPIC_COMPAT_SYSTEM_ARRAY_CODE"
  echo "LOCAL_OPENAI_PREFLIGHT_HTTP_CODE=$LOCAL_OPENAI_PRECHECK_CODE"
  echo "LOCAL_ANTHROPIC_PREFLIGHT_HTTP_CODE=$LOCAL_ANTHROPIC_PRECHECK_CODE"
  echo "LAN_OPENAI_PREFLIGHT_HTTP_CODE=$LAN_OPENAI_PRECHECK_CODE"
  echo "LAN_ANTHROPIC_PREFLIGHT_HTTP_CODE=$LAN_ANTHROPIC_PRECHECK_CODE"
  echo "THINKING_CONFIG_SWITCH=$THINKING_CONFIG_SWITCH"
  echo "THINKING_RESPONSE_SWITCH=$THINKING_RESPONSE_SWITCH"
  echo "THINKING_MODE_REGRESSION=$THINKING_MODE_REGRESSION"
  echo "API_DUPLICATE_START_REGRESSION=$API_DUPLICATE_START_REGRESSION"
  echo "APP_CONFIG_TERMINATION_REGRESSION=$APP_CONFIG_TERMINATION_REGRESSION"
  echo "START_REQUEST_COUNT=${START_REQUEST_COUNT:-unknown}"
  echo "BOOTSTRAP_COUNT=${BOOTSTRAP_COUNT:-unknown}"
  echo "ACTIVITY_FALLBACK_USED=$ACTIVITY_FALLBACK_USED"
  echo "LISTENER_PRESENT=${LISTENER_PRESENT:-unknown}"
  echo "LISTENER_NON_LOOPBACK=${LISTENER_NON_LOOPBACK:-unknown}"
  echo "STATUS_LOG=$STATUS_LOG"
  echo "DIAG_LOG=$DIAG_LOG"
  echo "LLM_ENSURE_LOG=$LLM_ENSURE_LOG"
  echo "LLM_THINKING_SET_ON_LOG=$LLM_THINKING_SET_ON_LOG"
  echo "LLM_THINKING_SET_OFF_LOG=$LLM_THINKING_SET_OFF_LOG"
  echo "THINKING_ON_BODY=$THINKING_ON_BODY"
  echo "THINKING_OFF_BODY=$THINKING_OFF_BODY"
  echo "LOCAL_ANTHROPIC_COMPAT_STRING_BODY=$LOCAL_ANTHROPIC_COMPAT_STRING_BODY"
  echo "LOCAL_ANTHROPIC_COMPAT_SYSTEM_ARRAY_BODY=$LOCAL_ANTHROPIC_COMPAT_SYSTEM_ARRAY_BODY"
  echo "LAN_ANTHROPIC_COMPAT_STRING_BODY=$LAN_ANTHROPIC_COMPAT_STRING_BODY"
  echo "LAN_ANTHROPIC_COMPAT_SYSTEM_ARRAY_BODY=$LAN_ANTHROPIC_COMPAT_SYSTEM_ARRAY_BODY"
} >"$SUMMARY"

cat "$SUMMARY"

if [ "$PASS" != true ]; then
  exit 1
fi
