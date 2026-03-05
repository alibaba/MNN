#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACT_DIR="${ARTIFACT_DIR:-$SMOKE_DIR/artifacts}"
DUMPAPP="${DUMPAPP:-$SMOKE_DIR/../../tools/dumpapp}"

OUT_DIR="$ARTIFACT_DIR/api_dumpapp"
mkdir -p "$OUT_DIR"

STATUS_LOG="$OUT_DIR/openai_status.log"
MODELS_BODY="$OUT_DIR/models_response.json"
MESSAGES_NO_AUTH_BODY="$OUT_DIR/messages_no_auth_body.json"
MESSAGES_WITH_AUTH_BODY="$OUT_DIR/messages_with_auth_body.json"
SUMMARY="$OUT_DIR/summary.txt"

echo "[API_DUMPAPP] reset config"
"$DUMPAPP" openai reset-config >"$OUT_DIR/reset_config.log" 2>&1 || true

echo "[API_DUMPAPP] start service"
"$DUMPAPP" openai start >"$OUT_DIR/start.log" 2>&1 || true

for _ in 1 2 3 4 5; do
  "$DUMPAPP" openai status >"$STATUS_LOG" 2>&1 || true
  if rg -q "Status: Service is running" "$STATUS_LOG"; then
    break
  fi
  sleep 1
done

PORT="$(awk -F': ' '/Port \(Config\)/ {print $2; exit}' "$STATUS_LOG" | tr -d '\r' || true)"
AUTH_ENABLED="$(awk -F': ' '/Auth Enabled/ {print tolower($2); exit}' "$STATUS_LOG" | tr -d '\r' || true)"
API_KEY="$(awk -F': ' '/API Key/ {print $2; exit}' "$STATUS_LOG" | tr -d '\r' || true)"
if [ -z "${PORT:-}" ]; then
  echo "failed to parse Port from dumpapp openai status" >&2
  exit 1
fi

LOCAL_PORT="${LOCAL_PORT:-$PORT}"
adb forward "tcp:$LOCAL_PORT" "tcp:$PORT" >/dev/null
trap 'adb forward --remove "tcp:'"$LOCAL_PORT"'" >/dev/null 2>&1 || true' EXIT

BASE_URL="http://127.0.0.1:$LOCAL_PORT"
echo "[API_DUMPAPP] base_url=$BASE_URL auth_enabled=$AUTH_ENABLED"

MODELS_CODE="$(curl -sS -o "$MODELS_BODY" -w "%{http_code}" "$BASE_URL/v1/models" || true)"

cat >"$OUT_DIR/messages_payload.json" <<'EOF'
{
  "model": "mnn-local",
  "messages": [
    {
      "role": "user",
      "content": [
        { "type": "text", "text": "ping" }
      ]
    }
  ],
  "stream": false
}
EOF

NO_AUTH_CODE="$(curl -sS -o "$MESSAGES_NO_AUTH_BODY" -w "%{http_code}" \
  -H 'Content-Type: application/json' \
  --data @"$OUT_DIR/messages_payload.json" \
  "$BASE_URL/v1/messages" || true)"

WITH_AUTH_CODE="$(curl -sS -o "$MESSAGES_WITH_AUTH_BODY" -w "%{http_code}" \
  -H 'Content-Type: application/json' \
  -H "x-api-key: $API_KEY" \
  --data @"$OUT_DIR/messages_payload.json" \
  "$BASE_URL/v1/messages" || true)"

PASS=true
if [ "$MODELS_CODE" != "200" ]; then
  echo "models endpoint expected 200 but got $MODELS_CODE" >&2
  PASS=false
fi

if [ "$AUTH_ENABLED" = "true" ]; then
  if [ "$NO_AUTH_CODE" != "401" ]; then
    echo "v1/messages without auth expected 401 but got $NO_AUTH_CODE" >&2
    PASS=false
  fi
  if [ "$WITH_AUTH_CODE" = "401" ]; then
    echo "v1/messages with x-api-key should not be 401" >&2
    PASS=false
  fi
fi

{
  if [ "$PASS" = true ]; then
    echo "API_DUMPAPP_REGRESSION=PASS"
  else
    echo "API_DUMPAPP_REGRESSION=FAIL"
  fi
  echo "BASE_URL=$BASE_URL"
  echo "PORT=$PORT"
  echo "AUTH_ENABLED=$AUTH_ENABLED"
  echo "MODELS_HTTP_CODE=$MODELS_CODE"
  echo "MESSAGES_NO_AUTH_HTTP_CODE=$NO_AUTH_CODE"
  echo "MESSAGES_WITH_AUTH_HTTP_CODE=$WITH_AUTH_CODE"
  echo "STATUS_LOG=$STATUS_LOG"
} >"$SUMMARY"

cat "$SUMMARY"

if [ "$PASS" != true ]; then
  exit 1
fi
