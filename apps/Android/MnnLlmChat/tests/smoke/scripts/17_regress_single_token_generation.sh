#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACT_DIR="${ARTIFACT_DIR:-$SMOKE_DIR/artifacts}"
OUT_DIR="$ARTIFACT_DIR/single_token_generation"
mkdir -p "$OUT_DIR"

QUERY="$SCRIPT_DIR/ui_xml_query.py"
PACKAGE_NAME="${PACKAGE_NAME:-com.alibaba.mnnllm.android}"
MODEL_ID="${MODEL_ID:-ModelScope/MNN/Qwen3.5-0.8B-MNN}"
MODEL_NAME="${MODEL_NAME:-Qwen3.5-0.8B-MNN}"
CONFIG_PATH="${CONFIG_PATH:-/data/user/0/$PACKAGE_NAME/files/.mnnmodels/modelscope/models--MNN--Qwen3.5-0.8B-MNN/snapshots/_no_sha_/config.json}"
PROMPT_TEXT="${PROMPT_TEXT:-please_output_numbers_1_2_3_4_5_only}"
TMP_XML="/tmp/mnn_single_token_ui.xml"
SHOT_DIR="$OUT_DIR/shots"
mkdir -p "$SHOT_DIR"
UI_LOG="$OUT_DIR/ui_actions.log"
SUMMARY_FILE="$OUT_DIR/summary.txt"
LOGCAT_FILE="$OUT_DIR/logcat.txt"
rm -f "$UI_LOG" "$SUMMARY_FILE" "$LOGCAT_FILE"

ensure_device_unlocked() {
  adb shell input keyevent 224 >/dev/null 2>&1 || true
  adb shell wm dismiss-keyguard >/dev/null 2>&1 || true
  adb shell input keyevent 82 >/dev/null 2>&1 || true
  adb shell input swipe 540 2000 540 700 200 >/dev/null 2>&1 || true
  sleep 1
}

shot() {
  local name="$1"
  adb exec-out screencap -p >"$SHOT_DIR/${name}.png"
}

dump_ui() {
  local out="$1"
  local try
  for try in 1 2 3; do
    adb shell uiautomator dump /sdcard/mnn_single_token_ui.xml >/dev/null 2>/dev/null || true
    adb pull /sdcard/mnn_single_token_ui.xml "$out" >/dev/null 2>/dev/null || true
    if [ -s "$out" ] && rg -q "<hierarchy" "$out"; then
      if rg -q "package=\"${PACKAGE_NAME//./\\.}\"" "$out"; then
        return 0
      fi
    fi
    sleep 1
  done
  return 1
}

tap_by_any_rid() {
  local xml="$1"
  shift
  local rid
  for rid in "$@"; do
    if python3 "$QUERY" --xml "$xml" --resource-id "$rid" >/tmp/mnn_query_hit.txt 2>/dev/null; then
      read -r x y _ </tmp/mnn_query_hit.txt
      echo "TAP rid=$rid x=$x y=$y ts=$(date '+%H:%M:%S')" | tee -a "$UI_LOG"
      adb shell input tap "$x" "$y"
      return 0
    fi
  done
  return 1
}

wait_for_log() {
  local pattern="$1"
  local timeout_sec="$2"
  local deadline=$((SECONDS + timeout_sec))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if adb logcat -d | rg -q "$pattern"; then
      return 0
    fi
    sleep 1
  done
  return 1
}

benchmark_decode_tokens() {
  local xml="$1"
  python3 - "$xml" "$PACKAGE_NAME" <<'PY'
import re
import sys
import xml.etree.ElementTree as ET

xml_path = sys.argv[1]
package_name = sys.argv[2]
resource_ids = {
    f"{package_name}:id/tv_chat_benchmark",
    "com.alibaba.mnnllm.android:id/tv_chat_benchmark",
    "com.alibaba.mnnllm.android.release:id/tv_chat_benchmark",
}
root = ET.parse(xml_path).getroot()
for node in root.iter("node"):
    rid = node.attrib.get("resource-id")
    text = node.attrib.get("text") or ""
    if rid in resource_ids and "Decode:" in text:
        match = re.search(r"Decode:\s*[^,]+,\s*(\d+)\s+tokens", text)
        if match:
            print(match.group(1))
            sys.exit(0)
print("")
PY
}

ensure_device_unlocked
adb logcat -c
adb shell am force-stop "$PACKAGE_NAME"
sleep 1

echo "LAUNCH_CHAT model_id=$MODEL_ID ts=$(date '+%H:%M:%S')" | tee -a "$UI_LOG"
adb shell am start -W \
  -n "$PACKAGE_NAME/.chat.ChatActivity" \
  --es modelId "$MODEL_ID" \
  --es modelName "$MODEL_NAME" \
  --es configFilePath "$CONFIG_PATH" >/dev/null

if ! wait_for_log "ChatPresenter: chatSession loaded" 90; then
  adb logcat -d >"$LOGCAT_FILE" || true
  echo "CHAT_LOAD_TIMEOUT" >&2
  exit 1
fi

dump_ui "$TMP_XML"
shot "01_chat_loaded"

tap_by_any_rid "$TMP_XML" \
  "$PACKAGE_NAME:id/et_message" \
  "com.alibaba.mnnllm.android:id/et_message" \
  "com.alibaba.mnnllm.android.release:id/et_message"
adb shell input text "$PROMPT_TEXT"
sleep 1
dump_ui "$TMP_XML"
shot "02_after_input"

tap_by_any_rid "$TMP_XML" \
  "$PACKAGE_NAME:id/btn_send" \
  "com.alibaba.mnnllm.android:id/btn_send" \
  "com.alibaba.mnnllm.android.release:id/btn_send"

decode_tokens=""
deadline=$((SECONDS + 120))
while [ "$SECONDS" -lt "$deadline" ]; do
  sleep 2
  dump_ui "$TMP_XML" || true
  decode_tokens="$(benchmark_decode_tokens "$TMP_XML" | tr -d '\r')"
  if [ -n "$decode_tokens" ]; then
    break
  fi
done

adb logcat -d >"$LOGCAT_FILE" || true
shot "03_after_generation"

{
  echo "MODEL_ID=$MODEL_ID"
  echo "PROMPT_TEXT=$PROMPT_TEXT"
  echo "DECODE_TOKENS=${decode_tokens:-missing}"
  echo "LOGCAT_FILE=$LOGCAT_FILE"
} >"$SUMMARY_FILE"

cat "$SUMMARY_FILE"

if [ -z "${decode_tokens:-}" ]; then
  echo "Failed to capture benchmark decode token count from UI." >&2
  exit 1
fi

if [ "$decode_tokens" -le 1 ]; then
  echo "Expected decode token count > 1, got $decode_tokens" >&2
  exit 1
fi
