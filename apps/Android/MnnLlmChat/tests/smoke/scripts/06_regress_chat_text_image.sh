#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACT_DIR="${ARTIFACT_DIR:-$SMOKE_DIR/artifacts}"
mkdir -p "$ARTIFACT_DIR/chat_io"

QUERY="$SCRIPT_DIR/ui_xml_query.py"
PACKAGE_NAME="${PACKAGE_NAME:-com.alibaba.mnnllm.android}"
TMP_XML="/tmp/mnn_chat_ui.xml"
SHOT_DIR="$ARTIFACT_DIR/chat_io/shots"
mkdir -p "$SHOT_DIR"
UI_LOG="$ARTIFACT_DIR/chat_io/ui_actions.log"
rm -f "$UI_LOG"

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

tap_by_any_contains_text() {
  local xml="$1"
  shift
  local text
  for text in "$@"; do
    if python3 "$QUERY" --xml "$xml" --contains-text "$text" >/tmp/mnn_query_hit.txt 2>/dev/null; then
      read -r x y _ </tmp/mnn_query_hit.txt
      echo "TAP contains=$text x=$x y=$y ts=$(date '+%H:%M:%S')" | tee -a "$UI_LOG"
      adb shell input tap "$x" "$y"
      return 0
    fi
  done
  return 1
}

dump_ui() {
  local out="$1"
  local try
  for try in 1 2 3; do
    adb shell uiautomator dump /sdcard/mnn_chat_ui.xml >/dev/null 2>/dev/null || true
    adb pull /sdcard/mnn_chat_ui.xml "$out" >/dev/null 2>/dev/null || true
    if [ -s "$out" ] && rg -q "<hierarchy" "$out"; then
      if rg -q "package=\"com\\.lbe\\.security\\.miui\"" "$out"; then
        echo "MIUI_PERMISSION_OVERLAY try=$try ts=$(date '+%H:%M:%S')" | tee -a "$UI_LOG"
        tap_by_any_contains_text "$out" "始终允许" "仅在使用中允许" "允许" "同意" || true
        sleep 1
        continue
      fi
      if rg -q "package=\"com\\.alibaba\\.mnnllm\\.android(\\.release)?\"" "$out"; then
        return 0
      fi
    fi
    echo "UI_DUMP_RETRY try=$try ts=$(date '+%H:%M:%S')" | tee -a "$UI_LOG"
    ensure_device_unlocked
    adb shell monkey -p "$PACKAGE_NAME" -c android.intent.category.LAUNCHER 1 >/dev/null 2>&1 || true
    sleep 1
  done
  return 1
}

tap_by_rid() {
  local xml="$1"
  local rid="$2"
  read -r x y _ < <(python3 "$QUERY" --xml "$xml" --resource-id "$rid")
  adb shell input tap "$x" "$y"
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
  echo "TAP_NOT_FOUND candidates=$*" | tee -a "$UI_LOG"
  return 1
}

tap_by_rid_text() {
  local xml="$1"
  local text="$2"
  shift 2
  local rid
  for rid in "$@"; do
    if python3 "$QUERY" --xml "$xml" --resource-id "$rid" --text "$text" >/tmp/mnn_query_hit.txt 2>/dev/null; then
      read -r x y _ </tmp/mnn_query_hit.txt
      echo "TAP rid=$rid text=$text x=$x y=$y ts=$(date '+%H:%M:%S')" | tee -a "$UI_LOG"
      adb shell input tap "$x" "$y"
      return 0
    fi
  done
  echo "TAP_TEXT_NOT_FOUND text=$text candidates=$*" | tee -a "$UI_LOG"
  return 1
}

tap_by_rid_contains_text() {
  local xml="$1"
  local text="$2"
  shift 2
  local rid
  for rid in "$@"; do
    if python3 "$QUERY" --xml "$xml" --resource-id "$rid" --contains-text "$text" >/tmp/mnn_query_hit.txt 2>/dev/null; then
      read -r x y _ </tmp/mnn_query_hit.txt
      echo "TAP rid=$rid contains=$text x=$x y=$y ts=$(date '+%H:%M:%S')" | tee -a "$UI_LOG"
      adb shell input tap "$x" "$y"
      return 0
    fi
  done
  echo "TAP_CONTAINS_NOT_FOUND text=$text candidates=$*" | tee -a "$UI_LOG"
  return 1
}

exists_any_rid() {
  local xml="$1"
  shift
  local rid
  for rid in "$@"; do
    if python3 "$QUERY" --xml "$xml" --resource-id "$rid" >/dev/null 2>&1; then
      return 0
    fi
  done
  return 1
}

wait_for_chat_ready() {
  local attempt
  for attempt in 1 2 3 4 5 6; do
    sleep 2
    dump_ui "$TMP_XML" || true
    if exists_any_rid "$TMP_XML" \
      "com.alibaba.mnnllm.android:id/et_message" \
      "com.alibaba.mnnllm.android.release:id/et_message"; then
      echo "CHAT_READY attempt=$attempt ts=$(date '+%H:%M:%S')" | tee -a "$UI_LOG"
      return 0
    fi
    echo "CHAT_READY_RETRY attempt=$attempt ts=$(date '+%H:%M:%S')" | tee -a "$UI_LOG"
  done
  return 1
}

# Ensure app on foreground and open model market.
ensure_device_unlocked
adb shell am force-stop "$PACKAGE_NAME"
sleep 1
adb shell monkey -p "$PACKAGE_NAME" -c android.intent.category.LAUNCHER 1 >/dev/null
sleep 2
dump_ui "$TMP_XML"
shot "01_after_launch"

if tap_by_any_rid "$TMP_XML" \
  "com.alibaba.mnnllm.android:id/tab_model_market" \
  "com.alibaba.mnnllm.android.release:id/tab_model_market"; then
  sleep 2
  dump_ui "$TMP_XML"
elif exists_any_rid "$TMP_XML" \
  "com.alibaba.mnnllm.android:id/et_message" \
  "com.alibaba.mnnllm.android.release:id/et_message"; then
  echo "ENTRY_ALREADY_ON_CHAT ts=$(date '+%H:%M:%S')" | tee -a "$UI_LOG"
else
  echo "ENTRY_MARKET_TAB_NOT_FOUND ts=$(date '+%H:%M:%S')" | tee -a "$UI_LOG"
fi
shot "02_market_tab"

# Tap chat-entry button by text first to avoid hitting unrelated "下载" buttons.
if tap_by_rid_text "$TMP_XML" "对话" \
  "com.alibaba.mnnllm.android:id/btn_download_action" \
  "com.alibaba.mnnllm.android.release:id/btn_download_action" \
  || tap_by_rid_contains_text "$TMP_XML" "对话" \
  "com.alibaba.mnnllm.android:id/btn_download_action" \
  "com.alibaba.mnnllm.android.release:id/btn_download_action"; then
  wait_for_chat_ready || true
elif exists_any_rid "$TMP_XML" \
  "com.alibaba.mnnllm.android:id/et_message" \
  "com.alibaba.mnnllm.android.release:id/et_message"; then
  :
elif tap_by_any_rid "$TMP_XML" \
  "com.alibaba.mnnllm.android:id/tab_chat" \
  "com.alibaba.mnnllm.android.release:id/tab_chat"; then
  wait_for_chat_ready || true
else
  echo "CHAT_ENTRY_NOT_FOUND (no 对话 button, no chat input, no chat tab)" >&2
  exit 1
fi
cp "$TMP_XML" "$ARTIFACT_DIR/chat_io/chat_entered.xml"
shot "03_chat_entered"

# Text input + send.
tap_by_any_rid "$TMP_XML" \
  "com.alibaba.mnnllm.android:id/et_message" \
  "com.alibaba.mnnllm.android.release:id/et_message"
adb shell input text "smoke_text_case_qwen35"
sleep 1
dump_ui "$TMP_XML"
cp "$TMP_XML" "$ARTIFACT_DIR/chat_io/after_text_input.xml"
shot "04_after_text_input"
tap_by_any_rid "$TMP_XML" \
  "com.alibaba.mnnllm.android:id/btn_send" \
  "com.alibaba.mnnllm.android.release:id/btn_send"
sleep 2
dump_ui "$TMP_XML"
cp "$TMP_XML" "$ARTIFACT_DIR/chat_io/after_send.xml"
shot "05_after_send"

# Image input entry check: open plus panel and capture UI state.
tap_by_any_rid "$TMP_XML" \
  "com.alibaba.mnnllm.android:id/bt_plus" \
  "com.alibaba.mnnllm.android.release:id/bt_plus"
sleep 2
dump_ui "$TMP_XML"
cp "$TMP_XML" "$ARTIFACT_DIR/chat_io/after_plus.xml"
shot "06_after_plus_panel"

# Image testing entry evidence: hide keyboard then re-open plus entry and capture one more shot.
adb shell input keyevent 4 >/dev/null 2>&1 || true
sleep 1
dump_ui "$TMP_XML"
tap_by_any_rid "$TMP_XML" \
  "com.alibaba.mnnllm.android:id/bt_plus" \
  "com.alibaba.mnnllm.android.release:id/bt_plus" || true
sleep 2
dump_ui "$TMP_XML"
cp "$TMP_XML" "$ARTIFACT_DIR/chat_io/image_test_entry.xml"
shot "07_image_test_entry"

# Mark PASS if core controls exist in dumped UI.
python3 "$QUERY" --xml "$ARTIFACT_DIR/chat_io/chat_entered.xml" --resource-id "com.alibaba.mnnllm.android:id/et_message" >/dev/null
python3 "$QUERY" --xml "$ARTIFACT_DIR/chat_io/chat_entered.xml" --resource-id "com.alibaba.mnnllm.android:id/btn_send" >/dev/null
python3 "$QUERY" --xml "$ARTIFACT_DIR/chat_io/chat_entered.xml" --resource-id "com.alibaba.mnnllm.android:id/bt_plus" >/dev/null

{
  echo "CHAT_TEXT_IMAGE_ENTRY=PASS"
  echo "TEXT_INPUT_SENT=smoke_text_case_qwen35"
  echo "NOTE=Image flow validated to attachment-entry level via plus button and UI capture."
  echo "SCREENSHOTS=$SHOT_DIR"
  echo "UI_ACTION_LOG=$UI_LOG"
} >"$ARTIFACT_DIR/chat_io/summary.txt"
