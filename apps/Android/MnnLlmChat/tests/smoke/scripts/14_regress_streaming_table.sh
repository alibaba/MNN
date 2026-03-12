#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACT_DIR="${ARTIFACT_DIR:-$SMOKE_DIR/artifacts}"
mkdir -p "$ARTIFACT_DIR/table_io"

QUERY="$SCRIPT_DIR/ui_xml_query.py"
PACKAGE_NAME="${PACKAGE_NAME:-com.alibaba.mnnllm.android}"
TMP_XML="/tmp/mnn_chat_ui_table.xml"
SHOT_DIR="$ARTIFACT_DIR/table_io/shots"
mkdir -p "$SHOT_DIR"
UI_LOG="$ARTIFACT_DIR/table_io/ui_actions.log"
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

dump_ui() {
  local out="$1"
  local try
  for try in 1 2 3; do
    adb shell uiautomator dump /sdcard/mnn_chat_ui_table.xml >/dev/null 2>/dev/null || true
    adb pull /sdcard/mnn_chat_ui_table.xml "$out" >/dev/null 2>/dev/null || true
    if [ -s "$out" ] && rg -q "<hierarchy" "$out"; then
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

ensure_device_unlocked
adb shell am force-stop "$PACKAGE_NAME"
sleep 1
adb shell monkey -p "$PACKAGE_NAME" -c android.intent.category.LAUNCHER 1 >/dev/null
sleep 3
dump_ui "$TMP_XML"
shot "01_after_launch"

MOCK_CONTENT_FILE="$ARTIFACT_DIR/mock_table.md"
cat << 'EOF' >"$MOCK_CONTENT_FILE"
下面是一个用于验证 Markdown 表格渲染的流式输出：

| 模型 | 参数量 | 备注 |
| --- | --- | --- |
| Qwen2.5-1.5B | 1.5B | 基础对话模型 |
| Qwen3-4B | 4B | 这一列包含更长的描述，用来观察窄屏下表格换行和列宽压缩是否稳定 |
| DeepSeek-R1-7B | 7B | 推理模型 |

表格后面这段普通文字应该继续正常显示，不能因为表格渲染而消失或错位。
EOF
adb push "$MOCK_CONTENT_FILE" /data/local/tmp/mock_table.md

# Reuse the existing mock streaming hook. The command name is historical,
# but it can load arbitrary markdown content from a file.
python3 "$SMOKE_DIR/../../tools/dumpapp" llm mock-latex on /data/local/tmp/mock_table.md | tee -a "$UI_LOG" || true

if exists_any_rid "$TMP_XML" "com.alibaba.mnnllm.android:id/et_message" "com.alibaba.mnnllm.android.release:id/et_message"; then
  echo "ENTRY_ALREADY_ON_CHAT ts=$(date '+%H:%M:%S')" | tee -a "$UI_LOG"
elif exists_any_rid "$TMP_XML" \
  "com.alibaba.mnnllm.android:id/tab_local_models" \
  "com.alibaba.mnnllm.android.release:id/tab_local_models" \
  && exists_any_rid "$TMP_XML" \
  "com.alibaba.mnnllm.android:id/recycler_item_model_parent" \
  "com.alibaba.mnnllm.android.release:id/recycler_item_model_parent"; then
  tap_by_any_rid "$TMP_XML" \
    "com.alibaba.mnnllm.android:id/recycler_item_model_parent" \
    "com.alibaba.mnnllm.android.release:id/recycler_item_model_parent"
  sleep 3
  dump_ui "$TMP_XML"
elif tap_by_rid_text "$TMP_XML" "对话" \
  "com.alibaba.mnnllm.android:id/btn_download_action" \
  "com.alibaba.mnnllm.android.release:id/btn_download_action" \
  || tap_by_rid_contains_text "$TMP_XML" "对话" \
  "com.alibaba.mnnllm.android:id/btn_download_action" \
  "com.alibaba.mnnllm.android.release:id/btn_download_action"; then
  sleep 3
  dump_ui "$TMP_XML"
elif tap_by_any_rid "$TMP_XML" \
  "com.alibaba.mnnllm.android:id/tab_model_market" \
  "com.alibaba.mnnllm.android.release:id/tab_model_market"; then
  sleep 2
  dump_ui "$TMP_XML"
  if tap_by_rid_text "$TMP_XML" "对话" \
    "com.alibaba.mnnllm.android:id/btn_download_action" \
    "com.alibaba.mnnllm.android.release:id/btn_download_action" \
    || tap_by_rid_contains_text "$TMP_XML" "对话" \
    "com.alibaba.mnnllm.android:id/btn_download_action" \
    "com.alibaba.mnnllm.android.release:id/btn_download_action"; then
    sleep 3
    dump_ui "$TMP_XML"
  fi
elif tap_by_any_rid "$TMP_XML" "com.alibaba.mnnllm.android:id/tab_chat" "com.alibaba.mnnllm.android.release:id/tab_chat"; then
  sleep 2
  dump_ui "$TMP_XML"
elif tap_by_any_rid "$TMP_XML" "com.alibaba.mnnllm.android:id/btn_cloud_chat" "com.alibaba.mnnllm.android.release:id/btn_cloud_chat"; then
  sleep 2
  dump_ui "$TMP_XML"
else
  tap_by_any_rid "$TMP_XML" "com.alibaba.mnnllm.android:id/tvModelTitle" "com.alibaba.mnnllm.android.release:id/tvModelTitle" || true
  sleep 2
  dump_ui "$TMP_XML"
  tap_by_rid_contains_text "$TMP_XML" "对话" \
    "com.alibaba.mnnllm.android:id/btn_download_action" \
    "com.alibaba.mnnllm.android.release:id/btn_download_action" || true
  sleep 3
  dump_ui "$TMP_XML"
  if ! exists_any_rid "$TMP_XML" "com.alibaba.mnnllm.android:id/et_message" "com.alibaba.mnnllm.android.release:id/et_message"; then
    echo "CHAT_ENTRY_NOT_FOUND" >&2
    exit 1
  fi
fi

if ! exists_any_rid "$TMP_XML" "com.alibaba.mnnllm.android:id/et_message" "com.alibaba.mnnllm.android.release:id/et_message"; then
  echo "CHAT_ENTRY_NOT_FOUND" >&2
  exit 1
fi
shot "02_chat_entered"

python3 "$SMOKE_DIR/../../tools/dumpapp" llm mock-latex on /data/local/tmp/mock_table.md | tee -a "$UI_LOG" || true

tap_by_any_rid "$TMP_XML" "com.alibaba.mnnllm.android:id/et_message" "com.alibaba.mnnllm.android.release:id/et_message"
adb shell ime set com.google.android.inputmethod.latin/com.android.inputmethod.latin.LatinIME 2>/dev/null || \
adb shell ime set com.android.inputmethod.latin/.LatinIME 2>/dev/null || true
sleep 1
echo "CURRENT_IME=$(adb shell settings get secure default_input_method)" | tee -a "$UI_LOG"
adb shell input text "smoke_test_table_streaming"
adb shell ime reset 2>/dev/null || true
sleep 1
dump_ui "$TMP_XML"
shot "03_before_send"
tap_by_any_rid "$TMP_XML" "com.alibaba.mnnllm.android:id/btn_send" "com.alibaba.mnnllm.android.release:id/btn_send"

sleep 1
shot "04_generating_1"
dump_ui "$TMP_XML"
cp "$TMP_XML" "$ARTIFACT_DIR/table_io/generating_1.xml"

sleep 1
shot "05_generating_2"
dump_ui "$TMP_XML"
cp "$TMP_XML" "$ARTIFACT_DIR/table_io/generating_2.xml"

sleep 4
shot "06_finished"
dump_ui "$TMP_XML"
cp "$TMP_XML" "$ARTIFACT_DIR/table_io/finished.xml"

{
  echo "TABLE_RENDER_TEST=PASS"
  echo "SCREENSHOTS=$SHOT_DIR"
  echo "UI_ACTION_LOG=$UI_LOG"
  echo "NOTE=Check screenshots to verify the markdown table renders cleanly without severe flicker or obvious raw-markdown fallback."
} >"$ARTIFACT_DIR/table_io/summary.txt"

python3 "$SMOKE_DIR/../../tools/dumpapp" llm mock-latex off | tee -a "$UI_LOG" || true

exit 0
