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

# Launch mock chat directly so this case only validates markdown-table rendering
echo "LAUNCH_MOCK_CHAT ts=$(date '+%H:%M:%S')" | tee -a "$UI_LOG"
adb shell am start -W \
  -n "$PACKAGE_NAME/.chat.ChatActivity" \
  --es modelName "Qwen" \
  --es modelId "mock" \
  --ez mock_stream_enable true \
  --es mock_stream_content_file "/data/local/tmp/mock_table.md" \
  --el mock_stream_interval_ms 20 >/dev/null
sleep 1
dump_ui "$TMP_XML"
shot "01_after_launch"
if ! exists_any_rid "$TMP_XML" "com.alibaba.mnnllm.android:id/recyclerView" "com.alibaba.mnnllm.android.release:id/recyclerView"; then
  echo "MOCK_CHAT_NOT_FOUND" >&2
  exit 1
fi
shot "02_chat_entered"

sleep 1
shot "03_generating_1"
dump_ui "$TMP_XML"
cp "$TMP_XML" "$ARTIFACT_DIR/table_io/generating_1.xml"

sleep 1
shot "04_generating_2"
dump_ui "$TMP_XML"
cp "$TMP_XML" "$ARTIFACT_DIR/table_io/generating_2.xml"

sleep 4
shot "05_finished"
dump_ui "$TMP_XML"
cp "$TMP_XML" "$ARTIFACT_DIR/table_io/finished.xml"

{
  echo "TABLE_RENDER_TEST=PASS"
  echo "SCREENSHOTS=$SHOT_DIR"
  echo "UI_ACTION_LOG=$UI_LOG"
  echo "NOTE=Check screenshots to verify the markdown table renders cleanly without severe flicker or obvious raw-markdown fallback."
} >"$ARTIFACT_DIR/table_io/summary.txt"

exit 0
