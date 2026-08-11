#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACT_DIR="${ARTIFACT_DIR:-$SMOKE_DIR/artifacts}"
mkdir -p "$ARTIFACT_DIR/latex_io"

QUERY="$SCRIPT_DIR/ui_xml_query.py"
PACKAGE_NAME="${PACKAGE_NAME:-com.alibaba.mnnllm.android}"
TMP_XML="/tmp/mnn_chat_ui_latex.xml"
SHOT_DIR="$ARTIFACT_DIR/latex_io/shots"
mkdir -p "$SHOT_DIR"
UI_LOG="$ARTIFACT_DIR/latex_io/ui_actions.log"
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
    adb shell uiautomator dump /sdcard/mnn_chat_ui_latex.xml >/dev/null 2>/dev/null || true
    adb pull /sdcard/mnn_chat_ui_latex.xml "$out" >/dev/null 2>/dev/null || true
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

# 1. Prepare mock latex content
ensure_device_unlocked
adb shell am force-stop "$PACKAGE_NAME"
sleep 1

# Push mock latex content with complex formulas
MOCK_CONTENT_FILE="$ARTIFACT_DIR/mock_latex.txt"
cat << 'EOF' > "$MOCK_CONTENT_FILE"
这里有一个复杂的矩阵：
$$
\begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6 \\
7 & 8 & 9
\end{bmatrix}
$$
还有一个复杂的行内公式 $\int_a^b f(x) dx = F(b) - F(a)$。
然后是一些带求和的块级公式：
$$
\sum_{i=1}^{n} i = \frac{n(n+1)}{2}
$$
测试截断时是否有大规模的高度闪耀或者文字跳动。
EOF
adb push "$MOCK_CONTENT_FILE" /data/local/tmp/mock_latex.txt

# 2. Launch mock chat directly so this case only validates latex rendering
echo "LAUNCH_MOCK_CHAT ts=$(date '+%H:%M:%S')" | tee -a "$UI_LOG"
adb shell am start -W \
  -n "$PACKAGE_NAME/.chat.ChatActivity" \
  --es modelName "Qwen" \
  --es modelId "mock" \
  --ez mock_stream_enable true \
  --es mock_stream_content_file "/data/local/tmp/mock_latex.txt" \
  --el mock_stream_interval_ms 20 >/dev/null
sleep 1
dump_ui "$TMP_XML"
shot "01_after_launch"
if ! exists_any_rid "$TMP_XML" "com.alibaba.mnnllm.android:id/recyclerView" "com.alibaba.mnnllm.android.release:id/recyclerView"; then
  echo "MOCK_CHAT_NOT_FOUND" >&2
  exit 1
fi
shot "02_chat_entered"

# 3. Observe streaming and dump UI during generation
sleep 1
shot "03_generating_1"
dump_ui "$TMP_XML"
cp "$TMP_XML" "$ARTIFACT_DIR/latex_io/generating_1.xml"

sleep 1
shot "04_generating_2"
dump_ui "$TMP_XML"
cp "$TMP_XML" "$ARTIFACT_DIR/latex_io/generating_2.xml"

# Wait for finish
sleep 4
shot "05_finished"
dump_ui "$TMP_XML"
cp "$TMP_XML" "$ARTIFACT_DIR/latex_io/finished.xml"

# Print check result
echo "LATEX_TEST=PASS" > "$ARTIFACT_DIR/latex_io/summary.txt"
echo "Check screenshots in $SHOT_DIR to verify no bad flickering or unrendered Math equations." >> "$ARTIFACT_DIR/latex_io/summary.txt"

exit 0
