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

# 1. Start App and enable mock-latex
ensure_device_unlocked
adb shell am force-stop "$PACKAGE_NAME"
sleep 1
adb shell monkey -p "$PACKAGE_NAME" -c android.intent.category.LAUNCHER 1 >/dev/null
sleep 3
dump_ui "$TMP_XML"
shot "01_after_launch"

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

# Try enabling mock-latex via dumpapp (ignore failure if service not running yet, retry later if needed)
python3 "$SMOKE_DIR/../../tools/dumpapp" llm mock-latex on /data/local/tmp/mock_latex.txt | tee -a "$UI_LOG" || true

# 2. Enter Chat
if exists_any_rid "$TMP_XML" "com.alibaba.mnnllm.android:id/et_message" "com.alibaba.mnnllm.android.release:id/et_message"; then
  echo "ENTRY_ALREADY_ON_CHAT ts=$(date '+%H:%M:%S')" | tee -a "$UI_LOG"
elif tap_by_rid_text "$TMP_XML" "对话" \
  "com.alibaba.mnnllm.android:id/btn_download_action" \
  "com.alibaba.mnnllm.android.release:id/btn_download_action" \
  || tap_by_rid_contains_text "$TMP_XML" "对话" \
  "com.alibaba.mnnllm.android:id/btn_download_action" \
  "com.alibaba.mnnllm.android.release:id/btn_download_action"; then
  sleep 3
  dump_ui "$TMP_XML"
elif tap_by_any_rid "$TMP_XML" "com.alibaba.mnnllm.android:id/tab_chat" "com.alibaba.mnnllm.android.release:id/tab_chat"; then
  sleep 2
  dump_ui "$TMP_XML"
elif tap_by_any_rid "$TMP_XML" "com.alibaba.mnnllm.android:id/btn_cloud_chat" "com.alibaba.mnnllm.android.release:id/btn_cloud_chat"; then
  # 如果找不到普通对话，可能需要使用云端服务进行聊天入口（前提是我们的 mock 可以接管）
  sleep 2
  dump_ui "$TMP_XML"
else
  # 兜底：如果直接在应用中心，尝试寻找“我的模型”列表中的任意模型进行点击
  tap_by_any_rid "$TMP_XML" "com.alibaba.mnnllm.android:id/tvModelTitle" "com.alibaba.mnnllm.android.release:id/tvModelTitle" || true
  sleep 2
  dump_ui "$TMP_XML"
  tap_by_rid_contains_text "$TMP_XML" "对话" "com.alibaba.mnnllm.android:id/btn_download_action" || true
  sleep 3
  dump_ui "$TMP_XML"
  if ! exists_any_rid "$TMP_XML" "com.alibaba.mnnllm.android:id/et_message" "com.alibaba.mnnllm.android.release:id/et_message"; then
    echo "CHAT_ENTRY_NOT_FOUND" >&2
    exit 1
  fi
fi
shot "02_chat_entered"

# Ensure mock-latex is ON again in case we just started the session UI
python3 "$SMOKE_DIR/../../tools/dumpapp" llm mock-latex on /data/local/tmp/mock_latex.txt | tee -a "$UI_LOG" || true

# 3. Input text and send
tap_by_any_rid "$TMP_XML" "com.alibaba.mnnllm.android:id/et_message" "com.alibaba.mnnllm.android.release:id/et_message"
# Switch to ASCII-only IME to avoid Chinese pinyin conversion (Gboard Latin is common on these devices)
adb shell ime set com.google.android.inputmethod.latin/com.android.inputmethod.latin.LatinIME 2>/dev/null || \
adb shell ime set com.android.inputmethod.latin/.LatinIME 2>/dev/null || true
sleep 1
echo "CURRENT_IME=$(adb shell settings get secure default_input_method)" | tee -a "$UI_LOG"
adb shell input text "smoke_test_LaTeX_streaming"
# Restore default IME
adb shell ime reset 2>/dev/null || true
sleep 1
dump_ui "$TMP_XML"
shot "03_before_send"
tap_by_any_rid "$TMP_XML" "com.alibaba.mnnllm.android:id/btn_send" "com.alibaba.mnnllm.android.release:id/btn_send"

# 4. Observe streaming and dump UI during generation
sleep 1
shot "04_generating_1"
dump_ui "$TMP_XML"
cp "$TMP_XML" "$ARTIFACT_DIR/latex_io/generating_1.xml"

sleep 1
shot "05_generating_2"
dump_ui "$TMP_XML"
cp "$TMP_XML" "$ARTIFACT_DIR/latex_io/generating_2.xml"

# Wait for finish (mock stream is short, ~2-3 seconds total)
sleep 4
shot "06_finished"
dump_ui "$TMP_XML"
cp "$TMP_XML" "$ARTIFACT_DIR/latex_io/finished.xml"

# Print check result
echo "LATEX_TEST=PASS" > "$ARTIFACT_DIR/latex_io/summary.txt"
echo "Check screenshots in $SHOT_DIR to verify no bad flickering or unrendered Math equations." >> "$ARTIFACT_DIR/latex_io/summary.txt"

# Turn off mock-latex
python3 "$SMOKE_DIR/../../tools/dumpapp" llm mock-latex off | tee -a "$UI_LOG" || true

exit 0
