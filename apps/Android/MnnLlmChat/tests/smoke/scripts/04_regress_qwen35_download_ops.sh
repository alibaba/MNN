#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACT_DIR="${ARTIFACT_DIR:-$SMOKE_DIR/artifacts}"
mkdir -p "$ARTIFACT_DIR/qwen35_download"

MODEL_ID="${QWEN35_MODEL_ID:-ModelScope/MNN/Qwen3.5-0.8B-MNN}"
DUMPAPP="${DUMPAPP:-$SMOKE_DIR/../../tools/dumpapp}"
LOG="$ARTIFACT_DIR/qwen35_download/download_ops.log"
SHOT_DIR="$ARTIFACT_DIR/qwen35_download/shots"
mkdir -p "$SHOT_DIR"
QUERY="$SCRIPT_DIR/ui_xml_query.py"
TMP_XML="/tmp/mnn_download_ui.xml"

shot() {
  local name="$1"
  adb exec-out screencap -p >"$SHOT_DIR/${name}.png"
}

dump_ui() {
  local out="$1"
  adb shell uiautomator dump /sdcard/mnn_download_ui.xml >/dev/null 2>/dev/null || true
  if adb shell ls /sdcard/mnn_download_ui.xml >/dev/null 2>&1; then
    adb pull /sdcard/mnn_download_ui.xml "$out" >/dev/null 2>/dev/null || true
  fi
}

tap_by_any_rid() {
  local xml="$1"
  shift
  local rid
  for rid in "$@"; do
    if python3 "$QUERY" --xml "$xml" --resource-id "$rid" >/tmp/mnn_download_hit.txt 2>/dev/null; then
      read -r x y _ </tmp/mnn_download_hit.txt
      adb shell input tap "$x" "$y"
      return 0
    fi
  done
  return 1
}

echo "[INFO] MODEL_ID=$MODEL_ID" | tee "$LOG"

"$DUMPAPP" models refresh | tee -a "$LOG"
adb shell monkey -p com.alibaba.mnnllm.android -c android.intent.category.LAUNCHER 1 >/dev/null
sleep 2
dump_ui "$TMP_XML"
tap_by_any_rid "$TMP_XML" "com.alibaba.mnnllm.android:id/tab_model_market" "com.alibaba.mnnllm.android.release:id/tab_model_market" || true
sleep 2
dump_ui "$TMP_XML"
shot "01_after_models_refresh"
"$DUMPAPP" download delete "$MODEL_ID" | tee -a "$LOG"
"$DUMPAPP" download status "$MODEL_ID" | tee "$ARTIFACT_DIR/qwen35_download/status_after_delete.txt"
shot "02_after_delete"

# Start download in background, pause after a short delay.
("$DUMPAPP" download test "$MODEL_ID" >"$ARTIFACT_DIR/qwen35_download/download_test.log" 2>&1 &) 
sleep 8
"$DUMPAPP" download pause "$MODEL_ID" | tee -a "$LOG"
sleep 2
"$DUMPAPP" download status "$MODEL_ID" | tee "$ARTIFACT_DIR/qwen35_download/status_after_pause.txt"
dump_ui "$TMP_XML"
shot "03_after_pause"
if rg -q "没有可用的模型" "$TMP_XML"; then
  echo "DOWNLOAD_UI_EMPTY_STATE_DETECTED" | tee -a "$LOG"
  exit 1
fi

# Resume download request (do not wait full completion in smoke).
("$DUMPAPP" download test "$MODEL_ID" >"$ARTIFACT_DIR/qwen35_download/download_resume_test.log" 2>&1 &) 
sleep 8
"$DUMPAPP" download status "$MODEL_ID" | tee "$ARTIFACT_DIR/qwen35_download/status_after_resume_probe.txt"
"$DUMPAPP" download pause "$MODEL_ID" | tee -a "$LOG"
shot "04_after_resume_probe"

{
  echo "QWEN35_DOWNLOAD_OPS=PASS"
  echo "SCREENSHOTS=$SHOT_DIR"
} | tee "$ARTIFACT_DIR/qwen35_download/summary.txt"
