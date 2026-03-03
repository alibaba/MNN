#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACT_DIR="${ARTIFACT_DIR:-$SMOKE_DIR/artifacts}"
mkdir -p "$ARTIFACT_DIR/qwen35_benchmark"

PACKAGE_NAME="${PACKAGE_NAME:-com.alibaba.mnnllm.android}"
MODEL_ID="${QWEN35_BENCH_MODEL_ID:-ModelScope/MNN/Qwen3.5-0.8B-MNN}"
DUMPAPP="${DUMPAPP:-$SMOKE_DIR/../../tools/dumpapp}"
SHOT_DIR="$ARTIFACT_DIR/qwen35_benchmark/shots"
mkdir -p "$SHOT_DIR"
QUERY="$SCRIPT_DIR/ui_xml_query.py"
TMP_XML="/tmp/mnn_benchmark_ui.xml"
CASE_TIMEOUT_SEC="${CASE_TIMEOUT_SEC:-220}"
UI_CASE_TIMEOUT_SEC="${UI_CASE_TIMEOUT_SEC:-300}"
UI_LOG="$ARTIFACT_DIR/qwen35_benchmark/ui_actions.log"
SUMMARY_FILE="$ARTIFACT_DIR/qwen35_benchmark/summary.txt"
rm -f "$UI_LOG"
ACTIVE_MODEL_ID="$MODEL_ID"

shot() {
  local name="$1"
  adb exec-out screencap -p >"$SHOT_DIR/${name}.png"
}

dump_ui() {
  local out="$1"
  local try
  for try in 1 2 3; do
    if adb shell uiautomator dump /sdcard/mnn_benchmark_ui.xml >/dev/null 2>/dev/null \
      && adb pull /sdcard/mnn_benchmark_ui.xml "$out" >/dev/null 2>/dev/null \
      && [ -s "$out" ] \
      && rg -q "<hierarchy" "$out"; then
      return 0
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
    if python3 "$QUERY" --xml "$xml" --resource-id "$rid" >/tmp/mnn_bench_query_hit.txt 2>/dev/null; then
      read -r x y _ </tmp/mnn_bench_query_hit.txt
      echo "TAP rid=$rid x=$x y=$y ts=$(date '+%H:%M:%S')" | tee -a "$UI_LOG"
      adb shell input tap "$x" "$y"
      return 0
    fi
  done
  echo "TAP_NOT_FOUND candidates=$*" | tee -a "$UI_LOG"
  return 1
}

ensure_benchmark_tab() {
  local try
  for try in 1 2 3 4 5; do
    dump_ui "$TMP_XML"
    if tap_by_any_rid "$TMP_XML" \
      "com.alibaba.mnnllm.android:id/tab_benchmark" \
      "com.alibaba.mnnllm.android.release:id/tab_benchmark"; then
      sleep 2
      return 0
    fi
    adb shell input keyevent 4 >/dev/null 2>&1 || true
    sleep 1
  done
  return 1
}

relaunch_and_enter_benchmark() {
  adb shell monkey -p "$PACKAGE_NAME" -c android.intent.category.LAUNCHER 1 >/dev/null
  sleep 2
  ensure_benchmark_tab
}

ensure_backend_selector_visible() {
  local try
  for try in 1 2 3 4; do
    dump_ui "$TMP_XML" || true
    if python3 "$QUERY" --xml "$TMP_XML" --resource-id "com.alibaba.mnnllm.android:id/backend_selector_group" >/dev/null 2>&1 \
      || python3 "$QUERY" --xml "$TMP_XML" --resource-id "com.alibaba.mnnllm.android.release:id/backend_selector_group" >/dev/null 2>&1; then
      return 0
    fi
    echo "SWIPE_TO_TOP try=$try ts=$(date '+%H:%M:%S')" | tee -a "$UI_LOG"
    adb shell input swipe 540 900 540 1750 240 >/dev/null 2>&1 || true
    sleep 1
  done
  return 1
}

select_backend() {
  local rid="$1"
  local label="$2"
  ensure_backend_selector_visible || true
  dump_ui "$TMP_XML" || return 1
  tap_by_any_rid "$TMP_XML" "$rid" || {
    if python3 "$QUERY" --xml "$TMP_XML" --text "$label" >/tmp/mnn_bench_query_hit.txt 2>/dev/null; then
      read -r x y _ </tmp/mnn_bench_query_hit.txt
      echo "TAP text=$label x=$x y=$y ts=$(date '+%H:%M:%S')" | tee -a "$UI_LOG"
      adb shell input tap "$x" "$y"
    else
      return 1
    fi
  }
  sleep 1
}

tap_start_test() {
  dump_ui "$TMP_XML" || return 1
  tap_by_any_rid "$TMP_XML" \
    "com.alibaba.mnnllm.android:id/start_test_button_container" \
    "com.alibaba.mnnllm.android:id/start_test_text" \
    "com.alibaba.mnnllm.android.release:id/start_test_button_container" \
    "com.alibaba.mnnllm.android.release:id/start_test_text" || {
      if python3 "$QUERY" --xml "$TMP_XML" --contains-text "开始测试" >/tmp/mnn_bench_query_hit.txt 2>/dev/null; then
        read -r x y _ </tmp/mnn_bench_query_hit.txt
        echo "TAP text=开始测试 x=$x y=$y ts=$(date '+%H:%M:%S')" | tee -a "$UI_LOG"
        adb shell input tap "$x" "$y"
      else
        return 1
      fi
    }
}

wait_ui_result() {
  local timeout_sec="$1"
  local start_ts now
  start_ts="$(date +%s)"
  while true; do
    dump_ui "$TMP_XML" || true
    if python3 "$QUERY" --xml "$TMP_XML" --contains-text "基准测试结果" >/dev/null 2>&1; then
      return 0
    fi
    now="$(date +%s)"
    if [ $((now - start_ts)) -ge "$timeout_sec" ]; then
      return 1
    fi
    sleep 3
  done
}

run_dumpapp_case() {
  local case_name="$1"
  shift
  local log="$ARTIFACT_DIR/qwen35_benchmark/${case_name}.log"
  local try
  for try in 1 2 3; do
    relaunch_and_enter_benchmark || true
    python3 - "$DUMPAPP" "$ACTIVE_MODEL_ID" "$log" "$CASE_TIMEOUT_SEC" "$@" <<'PY'
import subprocess
import sys

dumpapp = sys.argv[1]
model_id = sys.argv[2]
log_path = sys.argv[3]
timeout_sec = int(sys.argv[4])
args = sys.argv[5:]

cmd = [dumpapp, "benchmark", "run", model_id] + args
with open(log_path, "w", encoding="utf-8") as f:
    try:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout_sec)
        sys.exit(p.returncode)
    except subprocess.TimeoutExpired:
        f.write(f"\n[TIMEOUT] benchmark case exceeded {timeout_sec}s\n")
        sys.exit(124)
PY
    if rg -q "Benchmark completed!" "$log"; then
      return 0
    fi
    if rg -q "Unexpected end of stream|Failure to target process|model not found or already running" "$log"; then
      echo "DUMPAPP_RETRY case=$case_name try=$try" | tee -a "$UI_LOG"
      sleep 2
      continue
    fi
  done
  return 1
}

ui_cpu_status="FAIL"
ui_opencl_status="FAIL"
dump_cpu64_status="FAIL"
dump_cpu128_status="FAIL"
dump_opencl_status="FAIL"

relaunch_and_enter_benchmark || true
shot "01_after_launch"

if ensure_benchmark_tab; then
  shot "02_on_benchmark_tab"
else
  echo "UI_BENCHMARK_TAB=FAIL" | tee -a "$UI_LOG"
fi

# UI project - CPU
if relaunch_and_enter_benchmark && select_backend "com.alibaba.mnnllm.android:id/backend_cpu" "CPU" && tap_start_test; then
  if wait_ui_result "$UI_CASE_TIMEOUT_SEC"; then
    ui_cpu_status="PASS"
    shot "05_after_cpu_128_128_result"
  fi
fi

# UI project - OpenCL
if relaunch_and_enter_benchmark && select_backend "com.alibaba.mnnllm.android:id/backend_opencl" "OpenCL" && tap_start_test; then
  if wait_ui_result "$UI_CASE_TIMEOUT_SEC"; then
    ui_opencl_status="PASS"
    shot "06_after_opencl_64_64_result"
  fi
fi

# OpenCL failed => do not keep final screenshot
if [ "$ui_opencl_status" != "PASS" ]; then
  rm -f "$SHOT_DIR/06_after_opencl_64_64_result.png"
fi

# dumpapp project - CPU + OpenCL
"$DUMPAPP" models refresh >"$ARTIFACT_DIR/qwen35_benchmark/models_refresh.txt" || true
"$DUMPAPP" benchmark list >"$ARTIFACT_DIR/qwen35_benchmark/benchmark_list.txt" || true
shot "03_after_benchmark_list"
if ! rg -q "^\\s*${MODEL_ID//\//\\/}\\s+\\[local\\]" "$ARTIFACT_DIR/qwen35_benchmark/benchmark_list.txt"; then
  fallback_model="$(rg "\\[local\\]" "$ARTIFACT_DIR/qwen35_benchmark/benchmark_list.txt" | head -n1 | awk '{print $1}')"
  if [ -n "${fallback_model:-}" ]; then
    ACTIVE_MODEL_ID="$fallback_model"
  fi
fi
echo "ACTIVE_MODEL_ID=$ACTIVE_MODEL_ID" | tee -a "$UI_LOG"

if run_dumpapp_case case_cpu_64_64 --backend cpu --prompt 64 --gen 64 --repeat 1 --threads 4; then
  dump_cpu64_status="PASS"
fi
if run_dumpapp_case case_cpu_128_128 --backend cpu --prompt 128 --gen 128 --repeat 1 --threads 4; then
  dump_cpu128_status="PASS"
fi
if run_dumpapp_case case_opencl_64_64 --backend opencl --prompt 16 --gen 16 --repeat 1 --threads 4; then
  dump_opencl_status="PASS"
fi

cpu_project_status="FAIL"
opencl_project_status="FAIL"
ui_project_status="FAIL"
dumpapp_project_status="FAIL"
overall_status="FAIL"

if [ "$ui_cpu_status" = "PASS" ] && [ "$dump_cpu64_status" = "PASS" ] && [ "$dump_cpu128_status" = "PASS" ]; then
  cpu_project_status="PASS"
fi
if [ "$ui_opencl_status" = "PASS" ] && [ "$dump_opencl_status" = "PASS" ]; then
  opencl_project_status="PASS"
fi
if [ "$ui_cpu_status" = "PASS" ] && [ "$ui_opencl_status" = "PASS" ]; then
  ui_project_status="PASS"
fi
if [ "$dump_cpu64_status" = "PASS" ] && [ "$dump_cpu128_status" = "PASS" ] && [ "$dump_opencl_status" = "PASS" ]; then
  dumpapp_project_status="PASS"
fi
if [ "$ui_project_status" = "PASS" ] && [ "$dumpapp_project_status" = "PASS" ]; then
  overall_status="PASS"
fi

{
  echo "QWEN35_BENCHMARK=$overall_status"
  echo "UI_PROJECT=$ui_project_status"
  echo "DUMPAPP_PROJECT=$dumpapp_project_status"
  echo "CPU_PROJECT=$cpu_project_status"
  echo "OPENCL_PROJECT=$opencl_project_status"
  echo "UI_CPU_CASE=$ui_cpu_status"
  echo "UI_OPENCL_CASE=$ui_opencl_status"
  echo "DUMP_CPU64_CASE=$dump_cpu64_status"
  echo "DUMP_CPU128_CASE=$dump_cpu128_status"
  echo "DUMP_OPENCL_CASE=$dump_opencl_status"
  echo "SCREENSHOTS=$SHOT_DIR"
  echo "UI_ACTION_LOG=$UI_LOG"
  echo "ACTIVE_MODEL_ID=$ACTIVE_MODEL_ID"
} >"$SUMMARY_FILE"

cat "$SUMMARY_FILE"

if [ "$overall_status" != "PASS" ]; then
  exit 1
fi
