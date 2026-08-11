#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACT_DIR="${ARTIFACT_DIR:-$SMOKE_DIR/artifacts}"
mkdir -p "$ARTIFACT_DIR/qwen35_benchmark"

PACKAGE_NAME="${PACKAGE_NAME:-com.alibaba.mnnllm.android}"
SHOT_DIR="$ARTIFACT_DIR/qwen35_benchmark/shots"
mkdir -p "$SHOT_DIR"
QUERY="$SCRIPT_DIR/ui_xml_query.py"
TMP_XML="/tmp/mnn_benchmark_ui.xml"
UI_CASE_TIMEOUT_SEC="${UI_CASE_TIMEOUT_SEC:-300}"
OPENCL_UI_TIMEOUT_SEC="${OPENCL_UI_TIMEOUT_SEC:-60}"
UI_LOG="$ARTIFACT_DIR/qwen35_benchmark/ui_actions.log"
SUMMARY_FILE="$ARTIFACT_DIR/qwen35_benchmark/ui_summary.txt"
rm -f "$SHOT_DIR"/*.png "$SUMMARY_FILE"
: >"$UI_LOG"

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
    dump_ui "$TMP_XML" || true
    if tap_by_any_rid "$TMP_XML" \
      "com.alibaba.mnnllm.android:id/tab_benchmark" \
      "com.alibaba.mnnllm.android.release:id/tab_benchmark"; then
      sleep 3
      return 0
    fi
    adb shell input keyevent 4 >/dev/null 2>&1 || true
    sleep 1
  done
  return 1
}

restart_app_for_clean_state() {
  local reason="${1:-RESTART_APP}"
  echo "APP_RECOVERY=$reason ts=$(date '+%H:%M:%S')" | tee -a "$UI_LOG"
  adb shell am force-stop "$PACKAGE_NAME" >/dev/null 2>&1 || true
  sleep 1
  ensure_device_unlocked
  adb shell monkey -p "$PACKAGE_NAME" -c android.intent.category.LAUNCHER 1 >/dev/null 2>&1 || true
  sleep 2
}

relaunch_and_enter_benchmark() {
  ensure_device_unlocked
  adb shell monkey -p "$PACKAGE_NAME" -c android.intent.category.LAUNCHER 1 >/dev/null
  sleep 2
  ensure_benchmark_tab
}

ensure_backend_selector_visible() {
  local try
  # Get display height for device-agnostic scroll coordinates
  local h
  h=$(adb shell wm size 2>/dev/null | sed -n 's/.*: \([0-9]*\)x[0-9]*/\1/p' || echo "2400")
  local cy=$((h / 2))
  local top_y=$((h / 6))
  local bottom_y=$((h * 5 / 6))
  for try in 1 2 3 4 5; do
    dump_ui "$TMP_XML" || true
    if python3 "$QUERY" --xml "$TMP_XML" --resource-id "com.alibaba.mnnllm.android:id/backend_selector_group" >/dev/null 2>&1 \
      || python3 "$QUERY" --xml "$TMP_XML" --resource-id "com.alibaba.mnnllm.android.release:id/backend_selector_group" >/dev/null 2>&1 \
      || python3 "$QUERY" --xml "$TMP_XML" --contains-text "CPU" >/dev/null 2>&1; then
      return 0
    fi
    # Scroll down to reveal backend (finger moves down = see content below)
    echo "SWIPE_DOWN try=$try ts=$(date '+%H:%M:%S')" | tee -a "$UI_LOG"
    adb shell input swipe "$cy" "$cy" "$cy" "$bottom_y" 300 >/dev/null 2>&1 || true
    sleep 1
    dump_ui "$TMP_XML" || true
    if python3 "$QUERY" --xml "$TMP_XML" --contains-text "CPU" >/dev/null 2>&1; then
      return 0
    fi
    # Alternate: scroll up (in case we're past the backend)
    echo "SWIPE_UP try=$try ts=$(date '+%H:%M:%S')" | tee -a "$UI_LOG"
    adb shell input swipe "$cy" "$cy" "$cy" "$top_y" 300 >/dev/null 2>&1 || true
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
    elif python3 "$QUERY" --xml "$TMP_XML" --contains-text "$label" >/tmp/mnn_bench_query_hit.txt 2>/dev/null; then
      read -r x y _ </tmp/mnn_bench_query_hit.txt
      echo "TAP contains=$label x=$x y=$y ts=$(date '+%H:%M:%S')" | tee -a "$UI_LOG"
      adb shell input tap "$x" "$y"
    else
      return 1
    fi
  }
  sleep 1
}

tap_start_test() {
  dump_ui "$TMP_XML" || return 1
  if python3 "$QUERY" --xml "$TMP_XML" --contains-text "停止测试" >/dev/null 2>&1; then
    echo "START_SKIPPED_ALREADY_RUNNING ts=$(date '+%H:%M:%S')" | tee -a "$UI_LOG"
    return 2
  fi
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

is_result_visible() {
  local xml="$1"
  if python3 "$QUERY" --xml "$xml" --contains-text "基准测试结果" >/dev/null 2>&1; then
    return 0
  fi
  if python3 "$QUERY" --xml "$xml" --contains-text "预填充速度" >/dev/null 2>&1 \
    && python3 "$QUERY" --xml "$xml" --contains-text "解码速度" >/dev/null 2>&1 \
    && python3 "$QUERY" --xml "$xml" --contains-text "总时间" >/dev/null 2>&1; then
    return 0
  fi
  return 1
}

is_running_visible() {
  local xml="$1"
  if python3 "$QUERY" --xml "$xml" --contains-text "停止测试" >/dev/null 2>&1 \
    || python3 "$QUERY" --xml "$xml" --contains-text "正在运行性能测试" >/dev/null 2>&1 \
    || python3 "$QUERY" --xml "$xml" --resource-id "com.alibaba.mnnllm.android:id/start_test_progress" >/dev/null 2>&1 \
    || python3 "$QUERY" --xml "$xml" --resource-id "com.alibaba.mnnllm.android.release:id/start_test_progress" >/dev/null 2>&1; then
    return 0
  fi
  return 1
}

ensure_not_running() {
  local timeout_sec="${1:-15}"
  local start_ts now
  start_ts="$(date +%s)"
  while true; do
    dump_ui "$TMP_XML" || true
    if ! is_running_visible "$TMP_XML"; then
      return 0
    fi
    now="$(date +%s)"
    if [ $((now - start_ts)) -ge "$timeout_sec" ]; then
      return 1
    fi
    sleep 2
  done
}

wait_ui_result() {
  local timeout_sec="$1"
  local start_ts now
  start_ts="$(date +%s)"
  while true; do
    dump_ui "$TMP_XML" || true
    if is_result_visible "$TMP_XML"; then
      return 0
    fi
    now="$(date +%s)"
    if [ $((now - start_ts)) -ge "$timeout_sec" ]; then
      return 1
    fi
    sleep 3
  done
}

ui_cpu_status="FAIL"
ui_opencl_status="FAIL"
opencl_timeout="false"
ui_project_status="FAIL"

relaunch_and_enter_benchmark || true
shot "01_after_launch"

if ensure_benchmark_tab; then
  shot "02_on_benchmark_tab"
else
  echo "UI_BENCHMARK_TAB=FAIL" | tee -a "$UI_LOG"
fi

for cpu_try in 1 2; do
  if ! relaunch_and_enter_benchmark || ! select_backend "com.alibaba.mnnllm.android:id/backend_cpu" "CPU"; then
    continue
  fi
  if ! ensure_not_running 12; then
    echo "CPU_PRECHECK_BUSY try=$cpu_try ts=$(date '+%H:%M:%S')" | tee -a "$UI_LOG"
    restart_app_for_clean_state "CPU_PRECHECK_BUSY"
    continue
  fi
  if tap_start_test; then
    if wait_ui_result "$UI_CASE_TIMEOUT_SEC"; then
      ui_cpu_status="PASS"
      shot "05_after_cpu_128_128_result"
      break
    fi
    echo "CPU_WAIT_TIMEOUT try=$cpu_try ts=$(date '+%H:%M:%S')" | tee -a "$UI_LOG"
    restart_app_for_clean_state "CPU_WAIT_TIMEOUT"
    continue
  fi
  cpu_start_rc="$?"
  if [ "$cpu_start_rc" -eq 2 ]; then
    echo "CPU_START_SKIPPED_ALREADY_RUNNING try=$cpu_try ts=$(date '+%H:%M:%S')" | tee -a "$UI_LOG"
    restart_app_for_clean_state "CPU_ALREADY_RUNNING"
    continue
  fi
done

if relaunch_and_enter_benchmark && select_backend "com.alibaba.mnnllm.android:id/backend_opencl" "OpenCL"; then
  if ! ensure_not_running 12; then
    echo "OPENCL_PRECHECK_BUSY=RESTART_APP ts=$(date '+%H:%M:%S')" | tee -a "$UI_LOG"
    restart_app_for_clean_state "OPENCL_PRECHECK_BUSY"
    relaunch_and_enter_benchmark && select_backend "com.alibaba.mnnllm.android:id/backend_opencl" "OpenCL" || true
  fi
  if tap_start_test; then
    if wait_ui_result "$OPENCL_UI_TIMEOUT_SEC"; then
      ui_opencl_status="PASS"
      shot "06_after_opencl_64_64_result"
    else
      opencl_timeout="true"
      restart_app_for_clean_state "OPENCL_WAIT_TIMEOUT"
    fi
  else
    opencl_start_rc="$?"
    if [ "$opencl_start_rc" -eq 2 ]; then
      opencl_timeout="true"
      echo "OPENCL_START_SKIPPED_ALREADY_RUNNING ts=$(date '+%H:%M:%S')" | tee -a "$UI_LOG"
      restart_app_for_clean_state "OPENCL_ALREADY_RUNNING"
    fi
  fi
fi

if [ "$ui_opencl_status" != "PASS" ]; then
  rm -f "$SHOT_DIR/06_after_opencl_64_64_result.png"
fi

if [ "$ui_cpu_status" = "PASS" ] && [ "$ui_opencl_status" = "PASS" ]; then
  ui_project_status="PASS"
fi

{
  echo "QWEN35_BENCHMARK_UI=$ui_project_status"
  echo "UI_PROJECT=$ui_project_status"
  echo "UI_CPU_CASE=$ui_cpu_status"
  echo "UI_OPENCL_CASE=$ui_opencl_status"
  echo "OPENCL_UI_TIMEOUT_SEC=$OPENCL_UI_TIMEOUT_SEC"
  echo "OPENCL_UI_TIMEOUT_HIT=$opencl_timeout"
  echo "SCREENSHOTS=$SHOT_DIR"
  echo "UI_ACTION_LOG=$UI_LOG"
} >"$SUMMARY_FILE"

cat "$SUMMARY_FILE"

if [ "$ui_project_status" != "PASS" ]; then
  exit 1
fi
