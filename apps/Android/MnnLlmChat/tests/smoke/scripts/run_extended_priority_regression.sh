#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACT_DIR="${ARTIFACT_DIR:-$SMOKE_DIR/artifacts}"
mkdir -p "$ARTIFACT_DIR"
STEP_WATCH_TIMEOUT_SEC="${STEP_WATCH_TIMEOUT_SEC:-1800}"

step_status() {
  local k="$1"
  local v="$2"
  echo "${k}=${v}" >>"$ARTIFACT_DIR/extended_priority_steps.txt"
}

run_step_with_watch() {
  local step_key="$1"
  local step_label="$2"
  local cmd="$3"
  local watch_file="${4:-}"
  local timeout_sec="${5:-$STEP_WATCH_TIMEOUT_SEC}"

  echo "$step_label"
  local start_ts now pid
  start_ts="$(date +%s)"

  bash -lc "$cmd" &
  pid=$!

  while kill -0 "$pid" >/dev/null 2>&1; do
    now="$(date +%s)"
    if [ $((now - start_ts)) -ge "$timeout_sec" ]; then
      kill "$pid" >/dev/null 2>&1 || true
      wait "$pid" || true
      step_status "$step_key" "FAIL_TIMEOUT"
      overall="FAIL"
      return 1
    fi
    if [ -n "$watch_file" ] && [ -f "$watch_file" ]; then
      # heartbeat by touching a stamp to indicate progress file exists
      : >"$ARTIFACT_DIR/.${step_key}_watch_seen"
    fi
    sleep 2
  done

  if wait "$pid"; then
    step_status "$step_key" PASS
    return 0
  fi
  step_status "$step_key" FAIL
  overall="FAIL"
  return 1
}

: >"$ARTIFACT_DIR/extended_priority_steps.txt"
overall="PASS"

run_step_with_watch "STEP1" "[STEP 1] Baseline install/launch smoke (standardDebug + aabRelease)" \
  "$SCRIPT_DIR/run_priority_regression.sh" \
  "$ARTIFACT_DIR/priority_regression_summary.txt"

run_step_with_watch "STEP2" "[STEP 2] Qwen3.5 benchmark multi-case regression" \
  "$SCRIPT_DIR/05_regress_qwen35_benchmark.sh" \
  "$ARTIFACT_DIR/qwen35_benchmark/summary.txt"

run_step_with_watch "STEP3" "[STEP 3] Chat text/image input entry regression" \
  "$SCRIPT_DIR/06_regress_chat_text_image.sh" \
  "$ARTIFACT_DIR/chat_io/summary.txt"

run_step_with_watch "STEP4" "[STEP 4] Qwen3.5 download pause/resume/delete regression" \
  "$SCRIPT_DIR/04_regress_qwen35_download_ops.sh" \
  "$ARTIFACT_DIR/qwen35_download/summary.txt"

run_step_with_watch "STEP5" "[STEP 5] Generate single-page HTML report" \
  "$SCRIPT_DIR/07_generate_report.sh" \
  "$ARTIFACT_DIR/report.html" \
  300

{
  echo "EXTENDED_PRIORITY_REGRESSION=${overall}"
  echo "COVERAGE=text_input,image_input_entry,benchmark_multi_case,download_pause_resume_delete"
  echo "ARTIFACT_ROOT=$ARTIFACT_DIR"
  echo "REPORT_HTML=$ARTIFACT_DIR/report.html"
} >"$ARTIFACT_DIR/extended_priority_summary.txt"

cat "$ARTIFACT_DIR/extended_priority_summary.txt"
cat "$ARTIFACT_DIR/extended_priority_steps.txt"

if [ "$overall" != "PASS" ]; then
  exit 1
fi
