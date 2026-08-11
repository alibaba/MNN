#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
ARTIFACT_DIR="${ARTIFACT_DIR:-$SMOKE_DIR/artifacts}"
mkdir -p "$ARTIFACT_DIR/qwen35_benchmark"

PACKAGE_NAME="${PACKAGE_NAME:-com.alibaba.mnnllm.android}"
MODEL_ID="${QWEN35_BENCH_MODEL_ID:-ModelScope/MNN/Qwen3.5-0.8B-MNN}"
DUMPAPP="${DUMPAPP:-$SMOKE_DIR/../../tools/dumpapp}"
CASE_TIMEOUT_SEC="${CASE_TIMEOUT_SEC:-220}"
UI_LOG="$ARTIFACT_DIR/qwen35_benchmark/ui_actions.log"
SUMMARY_FILE="$ARTIFACT_DIR/qwen35_benchmark/noui_summary.txt"
ACTIVE_MODEL_ID="$MODEL_ID"

ensure_model_ready() {
  "$DUMPAPP" -p "$PACKAGE_NAME" market allow on >"$ARTIFACT_DIR/qwen35_benchmark/market_allow.txt" 2>&1 || true
  "$DUMPAPP" -p "$PACKAGE_NAME" market env prod >"$ARTIFACT_DIR/qwen35_benchmark/market_env.txt" 2>&1 || true
  "$DUMPAPP" -p "$PACKAGE_NAME" market refresh >"$ARTIFACT_DIR/qwen35_benchmark/market_refresh.txt" 2>&1 || true
  "$DUMPAPP" -p "$PACKAGE_NAME" models refresh >"$ARTIFACT_DIR/qwen35_benchmark/models_refresh.txt" 2>&1 || true
  "$DUMPAPP" -p "$PACKAGE_NAME" benchmark list >"$ARTIFACT_DIR/qwen35_benchmark/benchmark_list.txt" 2>&1 || true

  if rg -q "^\\s*${MODEL_ID//\//\\/}\\s+\\[(downloaded|local)\\]" "$ARTIFACT_DIR/qwen35_benchmark/benchmark_list.txt"; then
    ACTIVE_MODEL_ID="$MODEL_ID"
    return 0
  fi

  fallback_model="$(rg "\\[(downloaded|local)\\]" "$ARTIFACT_DIR/qwen35_benchmark/benchmark_list.txt" | head -n1 | awk '{print $1}')"
  if [ -n "${fallback_model:-}" ]; then
    ACTIVE_MODEL_ID="$fallback_model"
    return 0
  fi

  "$DUMPAPP" -p "$PACKAGE_NAME" download test "$MODEL_ID" >"$ARTIFACT_DIR/qwen35_benchmark/download_test.log" 2>&1 || true
  "$DUMPAPP" -p "$PACKAGE_NAME" models refresh >"$ARTIFACT_DIR/qwen35_benchmark/models_refresh_after_download.txt" 2>&1 || true
  "$DUMPAPP" -p "$PACKAGE_NAME" benchmark list >"$ARTIFACT_DIR/qwen35_benchmark/benchmark_list_after_download.txt" 2>&1 || true
  cp "$ARTIFACT_DIR/qwen35_benchmark/benchmark_list_after_download.txt" "$ARTIFACT_DIR/qwen35_benchmark/benchmark_list.txt"

  if rg -q "^\\s*${MODEL_ID//\//\\/}\\s+\\[(downloaded|local)\\]" "$ARTIFACT_DIR/qwen35_benchmark/benchmark_list_after_download.txt"; then
    ACTIVE_MODEL_ID="$MODEL_ID"
    return 0
  fi

  fallback_model="$(rg "\\[(downloaded|local)\\]" "$ARTIFACT_DIR/qwen35_benchmark/benchmark_list_after_download.txt" | head -n1 | awk '{print $1}')"
  if [ -n "${fallback_model:-}" ]; then
    ACTIVE_MODEL_ID="$fallback_model"
    return 0
  fi

  echo "MODEL_READY=FAIL" | tee -a "$UI_LOG"
  return 1
}

run_dumpapp_case() {
  local case_name="$1"
  shift
  local log="$ARTIFACT_DIR/qwen35_benchmark/${case_name}.log"
  local try
  for try in 1 2 3; do
    "$DUMPAPP" -p "$PACKAGE_NAME" benchmark stop >/dev/null 2>&1 || true
    adb shell monkey -p "$PACKAGE_NAME" -c android.intent.category.LAUNCHER 1 >/dev/null 2>&1 || true
    sleep 2
    python3 - "$DUMPAPP" "$PACKAGE_NAME" "$ACTIVE_MODEL_ID" "$log" "$CASE_TIMEOUT_SEC" "$@" <<'PY'
import subprocess
import sys

dumpapp = sys.argv[1]
pkg = sys.argv[2]
model_id = sys.argv[3]
log_path = sys.argv[4]
timeout_sec = int(sys.argv[5])
args = sys.argv[6:]

cmd = [dumpapp, "-p", pkg, "benchmark", "run", model_id] + args
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
    if rg -q "Unexpected end of stream|Failure to target process|model not found or already running|closed \\(is it running\\?\\)" "$log"; then
      echo "NOUI_DUMPAPP_RETRY case=$case_name try=$try" | tee -a "$UI_LOG"
      sleep 2
      continue
    fi
  done
  return 1
}

dump_cpu64_status="FAIL"
dump_cpu128_status="FAIL"
dump_opencl_status="FAIL"
dumpapp_project_status="FAIL"

if ! ensure_model_ready; then
  {
    echo "QWEN35_BENCHMARK_NOUI=FAIL"
    echo "DUMPAPP_PROJECT=FAIL"
    echo "DUMP_CPU64_CASE=FAIL"
    echo "DUMP_CPU128_CASE=FAIL"
    echo "DUMP_OPENCL_CASE=FAIL"
    echo "ACTIVE_MODEL_ID=NONE"
    echo "MODEL_READY=FAIL"
  } >"$SUMMARY_FILE"
  cat "$SUMMARY_FILE"
  exit 1
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

if [ "$dump_cpu64_status" = "PASS" ] && [ "$dump_cpu128_status" = "PASS" ] && [ "$dump_opencl_status" = "PASS" ]; then
  dumpapp_project_status="PASS"
fi

{
  echo "QWEN35_BENCHMARK_NOUI=$dumpapp_project_status"
  echo "DUMPAPP_PROJECT=$dumpapp_project_status"
  echo "DUMP_CPU64_CASE=$dump_cpu64_status"
  echo "DUMP_CPU128_CASE=$dump_cpu128_status"
  echo "DUMP_OPENCL_CASE=$dump_opencl_status"
  echo "ACTIVE_MODEL_ID=$ACTIVE_MODEL_ID"
} >"$SUMMARY_FILE"

cat "$SUMMARY_FILE"

if [ "$dumpapp_project_status" != "PASS" ]; then
  exit 1
fi
