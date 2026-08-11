#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
ARTIFACT_DIR="${ARTIFACT_DIR:-$SMOKE_DIR/artifacts}"
PACKAGE_NAME="${PACKAGE_NAME:-com.alibaba.mnnllm.android}"
DUMPAPP="${DUMPAPP:-$SMOKE_DIR/../../tools/dumpapp}"
ALLOW_MISSING_MODELS="${ALLOW_MISSING_MODELS:-false}"

OUT_DIR="$ARTIFACT_DIR/sana_diffusion_dumpapp"
mkdir -p "$OUT_DIR"

SANA_LIST_LOG="$OUT_DIR/sana_list.log"
DIFFUSION_LIST_LOG="$OUT_DIR/diffusion_list.log"
SANA_RUN_LOG="$OUT_DIR/sana_run.log"
DIFFUSION_RUN_LOG="$OUT_DIR/diffusion_run.log"
SUMMARY_FILE="$OUT_DIR/summary.txt"

SANA_PROMPT="${SANA_PROMPT:-anime portrait, clean line art, high quality}"
SANA_STEPS="${SANA_STEPS:-2}"
SANA_SEED="${SANA_SEED:-42}"
SANA_WIDTH="${SANA_WIDTH:-256}"
SANA_HEIGHT="${SANA_HEIGHT:-256}"

DIFFUSION_PROMPT="${DIFFUSION_PROMPT:-a cute cat, clean background, high quality}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-8}"
DIFFUSION_SEED="${DIFFUSION_SEED:-42}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
SANA_OUTPUT_REMOTE="${SANA_OUTPUT_REMOTE:-/sdcard/smoke_sana_${RUN_ID}.jpg}"
DIFFUSION_OUTPUT_REMOTE="${DIFFUSION_OUTPUT_REMOTE:-/sdcard/smoke_diffusion_${RUN_ID}.jpg}"

trim_cr() {
  tr -d '\r'
}

resolve_sana_model_path() {
  if [ -n "${SANA_MODEL_PATH:-}" ]; then
    echo "$SANA_MODEL_PATH"
    return 0
  fi

  local from_local
  from_local="$(awk '/^\s*\/[^ ]+ \[VALID\]/{print $1; exit}' "$SANA_LIST_LOG" | trim_cr || true)"
  if [ -n "$from_local" ]; then
    echo "$from_local"
    return 0
  fi

  awk '/Local: true, Path: /{
    sub(/^.*Path: /, "", $0);
    gsub(/\r/, "", $0);
    if ($0 != "" && $0 != "null") {
      print $0;
      exit
    }
  }' "$SANA_LIST_LOG" || true
}

resolve_diffusion_model_id() {
  if [ -n "${DIFFUSION_MODEL_ID:-}" ]; then
    echo "$DIFFUSION_MODEL_ID"
    return 0
  fi

  awk '/\[(downloaded|local)\]/{print $1; exit}' "$DIFFUSION_LIST_LOG" | trim_cr || true
}

is_true() {
  [ "$(echo "$1" | tr '[:upper:]' '[:lower:]')" = "true" ]
}

require_or_skip() {
  local value="$1"
  local case_name="$2"
  local reason="$3"

  if [ -n "$value" ]; then
    return 0
  fi

  if is_true "$ALLOW_MISSING_MODELS"; then
    echo "${case_name}=SKIP_MISSING_MODEL ($reason)" | tee -a "$OUT_DIR/warnings.log"
    return 2
  fi

  echo "${case_name}=FAIL_MISSING_MODEL ($reason)" | tee -a "$OUT_DIR/warnings.log"
  return 1
}

echo "Launching app to ensure dumpapp target process is alive..."
adb shell monkey -p "$PACKAGE_NAME" -c android.intent.category.LAUNCHER 1 >/dev/null 2>&1 || true
sleep 2

"$DUMPAPP" -p "$PACKAGE_NAME" market allow on >"$OUT_DIR/market_allow.log" 2>&1 || true
"$DUMPAPP" -p "$PACKAGE_NAME" market env prod >"$OUT_DIR/market_env.log" 2>&1 || true
"$DUMPAPP" -p "$PACKAGE_NAME" market refresh >"$OUT_DIR/market_refresh.log" 2>&1 || true
"$DUMPAPP" -p "$PACKAGE_NAME" models refresh >"$OUT_DIR/models_refresh.log" 2>&1 || true

"$DUMPAPP" -p "$PACKAGE_NAME" sana list >"$SANA_LIST_LOG" 2>&1 || true
"$DUMPAPP" -p "$PACKAGE_NAME" diffusion list >"$DIFFUSION_LIST_LOG" 2>&1 || true

SANA_MODEL_PATH_RESOLVED="$(resolve_sana_model_path)"
DIFFUSION_MODEL_ID_RESOLVED="$(resolve_diffusion_model_id)"

sana_status="FAIL"
diffusion_status="FAIL"
overall="FAIL"

check_sana=0
require_or_skip "$SANA_MODEL_PATH_RESOLVED" "SANA_DUMPAPP_CASE" "no valid model path found in sana list" || check_sana=$?
if [ "$check_sana" -eq 0 ]; then
  "$DUMPAPP" -p "$PACKAGE_NAME" sana run "$SANA_MODEL_PATH_RESOLVED" "$SANA_PROMPT" \
    --steps "$SANA_STEPS" \
    --seed "$SANA_SEED" \
    --width "$SANA_WIDTH" \
    --height "$SANA_HEIGHT" \
    --output "$SANA_OUTPUT_REMOTE" >"$SANA_RUN_LOG" 2>&1 || true

  if rg -qi "^Success:\s*true" "$SANA_RUN_LOG" \
    && adb shell "[ -s '$SANA_OUTPUT_REMOTE' ]" >/dev/null 2>&1; then
    sana_status="PASS"
    adb pull "$SANA_OUTPUT_REMOTE" "$OUT_DIR/" >"$OUT_DIR/sana_pull.log" 2>&1 || true
  fi
elif [ "$check_sana" -eq 2 ]; then
  sana_status="SKIP"
fi

check_diffusion=0
require_or_skip "$DIFFUSION_MODEL_ID_RESOLVED" "DIFFUSION_DUMPAPP_CASE" "no downloaded diffusion model in diffusion list" || check_diffusion=$?
if [ "$check_diffusion" -eq 0 ]; then
  "$DUMPAPP" -p "$PACKAGE_NAME" diffusion run "$DIFFUSION_MODEL_ID_RESOLVED" "$DIFFUSION_PROMPT" \
    --steps "$DIFFUSION_STEPS" \
    --seed "$DIFFUSION_SEED" \
    --output "$DIFFUSION_OUTPUT_REMOTE" >"$DIFFUSION_RUN_LOG" 2>&1 || true

  if rg -qi "^Success:\s*true" "$DIFFUSION_RUN_LOG" \
    && adb shell "[ -s '$DIFFUSION_OUTPUT_REMOTE' ]" >/dev/null 2>&1; then
    diffusion_status="PASS"
    adb pull "$DIFFUSION_OUTPUT_REMOTE" "$OUT_DIR/" >"$OUT_DIR/diffusion_pull.log" 2>&1 || true
  fi
elif [ "$check_diffusion" -eq 2 ]; then
  diffusion_status="SKIP"
fi

if [ "$sana_status" = "PASS" ] && [ "$diffusion_status" = "PASS" ]; then
  overall="PASS"
elif [ "$sana_status" = "SKIP" ] || [ "$diffusion_status" = "SKIP" ]; then
  overall="SKIP"
fi

{
  echo "SANA_DIFFUSION_DUMPAPP=$overall"
  echo "SANA_DUMPAPP_CASE=$sana_status"
  echo "DIFFUSION_DUMPAPP_CASE=$diffusion_status"
  echo "SANA_MODEL_PATH=${SANA_MODEL_PATH_RESOLVED:-NONE}"
  echo "DIFFUSION_MODEL_ID=${DIFFUSION_MODEL_ID_RESOLVED:-NONE}"
  echo "SANA_OUTPUT_REMOTE=$SANA_OUTPUT_REMOTE"
  echo "DIFFUSION_OUTPUT_REMOTE=$DIFFUSION_OUTPUT_REMOTE"
  echo "ALLOW_MISSING_MODELS=$ALLOW_MISSING_MODELS"
  echo "SANA_LIST_LOG=$SANA_LIST_LOG"
  echo "DIFFUSION_LIST_LOG=$DIFFUSION_LIST_LOG"
  echo "SANA_RUN_LOG=$SANA_RUN_LOG"
  echo "DIFFUSION_RUN_LOG=$DIFFUSION_RUN_LOG"
} >"$SUMMARY_FILE"

cat "$SUMMARY_FILE"

if [ "$overall" != "PASS" ]; then
  exit 1
fi
