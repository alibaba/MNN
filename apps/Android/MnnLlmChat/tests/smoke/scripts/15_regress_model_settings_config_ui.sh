#!/usr/bin/env bash
set -euo pipefail

# Guards issue #4259: model settings from BOTH home and ChatActivity must not corrupt config.
# 1. Runs ModelSettingsConfigUiAutomatorTest (home + chat flows, both edit System Prompt -> save -> send)
# 2. After each flow, dumpapp config dump validates merged config (llm_model, llm_weight not empty)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_DIR="$(cd "$SMOKE_DIR/../.." && pwd)"
ARTIFACT_DIR="${ARTIFACT_DIR:-$SMOKE_DIR/artifacts}"
OUT_DIR="$ARTIFACT_DIR/model_settings_config_ui"
mkdir -p "$OUT_DIR"

DEVICE_ID="${DEVICE_ID:-$(adb devices | awk 'NR>1 && $2=="device" {print $1; exit}')}"
if [ -z "${DEVICE_ID:-}" ]; then
  echo "No adb device found." >&2
  exit 1
fi

MODEL_ID="${MODEL_ID:-ModelScope/MNN/Qwen3.5-0.8B-MNN}"
PACKAGE_NAME="${PACKAGE_NAME:-com.alibaba.mnnllm.android}"
TEST_CLASS="com.alibaba.mnnllm.android.modelsettings.ModelSettingsConfigUiAutomatorTest"
INSTRUMENTATION="com.alibaba.mnnllm.android.test/androidx.test.runner.AndroidJUnitRunner"
LOG_FILE="$OUT_DIR/instrumentation.log"
SUMMARY_FILE="$OUT_DIR/summary.txt"
DUMPAPP="${DUMPAPP:-$PROJECT_DIR/tools/dumpapp}"
CONFIG_DUMP_LOG="$OUT_DIR/config_dump_after.log"
ENSURE_LOG="$OUT_DIR/model_ensure.log"

export ANDROID_SERIAL="$DEVICE_ID"

pushd "$PROJECT_DIR" >/dev/null
./gradlew :app:assembleStandardDebug :app:assembleStandardDebugAndroidTest >/dev/null
./gradlew :app:installStandardDebug :app:installStandardDebugAndroidTest >/dev/null
popd >/dev/null

# Ensure model is available (E2E may have uninstalled at start, leaving no models)
adb -s "$DEVICE_ID" shell am start -W -n "$PACKAGE_NAME/com.alibaba.mnnllm.android.main.MainActivity" >/dev/null 2>&1 || true
sleep 3
"$DUMPAPP" -p "$PACKAGE_NAME" market allow on >>"$ENSURE_LOG" 2>&1 || true
"$DUMPAPP" -p "$PACKAGE_NAME" market refresh >>"$ENSURE_LOG" 2>&1 || true
"$DUMPAPP" -p "$PACKAGE_NAME" models refresh >>"$ENSURE_LOG" 2>&1 || true
"$DUMPAPP" -p "$PACKAGE_NAME" benchmark list >"$OUT_DIR/benchmark_list.txt" 2>>"$ENSURE_LOG" || true
if ! rg -q "\[(downloaded|local)\]" "$OUT_DIR/benchmark_list.txt" 2>/dev/null; then
  echo "[15] No model ready, triggering download for $MODEL_ID" >>"$ENSURE_LOG"
  "$DUMPAPP" -p "$PACKAGE_NAME" download test "$MODEL_ID" >>"$ENSURE_LOG" 2>&1 || true
  sleep 30
  "$DUMPAPP" -p "$PACKAGE_NAME" models refresh >>"$ENSURE_LOG" 2>&1 || true
  "$DUMPAPP" -p "$PACKAGE_NAME" benchmark list >"$OUT_DIR/benchmark_list.txt" 2>>"$ENSURE_LOG" || true
fi

set +e
adb -s "$DEVICE_ID" shell am instrument -w -r -e class "$TEST_CLASS" "$INSTRUMENTATION" >"$LOG_FILE" 2>&1
RC=$?
set -e

PASS=false
if [ "$RC" -eq 0 ] && rg -q "OK \\(" "$LOG_FILE"; then
  PASS=true
fi

if [ "$PASS" = true ]; then
  adb shell am start -W -n com.alibaba.mnnllm.android/com.alibaba.mnnllm.android.main.MainActivity >/dev/null 2>&1 || true
  sleep 2
  "$DUMPAPP" -p "$PACKAGE_NAME" config dump "$MODEL_ID" >"$CONFIG_DUMP_LOG" 2>&1 || true
  if ! rg -q '^RESULT=OK$' "$CONFIG_DUMP_LOG"; then
    echo "[15] config dump after UI test FAILED; merged config may be corrupted" >&2
    cat "$CONFIG_DUMP_LOG" >&2
    PASS=false
  fi
fi

{
  if [ "$PASS" = true ]; then
    echo "MODEL_SETTINGS_CONFIG_UI_REGRESSION=PASS"
  else
    echo "MODEL_SETTINGS_CONFIG_UI_REGRESSION=FAIL"
  fi
  echo "DEVICE_ID=$DEVICE_ID"
  echo "TEST_CLASS=$TEST_CLASS"
  echo "MODEL_ID=$MODEL_ID"
  echo "LOG_FILE=$LOG_FILE"
  echo "CONFIG_DUMP_LOG=$CONFIG_DUMP_LOG"
  echo "NOTE=Guards #4259: home + chat settings -> save -> send; config dump verifies merged config"
} >"$SUMMARY_FILE"

cat "$SUMMARY_FILE"

if [ "$PASS" != true ]; then
  cat "$LOG_FILE"
  exit 1
fi
