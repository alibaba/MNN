#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_DIR="$(cd "$SMOKE_DIR/../.." && pwd)"
ARTIFACT_DIR="${ARTIFACT_DIR:-$SMOKE_DIR/artifacts}"
OUT_DIR="$ARTIFACT_DIR/api_uiautomator"
mkdir -p "$OUT_DIR"

DEVICE_ID="${DEVICE_ID:-$(adb devices | awk 'NR>1 && $2=="device" {print $1; exit}')}"
if [ -z "${DEVICE_ID:-}" ]; then
  echo "No adb device found." >&2
  exit 1
fi

TEST_CLASS="${API_UIAUTOMATOR_TEST_CLASS:-com.alibaba.mnnllm.android.api.ApiSettingsUiAutomatorTest}"
INSTRUMENTATION="com.alibaba.mnnllm.android.test/androidx.test.runner.AndroidJUnitRunner"
LOG_FILE="$OUT_DIR/instrumentation.log"
SUMMARY_FILE="$OUT_DIR/summary.txt"
GESTURE_NOTE="UiAutomator step 09 does not assert history-drawer swipe gesture. For left-swipe verification use mobile-mcp (mobile_swipe_on_screen + screenshot + list_elements)."

export ANDROID_SERIAL="$DEVICE_ID"

pushd "$PROJECT_DIR" >/dev/null
./gradlew :app:assembleStandardDebug :app:assembleStandardDebugAndroidTest >/dev/null
./gradlew :app:installStandardDebug :app:installStandardDebugAndroidTest >/dev/null
popd >/dev/null

# Android 13+ may surface the notification runtime permission on first launch.
# Grant it up front so the UiAutomator flow is not blocked by a system dialog.
adb -s "$DEVICE_ID" shell pm grant com.alibaba.mnnllm.android android.permission.POST_NOTIFICATIONS >/dev/null 2>&1 || true

set +e
adb -s "$DEVICE_ID" shell am instrument -w -r -e class "$TEST_CLASS" "$INSTRUMENTATION" >"$LOG_FILE" 2>&1
RC=$?
set -e

PASS=false
if [ "$RC" -eq 0 ] && rg -q "OK \\(" "$LOG_FILE"; then
  PASS=true
fi

{
  if [ "$PASS" = true ]; then
    echo "API_UIAUTOMATOR_REGRESSION=PASS"
  else
    echo "API_UIAUTOMATOR_REGRESSION=FAIL"
  fi
  echo "DEVICE_ID=$DEVICE_ID"
  echo "TEST_CLASS=$TEST_CLASS"
  echo "LOG_FILE=$LOG_FILE"
  echo "GESTURE_NOTE=$GESTURE_NOTE"
} >"$SUMMARY_FILE"

cat "$SUMMARY_FILE"

if [ "$PASS" != true ]; then
  cat "$LOG_FILE"
  exit 1
fi
