#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACT_DIR="${ARTIFACT_DIR:-$SMOKE_DIR/artifacts}"
ENV_FILE="$ARTIFACT_DIR/smoke_env.txt"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing env file: $ENV_FILE. Run 01_env_check.sh first." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$ENV_FILE"

adb -s "$DEVICE_ID" shell dumpsys window windows >"$ARTIFACT_DIR/window_dump.txt"
adb -s "$DEVICE_ID" shell uiautomator dump /sdcard/ui_dump.xml >/dev/null
adb -s "$DEVICE_ID" pull /sdcard/ui_dump.xml "$ARTIFACT_DIR/ui_dump.xml" >/dev/null
adb -s "$DEVICE_ID" exec-out screencap -p >"$ARTIFACT_DIR/main_screenshot.png"

echo "UI state captured."
