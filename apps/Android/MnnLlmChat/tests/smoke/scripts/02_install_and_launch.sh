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

APKS_PATH="/tmp/mnn_chat_googleplay_smoke.apks"
UNINSTALL_CONFLICTING="${UNINSTALL_CONFLICTING:-true}"

if [[ "$UNINSTALL_CONFLICTING" == "true" ]]; then
  adb -s "$DEVICE_ID" uninstall "$PACKAGE_NAME" >/dev/null 2>&1 || true
fi

if [[ "$BUILD_KIND" == "aab_release" ]]; then
  java -jar "$BUNDLETOOL_JAR" build-apks \
    --bundle="$AAB_PATH" \
    --output="$APKS_PATH" \
    --mode=universal \
    --overwrite

  java -jar "$BUNDLETOOL_JAR" install-apks \
    --apks="$APKS_PATH" \
    --device-id="$DEVICE_ID"
elif [[ "$BUILD_KIND" == "standard_debug" ]]; then
  adb -s "$DEVICE_ID" install -r -d "$DEBUG_APK_PATH"
else
  echo "Unsupported BUILD_KIND: $BUILD_KIND" >&2
  exit 1
fi

adb -s "$DEVICE_ID" shell pm list packages | rg "$PACKAGE_NAME" >"$ARTIFACT_DIR/package_list.txt"
adb -s "$DEVICE_ID" shell dumpsys package "$PACKAGE_NAME" | rg -n "versionName|versionCode|minSdk|targetSdk" >"$ARTIFACT_DIR/package_info.txt"
adb -s "$DEVICE_ID" shell monkey -p "$PACKAGE_NAME" -c android.intent.category.LAUNCHER 1 >"$ARTIFACT_DIR/launch_result.txt"

echo "Install and launch completed."
