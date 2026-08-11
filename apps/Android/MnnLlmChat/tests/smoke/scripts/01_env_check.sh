#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$(cd "$SMOKE_DIR/../../../../.." && pwd)"

DEVICE_ID="${DEVICE_ID:-}"
BUILD_KIND="${BUILD_KIND:-aab_release}"
PACKAGE_NAME="${PACKAGE_NAME:-}"
AAB_PATH="${AAB_PATH:-$ROOT_DIR/apps/Android/MnnLlmChat/release_outputs/googleplay/app-googleplay-release.aab}"
DEBUG_APK_PATH="${DEBUG_APK_PATH:-$ROOT_DIR/apps/Android/MnnLlmChat/app/build/outputs/apk/standard/debug/app-standard-debug.apk}"
BUNDLETOOL_JAR="${BUNDLETOOL_JAR:-/tmp/bundletool-all-1.17.1.jar}"
ARTIFACT_DIR="${ARTIFACT_DIR:-$SMOKE_DIR/artifacts}"

mkdir -p "$ARTIFACT_DIR"

if [[ "$BUILD_KIND" == "aab_release" ]]; then
  if [[ ! -f "$AAB_PATH" ]]; then
    echo "AAB not found: $AAB_PATH" >&2
    exit 1
  fi
  if [[ ! -f "$BUNDLETOOL_JAR" ]]; then
    echo "bundletool-all jar not found: $BUNDLETOOL_JAR" >&2
    exit 1
  fi
  if [[ -z "$PACKAGE_NAME" ]]; then
    PACKAGE_NAME="com.alibaba.mnnllm.android.release"
  fi
elif [[ "$BUILD_KIND" == "standard_debug" ]]; then
  if [[ ! -f "$DEBUG_APK_PATH" ]]; then
    echo "Debug APK not found: $DEBUG_APK_PATH" >&2
    exit 1
  fi
  if [[ -z "$PACKAGE_NAME" ]]; then
    PACKAGE_NAME="com.alibaba.mnnllm.android"
  fi
else
  echo "Unsupported BUILD_KIND: $BUILD_KIND (use aab_release or standard_debug)" >&2
  exit 1
fi

if [[ -z "$DEVICE_ID" ]]; then
  DEVICE_ID="$(adb devices | awk 'NR>1 && $2=="device" {print $1; exit}')"
fi

if [[ -z "$DEVICE_ID" ]]; then
  echo "No adb device found." >&2
  exit 1
fi

{
  echo "DEVICE_ID=$DEVICE_ID"
  echo "BUILD_KIND=$BUILD_KIND"
  echo "PACKAGE_NAME=$PACKAGE_NAME"
  echo "AAB_PATH=$AAB_PATH"
  echo "DEBUG_APK_PATH=$DEBUG_APK_PATH"
  echo "BUNDLETOOL_JAR=$BUNDLETOOL_JAR"
  echo "ARTIFACT_DIR=$ARTIFACT_DIR"
} >"$ARTIFACT_DIR/smoke_env.txt"

echo "Environment check passed."
