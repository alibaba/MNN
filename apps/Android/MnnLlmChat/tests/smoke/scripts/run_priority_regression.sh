#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACT_DIR="${ARTIFACT_DIR:-$SMOKE_DIR/artifacts}"
UNINSTALL_AT_START="${UNINSTALL_AT_START:-false}"
UNINSTALL_PACKAGES="${UNINSTALL_PACKAGES:-com.alibaba.mnnllm.android com.alibaba.mnnllm.android.release}"
DEVICE_ID="${DEVICE_ID:-$(adb devices | awk 'NR>1 && $2=="device" {print $1; exit}')}"

rm -rf "$ARTIFACT_DIR/standard_debug" "$ARTIFACT_DIR/aab_release"
mkdir -p "$ARTIFACT_DIR/standard_debug" "$ARTIFACT_DIR/aab_release"

if [[ "$UNINSTALL_AT_START" == "true" ]]; then
  if [[ -z "$DEVICE_ID" ]]; then
    echo "No adb device found for uninstall-at-start." >&2
    exit 1
  fi
  : >"$ARTIFACT_DIR/uninstall_at_start.log"
  for pkg in $UNINSTALL_PACKAGES; do
    echo "UNINSTALL pkg=$pkg device=$DEVICE_ID" | tee -a "$ARTIFACT_DIR/uninstall_at_start.log"
    adb -s "$DEVICE_ID" uninstall "$pkg" >>"$ARTIFACT_DIR/uninstall_at_start.log" 2>&1 || true
  done
fi

echo "[1/2] Running standard_debug smoke (main target)..."
BUILD_KIND=standard_debug ARTIFACT_DIR="$ARTIFACT_DIR/standard_debug" DEVICE_ID="$DEVICE_ID" UNINSTALL_CONFLICTING=false "$SCRIPT_DIR/run_smoke.sh"

AAB_SKIPPED="false"
if [[ -f "${BUNDLETOOL_JAR:-/tmp/bundletool-all-1.17.1.jar}" ]]; then
  echo "[2/2] Running aab_release smoke (sanity target)..."
  BUILD_KIND=aab_release ARTIFACT_DIR="$ARTIFACT_DIR/aab_release" DEVICE_ID="$DEVICE_ID" UNINSTALL_CONFLICTING=false "$SCRIPT_DIR/run_smoke.sh" || AAB_SKIPPED="failed"
else
  echo "[2/2] Skipping aab_release smoke (bundletool not found: ${BUNDLETOOL_JAR:-/tmp/bundletool-all-1.17.1.jar})"
  AAB_SKIPPED="no_bundletool"
fi

{
  echo "PRIORITY_REGRESSION=PASS"
  echo "MAIN_TARGET=standard_debug"
  echo "SECONDARY_TARGET=aab_release"
  echo "AAB_SKIPPED=$AAB_SKIPPED"
  echo "STANDARD_SUMMARY=$ARTIFACT_DIR/standard_debug/smoke_summary.txt"
  echo "AAB_SUMMARY=$ARTIFACT_DIR/aab_release/smoke_summary.txt"
  echo "QWEN35_CASES_REQUIRED=download,run,benchmark"
} >"$ARTIFACT_DIR/priority_regression_summary.txt"

cat "$ARTIFACT_DIR/priority_regression_summary.txt"
