#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACT_DIR="${ARTIFACT_DIR:-$SMOKE_DIR/artifacts}"

rm -rf "$ARTIFACT_DIR/standard_debug" "$ARTIFACT_DIR/aab_release"
mkdir -p "$ARTIFACT_DIR/standard_debug" "$ARTIFACT_DIR/aab_release"

echo "[1/2] Running standard_debug smoke (main target)..."
BUILD_KIND=standard_debug ARTIFACT_DIR="$ARTIFACT_DIR/standard_debug" "$SCRIPT_DIR/run_smoke.sh"

echo "[2/2] Running aab_release smoke (sanity target)..."
BUILD_KIND=aab_release ARTIFACT_DIR="$ARTIFACT_DIR/aab_release" "$SCRIPT_DIR/run_smoke.sh"

{
  echo "PRIORITY_REGRESSION=PASS"
  echo "MAIN_TARGET=standard_debug"
  echo "SECONDARY_TARGET=aab_release"
  echo "STANDARD_SUMMARY=$ARTIFACT_DIR/standard_debug/smoke_summary.txt"
  echo "AAB_SUMMARY=$ARTIFACT_DIR/aab_release/smoke_summary.txt"
  echo "QWEN35_CASES_REQUIRED=download,run,benchmark"
} >"$ARTIFACT_DIR/priority_regression_summary.txt"

cat "$ARTIFACT_DIR/priority_regression_summary.txt"
