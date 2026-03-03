#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACT_DIR="${ARTIFACT_DIR:-$SMOKE_DIR/artifacts}"
mkdir -p "$ARTIFACT_DIR"

START_TS="$(date '+%Y-%m-%d %H:%M:%S')"

"$SCRIPT_DIR/01_env_check.sh"
# shellcheck disable=SC1090
source "$ARTIFACT_DIR/smoke_env.txt"
"$SCRIPT_DIR/02_install_and_launch.sh"
"$SCRIPT_DIR/03_capture_ui_state.sh"

END_TS="$(date '+%Y-%m-%d %H:%M:%S')"

{
  echo "SMOKE_TEST=PASS"
  echo "BUILD_KIND=$BUILD_KIND"
  echo "PACKAGE_NAME=$PACKAGE_NAME"
  echo "START_TIME=$START_TS"
  echo "END_TIME=$END_TS"
  echo "ARTIFACT_DIR=$ARTIFACT_DIR"
  echo "KEY_FILES:"
  echo "  - $ARTIFACT_DIR/package_info.txt"
  echo "  - $ARTIFACT_DIR/launch_result.txt"
  echo "  - $ARTIFACT_DIR/window_dump.txt"
  echo "  - $ARTIFACT_DIR/ui_dump.xml"
  echo "  - $ARTIFACT_DIR/main_screenshot.png"
} >"$ARTIFACT_DIR/smoke_summary.txt"

cat "$ARTIFACT_DIR/smoke_summary.txt"
