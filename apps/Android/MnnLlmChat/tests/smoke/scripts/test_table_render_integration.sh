#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EXTENDED_SCRIPT="$SCRIPT_DIR/run_extended_priority_regression.sh"
REPORT_SCRIPT="$SCRIPT_DIR/07_generate_report.sh"
TABLE_SCRIPT="$SCRIPT_DIR/14_regress_streaming_table.sh"

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

[ -f "$TABLE_SCRIPT" ] || fail "missing table smoke script: $TABLE_SCRIPT"

rg -q 'tab_local_models' "$TABLE_SCRIPT" \
  || fail "missing tab_local_models id path in table smoke script"
rg -q 'recycler_item_model_parent' "$TABLE_SCRIPT" \
  || fail "missing recycler_item_model_parent id path in table smoke script"

rg -q 'RUN_TABLE_RENDER_SMOKE="\$\{RUN_TABLE_RENDER_SMOKE:-false\}"' "$EXTENDED_SCRIPT" \
  || fail "missing RUN_TABLE_RENDER_SMOKE env var in extended regression"
rg -q 'STEP_TABLE' "$EXTENDED_SCRIPT" \
  || fail "missing STEP_TABLE in extended regression"
rg -q '14_regress_streaming_table\.sh' "$EXTENDED_SCRIPT" \
  || fail "missing table smoke step in extended regression"
rg -q 'table_render_optional' "$EXTENDED_SCRIPT" \
  || fail "missing table_render_optional coverage token"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

mkdir -p "$TMP_DIR/table_io/shots"
touch "$TMP_DIR/table_io/shots/06_finished.png"
cat >"$TMP_DIR/table_io/summary.txt" <<'EOF'
TABLE_RENDER_TEST=PASS
SCREENSHOTS=/tmp/table_io/shots
UI_ACTION_LOG=/tmp/table_io/ui_actions.log
NOTE=Check screenshots for markdown table rendering.
EOF

ARTIFACT_DIR="$TMP_DIR" "$REPORT_SCRIPT" >/dev/null

REPORT_PATH="$TMP_DIR/report.html"
[ -f "$REPORT_PATH" ] || fail "report not generated"
rg -q 'Table rendering smoke' "$REPORT_PATH" \
  || fail "report missing table rendering stage"
rg -q 'table_io/summary.txt' "$REPORT_PATH" \
  || fail "report missing table summary path"
rg -q 'table_io/shots/06_finished.png' "$REPORT_PATH" \
  || fail "report missing table screenshot path"

echo "PASS: table render integration"
