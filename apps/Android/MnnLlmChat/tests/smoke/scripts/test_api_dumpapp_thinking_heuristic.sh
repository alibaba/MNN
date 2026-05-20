#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_SCRIPT="$SCRIPT_DIR/08_regress_api_dumpapp.sh"

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

[ -f "$TARGET_SCRIPT" ] || fail "missing target script: $TARGET_SCRIPT"

TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

export ARTIFACT_DIR="$TMP_DIR/artifacts"

restore_openai_state() {
  :
}

cat >"$TMP_DIR/on.json" <<'EOF'
{"choices":[{"message":{"content":"Reasoning on response with more detail but no explicit think tags."}}]}
EOF

cat >"$TMP_DIR/off.json" <<'EOF'
{"choices":[{"message":{"content":"Reasoning off response with different final wording and no explicit think tags."}}]}
EOF

source <(awk '
  /^source "\$SCRIPT_DIR\/smoke_state_helpers\.sh"$/ { next }
  /^snapshot_openai_state / { exit }
  { print }
' "$TARGET_SCRIPT")

if ! thinking_response_diverged "$TMP_DIR/on.json" "$TMP_DIR/off.json"; then
  fail "expected thinking_response_diverged to pass when on/off contents differ without think tags"
fi

cat >"$TMP_DIR/same.json" <<'EOF'
{"choices":[{"message":{"content":"same content"}}]}
EOF

if thinking_response_diverged "$TMP_DIR/same.json" "$TMP_DIR/same.json"; then
  fail "expected thinking_response_diverged to fail when contents are identical"
fi

echo "PASS: api dumpapp thinking heuristic"
