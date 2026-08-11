#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BISECT_SCRIPT="$SMOKE_DIR/scripts/noui/17_bisect_mnn_tts_lib.sh"

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

[ -f "$BISECT_SCRIPT" ] || fail "missing bisect script: $BISECT_SCRIPT"

rg -q 'MAX_WAIT_SECONDS="\$\{MAX_WAIT_SECONDS:-' "$BISECT_SCRIPT" \
  || fail "missing configurable max wait"
rg -q 'POLL_INTERVAL_SECONDS="\$\{POLL_INTERVAL_SECONDS:-' "$BISECT_SCRIPT" \
  || fail "missing configurable poll interval"
rg -q 'for \(\( elapsed=0; elapsed<MAX_WAIT_SECONDS; elapsed\+=POLL_INTERVAL_SECONDS \)\)' "$BISECT_SCRIPT" \
  || fail "missing polling loop"
rg -q 'adb -s "\$DEVICE_ID" logcat -d' "$BISECT_SCRIPT" \
  || fail "missing repeated logcat polling"
rg -q 'TTS_AUDIO_SHA256=' "$BISECT_SCRIPT" \
  || fail "missing audio hash parsing"
rg -q 'sleep "\$POLL_INTERVAL_SECONDS"' "$BISECT_SCRIPT" \
  || fail "missing poll sleep"

if rg -q 'sleep 12' "$BISECT_SCRIPT"; then
  fail "fixed sleep still present"
fi

echo "PASS: mnn tts bisect integration"
