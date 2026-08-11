#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACT_DIR="${ARTIFACT_DIR:-$SMOKE_DIR/artifacts}"
OUT_DIR="$ARTIFACT_DIR/fullscreen_swipe_ui"
SHOT_DIR="$OUT_DIR/shots"
mkdir -p "$OUT_DIR" "$SHOT_DIR"

QUERY="$SCRIPT_DIR/ui_xml_query.py"
PACKAGE_NAME="${PACKAGE_NAME:-com.alibaba.mnnllm.android}"
TMP_XML="/tmp/mnn_fullscreen_swipe.xml"
UI_LOG="$OUT_DIR/ui_actions.log"
SUMMARY_FILE="$OUT_DIR/summary.txt"

: >"$UI_LOG"

log() {
  echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$UI_LOG"
}

ensure_device_unlocked() {
  adb shell input keyevent 224 >/dev/null 2>&1 || true
  adb shell wm dismiss-keyguard >/dev/null 2>&1 || true
  adb shell input keyevent 82 >/dev/null 2>&1 || true
  adb shell input swipe 540 2000 540 700 200 >/dev/null 2>&1 || true
  sleep 1
}

shot() {
  local name="$1"
  adb exec-out screencap -p >"$SHOT_DIR/${name}.png"
  log "SCREENSHOT: ${name}.png"
}

dump_ui() {
  local out="$1"
  local try
  for try in 1 2 3; do
    adb shell uiautomator dump /sdcard/mnn_fullscreen_swipe.xml >/dev/null 2>/dev/null || true
    adb pull /sdcard/mnn_fullscreen_swipe.xml "$out" >/dev/null 2>/dev/null || true
    if [ -s "$out" ] && rg -q "<hierarchy" "$out"; then
      if rg -q "package=\"com\\.lbe\\.security\\.miui\"" "$out"; then
        log "MIUI_PERMISSION_OVERLAY try=$try"
        if python3 "$QUERY" --xml "$out" --contains-text "允许" >/tmp/mnn_query_hit.txt 2>/dev/null; then
          read -r x y _ </tmp/mnn_query_hit.txt
          adb shell input tap "$x" "$y"
        fi
        sleep 1
        continue
      fi
      if rg -q "package=\"com\\.alibaba\\.mnnllm\\.android(\\.release)?\"" "$out"; then
        return 0
      fi
    fi
    log "UI_DUMP_RETRY try=$try"
    ensure_device_unlocked
    adb shell monkey -p "$PACKAGE_NAME" -c android.intent.category.LAUNCHER 1 >/dev/null 2>&1 || true
    sleep 1
  done
  return 1
}

exists_any_rid() {
  local xml="$1"
  shift
  local rid
  for rid in "$@"; do
    if python3 "$QUERY" --xml "$xml" --resource-id "$rid" >/dev/null 2>&1; then
      return 0
    fi
  done
  return 1
}

ensure_device_unlocked
adb shell am force-stop "$PACKAGE_NAME"
sleep 1
adb shell monkey -p "$PACKAGE_NAME" -c android.intent.category.LAUNCHER 1 >/dev/null
sleep 3
dump_ui "$TMP_XML"
shot "01_after_launch"

if ! exists_any_rid "$TMP_XML" \
  "com.alibaba.mnnllm.android:id/drawer_layout" \
  "com.alibaba.mnnllm.android.release:id/drawer_layout"; then
  echo "DRAWER_LAYOUT_NOT_FOUND" >&2
  exit 1
fi

size="$(adb shell wm size 2>/dev/null | tr -d '\r' | rg -o '[0-9]+x[0-9]+' | head -n1)"
width="${size%x*}"
height="${size#*x}"
start_x=$((width * 20 / 100))
end_x=$((width * 80 / 100))
mid_y=$((height / 2))
log "SWIPE open start=($start_x,$mid_y) end=($end_x,$mid_y)"
adb shell input swipe "$start_x" "$mid_y" "$end_x" "$mid_y" 200
sleep 2
dump_ui "$TMP_XML"
cp "$TMP_XML" "$OUT_DIR/after_open.xml"
shot "02_after_open"

if ! exists_any_rid "$TMP_XML" \
  "com.alibaba.mnnllm.android:id/nav_view" \
  "com.alibaba.mnnllm.android.release:id/nav_view"; then
  echo "NAV_VIEW_NOT_VISIBLE_AFTER_SWIPE" >&2
  exit 1
fi

adb shell input keyevent 4
sleep 2
dump_ui "$TMP_XML"
cp "$TMP_XML" "$OUT_DIR/after_back.xml"
shot "03_after_back"

{
  echo "FULLSCREEN_SWIPE_UI=PASS"
  echo "SWIPE_OPEN_START_X=$start_x"
  echo "SWIPE_OPEN_END_X=$end_x"
  echo "SWIPE_MID_Y=$mid_y"
  echo "SCREENSHOTS=$SHOT_DIR"
  echo "UI_ACTION_LOG=$UI_LOG"
  echo "NOTE=Validated that a mid-screen right swipe opens the drawer and back closes it."
} >"$SUMMARY_FILE"
