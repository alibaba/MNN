#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACT_DIR="${ARTIFACT_DIR:-$SMOKE_DIR/artifacts}"
PACKAGE_NAME="${PACKAGE_NAME:-com.alibaba.mnnllm.android}"
QUERY="$SCRIPT_DIR/ui_xml_query.py"

OUT_DIR="$ARTIFACT_DIR/sana_diffusion_ui"
SHOT_DIR="$OUT_DIR/shots"
UI_LOG="$OUT_DIR/ui_actions.log"
SUMMARY_FILE="$OUT_DIR/summary.txt"
TMP_XML="/tmp/mnn_sana_diffusion_ui.xml"

DIFFUSION_UI_KEYWORDS="${DIFFUSION_UI_KEYWORDS:-stable-diffusion,diffusion}"
SANA_UI_KEYWORDS="${SANA_UI_KEYWORDS:-sana}"
MODEL_SEARCH_MAX_TRIES="${MODEL_SEARCH_MAX_TRIES:-8}"

mkdir -p "$OUT_DIR" "$SHOT_DIR"
rm -f "$SHOT_DIR"/*.png "$UI_LOG" "$SUMMARY_FILE"
: >"$UI_LOG"

log_line() {
  echo "$1 ts=$(date '+%H:%M:%S')" | tee -a "$UI_LOG"
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
}

dump_ui() {
  local out="$1"
  local try
  for try in 1 2 3; do
    adb shell uiautomator dump /sdcard/mnn_sana_diffusion_ui.xml >/dev/null 2>/dev/null || true
    adb pull /sdcard/mnn_sana_diffusion_ui.xml "$out" >/dev/null 2>/dev/null || true
    if [ -s "$out" ] && rg -q "<hierarchy" "$out"; then
      if rg -q "package=\"com\\.lbe\\.security\\.miui\"" "$out"; then
        log_line "MIUI_PERMISSION_OVERLAY try=$try"
        tap_allow_from_overlay "$out" || true
        sleep 1
        continue
      fi
      if rg -q "package=\"com\\.alibaba\\.mnnllm\\.android(\\.release)?\"" "$out"; then
        return 0
      fi
    fi
    ensure_device_unlocked
    adb shell monkey -p "$PACKAGE_NAME" -c android.intent.category.LAUNCHER 1 >/dev/null 2>&1 || true
    sleep 1
  done
  return 1
}

tap_by_any_contains_text() {
  local xml="$1"
  shift
  local text
  for text in "$@"; do
    if python3 "$QUERY" --xml "$xml" --contains-text "$text" >/tmp/mnn_sana_query_hit.txt 2>/dev/null; then
      read -r x y _ </tmp/mnn_sana_query_hit.txt
      log_line "TAP contains=$text x=$x y=$y"
      adb shell input tap "$x" "$y"
      return 0
    fi
  done
  return 1
}

tap_allow_from_overlay() {
  local xml="$1"
  tap_by_any_contains_text "$xml" "始终允许" "仅在使用中允许" "允许" "同意"
}

tap_by_any_rid() {
  local xml="$1"
  shift
  local rid
  for rid in "$@"; do
    if python3 "$QUERY" --xml "$xml" --resource-id "$rid" >/tmp/mnn_sana_query_hit.txt 2>/dev/null; then
      read -r x y _ </tmp/mnn_sana_query_hit.txt
      log_line "TAP rid=$rid x=$x y=$y"
      adb shell input tap "$x" "$y"
      return 0
    fi
  done
  return 1
}

ensure_market_tab() {
  local try
  for try in 1 2 3 4 5; do
    dump_ui "$TMP_XML" || true
    if tap_by_any_rid "$TMP_XML" \
      "com.alibaba.mnnllm.android:id/tab_model_market" \
      "com.alibaba.mnnllm.android.release:id/tab_model_market"; then
      sleep 2
      dump_ui "$TMP_XML" || true
      return 0
    fi
    adb shell input keyevent 4 >/dev/null 2>&1 || true
    sleep 1
  done
  return 1
}

find_model_action_for_keywords() {
  local xml="$1"
  local keywords_csv="$2"
  python3 - "$xml" "$keywords_csv" <<'PY'
import re
import sys
import xml.etree.ElementTree as ET

xml_path = sys.argv[1]
keywords = [x.strip().lower() for x in sys.argv[2].split(",") if x.strip()]

def parse_bounds(raw):
    m = re.match(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", raw or "")
    if not m:
        return None
    x1, y1, x2, y2 = map(int, m.groups())
    return x1, y1, x2, y2

def center(bounds):
    x1, y1, x2, y2 = bounds
    return (x1 + x2) // 2, (y1 + y2) // 2

root = ET.parse(xml_path).getroot()
titles = []
buttons = []

for node in root.iter("node"):
    rid = node.attrib.get("resource-id", "")
    text = (node.attrib.get("text") or "").strip()
    bounds = parse_bounds(node.attrib.get("bounds", ""))
    if not bounds:
        continue
    if rid.endswith(":id/tvModelTitle"):
        lower = text.lower()
        if keywords and not any(k in lower for k in keywords):
            continue
        cx, cy = center(bounds)
        titles.append((text, cx, cy))
    elif rid.endswith(":id/btn_download_action"):
        cx, cy = center(bounds)
        buttons.append((text, cx, cy))

if not titles or not buttons:
    sys.exit(2)

title_text, _, title_y = titles[0]
best = min(buttons, key=lambda item: abs(item[2] - title_y))
btn_text, btn_x, btn_y = best
delta_y = abs(btn_y - title_y)
print(f"{btn_x}|{btn_y}|{btn_text}|{title_text}|{delta_y}")
PY
}

ensure_selected_model_matches() {
  local xml="$1"
  local keywords_csv="$2"
  python3 - "$xml" "$keywords_csv" <<'PY'
import sys
import xml.etree.ElementTree as ET

xml_path = sys.argv[1]
keywords = [x.strip().lower() for x in sys.argv[2].split(",") if x.strip()]
root = ET.parse(xml_path).getroot()

for node in root.iter("node"):
    rid = node.attrib.get("resource-id", "")
    text = (node.attrib.get("text") or "").strip().lower()
    if rid.endswith(":id/tv_selected_model") and any(k in text for k in keywords):
        sys.exit(0)
sys.exit(2)
PY
}

enter_model_chat_from_market() {
  local case_name="$1"
  local keywords_csv="$2"
  local shot_prefix="$3"
  local line=""
  local try

  ensure_market_tab || return 1
  shot "${shot_prefix}_01_market_entry"

  for try in $(seq 1 "$MODEL_SEARCH_MAX_TRIES"); do
    dump_ui "$TMP_XML" || true
    cp "$TMP_XML" "$OUT_DIR/${case_name}_market_try${try}.xml" || true
    line="$(find_model_action_for_keywords "$TMP_XML" "$keywords_csv" || true)"
    if [ -n "$line" ]; then
      break
    fi
    log_line "${case_name}_MODEL_NOT_VISIBLE try=$try swipe_up"
    adb shell input swipe 540 1750 540 900 260 >/dev/null 2>&1 || true
    sleep 1
  done

  if [ -z "$line" ]; then
    log_line "${case_name}_MODEL_NOT_FOUND keywords=$keywords_csv"
    return 1
  fi

  IFS='|' read -r btn_x btn_y btn_text model_text delta_y <<<"$line"
  if [ -z "${btn_x:-}" ] || [ -z "${btn_y:-}" ]; then
    log_line "${case_name}_ACTION_PARSE_FAIL line=$line"
    return 1
  fi
  if [ "${delta_y:-9999}" -gt 260 ]; then
    log_line "${case_name}_ACTION_TOO_FAR delta_y=$delta_y model=$model_text button=$btn_text"
    return 1
  fi

  log_line "${case_name}_TAP_ACTION model=$model_text button_text=$btn_text x=$btn_x y=$btn_y"
  adb shell input tap "$btn_x" "$btn_y"
  sleep 3
  dump_ui "$TMP_XML" || true
  cp "$TMP_XML" "$OUT_DIR/${case_name}_chat.xml" || true
  shot "${shot_prefix}_02_after_action"

  if ! echo "$btn_text" | rg -qi "对话|chat"; then
    log_line "${case_name}_ACTION_NOT_CHAT button_text=$btn_text"
    return 1
  fi

  if ! ensure_selected_model_matches "$TMP_XML" "$keywords_csv"; then
    log_line "${case_name}_MODEL_SWITCH_NOT_CONFIRMED keywords=$keywords_csv"
    return 1
  fi

  return 0
}

validate_diffusion_chat_controls() {
  local xml="$1"
  python3 "$QUERY" --xml "$xml" --resource-id "com.alibaba.mnnllm.android:id/et_message" >/dev/null 2>&1 \
    || python3 "$QUERY" --xml "$xml" --resource-id "com.alibaba.mnnllm.android.release:id/et_message" >/dev/null 2>&1
  python3 "$QUERY" --xml "$xml" --resource-id "com.alibaba.mnnllm.android:id/btn_send" >/dev/null 2>&1 \
    || python3 "$QUERY" --xml "$xml" --resource-id "com.alibaba.mnnllm.android.release:id/btn_send" >/dev/null 2>&1
}

validate_sana_face_hint() {
  local xml="$1"
  if rg -qi "人脸图片|face image" "$xml"; then
    return 0
  fi
  validate_diffusion_chat_controls "$xml"
}

send_diffusion_smoke_prompt() {
  local xml="$1"
  tap_by_any_rid "$xml" \
    "com.alibaba.mnnllm.android:id/et_message" \
    "com.alibaba.mnnllm.android.release:id/et_message" || return 1
  adb shell input text "smoke_diffusion_case"
  sleep 1
  dump_ui "$TMP_XML" || true
  shot "diffusion_03_after_text"
  tap_by_any_rid "$TMP_XML" \
    "com.alibaba.mnnllm.android:id/btn_send" \
    "com.alibaba.mnnllm.android.release:id/btn_send" || return 1
  sleep 2
  dump_ui "$TMP_XML" || true
  cp "$TMP_XML" "$OUT_DIR/diffusion_after_send.xml" || true
  shot "diffusion_04_after_send"
}

ensure_device_unlocked
adb shell monkey -p "$PACKAGE_NAME" -c android.intent.category.LAUNCHER 1 >/dev/null 2>&1 || true
sleep 2

diffusion_status="FAIL"
sana_status="FAIL"
overall="FAIL"

if enter_model_chat_from_market "diffusion" "$DIFFUSION_UI_KEYWORDS" "diffusion" \
  && validate_diffusion_chat_controls "$TMP_XML" \
  && send_diffusion_smoke_prompt "$TMP_XML"; then
  diffusion_status="PASS"
else
  log_line "DIFFUSION_UI_CASE_FAIL"
fi

adb shell am force-stop "$PACKAGE_NAME" >/dev/null 2>&1 || true
sleep 1
ensure_device_unlocked
adb shell monkey -p "$PACKAGE_NAME" -c android.intent.category.LAUNCHER 1 >/dev/null 2>&1 || true
sleep 2

if enter_model_chat_from_market "sana" "$SANA_UI_KEYWORDS" "sana" \
  && validate_sana_face_hint "$TMP_XML"; then
  sana_status="PASS"
else
  log_line "SANA_UI_CASE_FAIL"
fi

if [ "$diffusion_status" = "PASS" ] && [ "$sana_status" = "PASS" ]; then
  overall="PASS"
fi

{
  echo "SANA_DIFFUSION_UIAUTOMATOR=$overall"
  echo "DIFFUSION_UI_CASE=$diffusion_status"
  echo "SANA_UI_CASE=$sana_status"
  echo "DIFFUSION_UI_KEYWORDS=$DIFFUSION_UI_KEYWORDS"
  echo "SANA_UI_KEYWORDS=$SANA_UI_KEYWORDS"
  echo "SCREENSHOTS=$SHOT_DIR"
  echo "UI_ACTION_LOG=$UI_LOG"
} >"$SUMMARY_FILE"

cat "$SUMMARY_FILE"

if [ "$overall" != "PASS" ]; then
  exit 1
fi
