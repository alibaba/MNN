#!/usr/bin/env bash
set -euo pipefail

# Voice Chat UI Smoke Test
#
# This script tests the Voice Chat UI flow:
# - Enter chat activity
# - Open Voice Chat from more menu
# - Verify voice chat UI elements appear
# - Capture status transitions (CONNECTING → LISTENING → etc)
# - End voice chat session
#
# Prerequisites:
# - TTS and ASR models downloaded and set as default
# - App running with a downloaded LLM model available for chat
#
# Usage:
#   ./scripts/16_regress_voice_ui.sh
#
# Environment Variables:
#   VOICE_UI_WAIT_SEC  - Wait time for voice chat to initialize (default: 10)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACT_DIR="${ARTIFACT_DIR:-$SMOKE_DIR/artifacts}"
OUT_DIR="$ARTIFACT_DIR/voice_ui"
SHOT_DIR="$OUT_DIR/shots"
mkdir -p "$OUT_DIR" "$SHOT_DIR"

QUERY="$SCRIPT_DIR/ui_xml_query.py"
PACKAGE_NAME="${PACKAGE_NAME:-com.alibaba.mnnllm.android}"
TMP_XML="/tmp/mnn_voice_ui.xml"
UI_LOG="$OUT_DIR/ui_actions.log"
SUMMARY_FILE="$OUT_DIR/summary.txt"

VOICE_UI_WAIT_SEC="${VOICE_UI_WAIT_SEC:-10}"

# Initialize log file
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
    adb shell uiautomator dump /sdcard/mnn_voice_ui.xml >/dev/null 2>/dev/null || true
    adb pull /sdcard/mnn_voice_ui.xml "$out" >/dev/null 2>/dev/null || true
    if [ -s "$out" ] && rg -q "<hierarchy" "$out"; then
      # Handle MIUI permission overlay
      if rg -q "package=\"com\\.lbe\\.security\\.miui\"" "$out"; then
        log "MIUI_PERMISSION_OVERLAY try=$try"
        tap_by_any_contains_text "$out" "始终允许" "仅在使用中允许" "允许" "同意" || true
        sleep 1
        continue
      fi
      # Handle audio permission dialog
      if rg -q "允许.*录音\|allow.*record\|RECORD_AUDIO" "$out"; then
        log "AUDIO_PERMISSION_DIALOG try=$try"
        tap_by_any_contains_text "$out" "允许" "Allow" "始终允许" "While using the app" || true
        sleep 1
        continue
      fi
      if rg -q "package=\"com\\.alibaba\\.mnnllm\\.android" "$out"; then
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

tap_by_any_rid() {
  local xml="$1"
  shift
  local rid
  for rid in "$@"; do
    if python3 "$QUERY" --xml "$xml" --resource-id "$rid" >/tmp/mnn_query_hit.txt 2>/dev/null; then
      read -r x y _ </tmp/mnn_query_hit.txt
      log "TAP rid=$rid x=$x y=$y"
      adb shell input tap "$x" "$y"
      return 0
    fi
  done
  log "TAP_NOT_FOUND candidates=$*"
  return 1
}

tap_by_any_contains_text() {
  local xml="$1"
  shift
  local text
  for text in "$@"; do
    if python3 "$QUERY" --xml "$xml" --contains-text "$text" >/tmp/mnn_query_hit.txt 2>/dev/null; then
      read -r x y _ </tmp/mnn_query_hit.txt
      log "TAP contains=$text x=$x y=$y"
      adb shell input tap "$x" "$y"
      return 0
    fi
  done
  return 1
}

open_overflow_menu() {
  adb shell input keyevent 82 >/dev/null 2>&1 || true
  sleep 1
  dump_ui "$TMP_XML" || true
}

tap_by_rid_text() {
  local xml="$1"
  local text="$2"
  shift 2
  local rid
  for rid in "$@"; do
    if python3 "$QUERY" --xml "$xml" --resource-id "$rid" --text "$text" >/tmp/mnn_query_hit.txt 2>/dev/null; then
      read -r x y _ </tmp/mnn_query_hit.txt
      log "TAP rid=$rid text=$text x=$x y=$y"
      adb shell input tap "$x" "$y"
      return 0
    fi
  done
  return 1
}

wait_for_chat_ready() {
  local attempt
  for attempt in 1 2 3 4 5 6; do
    sleep 2
    dump_ui "$TMP_XML" || true
    if exists_any_rid "$TMP_XML" \
      "com.alibaba.mnnllm.android:id/et_message" \
      "com.alibaba.mnnllm.android.release:id/et_message" \
      "com.alibaba.mnnllm.android:id/more_item_voice_chat" \
      "com.alibaba.mnnllm.android.release:id/more_item_voice_chat"; then
      log "CHAT_READY attempt=$attempt"
      return 0
    fi
    log "CHAT_READY_RETRY attempt=$attempt"
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

exists_any_text() {
  local xml="$1"
  shift
  local text
  for text in "$@"; do
    if python3 "$QUERY" --xml "$xml" --contains-text "$text" >/dev/null 2>&1; then
      return 0
    fi
  done
  return 1
}

get_voice_chat_status() {
  local xml="$1"
  # Check for various status texts in the voice chat UI
  if exists_any_text "$xml" "Listening" "正在聆听"; then
    echo "LISTENING"
  elif exists_any_text "$xml" "Speaking" "正在说话"; then
    echo "SPEAKING"
  elif exists_any_text "$xml" "Processing" "处理中"; then
    echo "PROCESSING"
  elif exists_any_text "$xml" "Thinking" "思考中"; then
    echo "THINKING"
  elif exists_any_text "$xml" "Connecting" "正在连接"; then
    echo "CONNECTING"
  elif exists_any_text "$xml" "Initializing" "初始化"; then
    echo "INITIALIZING"
  elif exists_any_text "$xml" "Stop" "停止"; then
    echo "ACTIVE"
  elif exists_any_text "$xml" "What can I help" "有什么可以帮助"; then
    echo "READY"
  else
    echo "UNKNOWN"
  fi
}

# Status tracking
enter_chat_status="FAIL"
open_voice_chat_status="FAIL"
voice_chat_initialized="FAIL"
voice_chat_listening="FAIL"
end_voice_chat_status="FAIL"
voice_status_seen=""

log "=== Voice Chat UI Smoke Test ==="
ensure_device_unlocked

# Step 1: Launch app and enter chat
log "Step 1: Launching app and entering chat..."
adb shell am force-stop "$PACKAGE_NAME"
sleep 1
adb shell monkey -p "$PACKAGE_NAME" -c android.intent.category.LAUNCHER 1 >/dev/null
sleep 2
dump_ui "$TMP_XML"
shot "01_after_launch"

# Navigate to chat - try model market tab first, then find a chat entry
if tap_by_any_rid "$TMP_XML" \
  "com.alibaba.mnnllm.android:id/tab_model_market" \
  "com.alibaba.mnnllm.android.release:id/tab_model_market"; then
  sleep 2
  dump_ui "$TMP_XML"
fi
shot "02_model_market"

# Try to enter chat via "对话" button or chat tab
if tap_by_rid_text "$TMP_XML" "对话" \
  "com.alibaba.mnnllm.android:id/btn_download_action" \
  "com.alibaba.mnnllm.android.release:id/btn_download_action" \
  || tap_by_any_contains_text "$TMP_XML" "对话" "Chat"; then
  wait_for_chat_ready || true
elif exists_any_rid "$TMP_XML" \
  "com.alibaba.mnnllm.android:id/et_message" \
  "com.alibaba.mnnllm.android.release:id/et_message"; then
  log "Already in chat activity"
elif tap_by_any_rid "$TMP_XML" \
  "com.alibaba.mnnllm.android:id/tab_chat" \
  "com.alibaba.mnnllm.android.release:id/tab_chat"; then
  wait_for_chat_ready || true
fi

# Verify we're in chat
if exists_any_rid "$TMP_XML" \
  "com.alibaba.mnnllm.android:id/et_message" \
  "com.alibaba.mnnllm.android.release:id/et_message" \
  "com.alibaba.mnnllm.android:id/more_item_voice_chat" \
  "com.alibaba.mnnllm.android.release:id/more_item_voice_chat"; then
  enter_chat_status="PASS"
  log "Successfully entered chat activity"
else
  log "Failed to enter chat activity"
fi
cp "$TMP_XML" "$OUT_DIR/chat_entered.xml"
shot "03_chat_entered"

# Step 2: Open Voice Chat
if [ "$enter_chat_status" = "PASS" ]; then
  log "Step 2: Opening Voice Chat..."
  
  # First try the more menu voice chat button
  if tap_by_any_rid "$TMP_XML" \
    "com.alibaba.mnnllm.android:id/more_item_voice_chat" \
    "com.alibaba.mnnllm.android.release:id/more_item_voice_chat" \
    "com.alibaba.mnnllm.android:id/start_voice_chat" \
    "com.alibaba.mnnllm.android.release:id/start_voice_chat"; then
    sleep 2
    dump_ui "$TMP_XML"
    open_voice_chat_status="PASS"
  # Then try the action menu item text from overflow / hardware menu.
  elif open_overflow_menu && tap_by_any_contains_text "$TMP_XML" \
    "Start Voice Chat" "开启语音会话" "Voice Chat" "语音会话" "语音聊天"; then
    sleep 2
    dump_ui "$TMP_XML"
    open_voice_chat_status="PASS"
  # Fallback: some builds attach action items to toolbar without exposing resource IDs in dump.
  elif tap_by_any_rid "$TMP_XML" \
    "com.alibaba.mnnllm.android:id/toolbar" \
    "com.alibaba.mnnllm.android.release:id/toolbar"; then
    sleep 1
    open_overflow_menu
    if tap_by_any_contains_text "$TMP_XML" \
      "Start Voice Chat" "开启语音会话" "Voice Chat" "语音会话" "语音聊天"; then
      sleep 2
      dump_ui "$TMP_XML"
      open_voice_chat_status="PASS"
    fi
  fi
  
  shot "04_voice_chat_opening"
  cp "$TMP_XML" "$OUT_DIR/voice_chat_opening.xml"
fi

# Step 3: Wait for Voice Chat to initialize and verify states
if [ "$open_voice_chat_status" = "PASS" ]; then
  log "Step 3: Waiting for Voice Chat to initialize..."
  
  # Handle possible permission dialogs
  for _ in 1 2 3; do
    dump_ui "$TMP_XML"
    if rg -q "允许\|Allow\|permission" "$TMP_XML"; then
      tap_by_any_contains_text "$TMP_XML" "允许" "Allow" "始终允许" "While using" || true
      sleep 1
    else
      break
    fi
  done
  
  # Check for voice chat UI elements
  wait_count=0
  while [ "$wait_count" -lt "$VOICE_UI_WAIT_SEC" ]; do
    dump_ui "$TMP_XML"
    current_status=$(get_voice_chat_status "$TMP_XML")
    
    if [ -n "$current_status" ] && [ "$current_status" != "UNKNOWN" ]; then
      voice_status_seen="${voice_status_seen}${current_status},"
      log "Voice Chat status: $current_status"
      
      if [ "$current_status" = "LISTENING" ] || [ "$current_status" = "READY" ]; then
        voice_chat_listening="PASS"
        voice_chat_initialized="PASS"
        break
      elif [ "$current_status" = "CONNECTING" ] || [ "$current_status" = "INITIALIZING" ]; then
        voice_chat_initialized="PASS"
      fi
    fi
    
    # Check for end call button as sign voice chat is active
    if exists_any_rid "$TMP_XML" \
      "com.alibaba.mnnllm.android:id/button_end_call" \
      "com.alibaba.mnnllm.android.release:id/button_end_call"; then
      voice_chat_initialized="PASS"
      log "Voice Chat UI detected (end call button found)"
    fi
    
    # Check for voice transcript recycler
    if exists_any_rid "$TMP_XML" \
      "com.alibaba.mnnllm.android:id/rv_voice_transcript" \
      "com.alibaba.mnnllm.android.release:id/rv_voice_transcript"; then
      voice_chat_initialized="PASS"
      log "Voice Chat transcript area found"
    fi
    
    shot "05_voice_chat_state_${wait_count}"
    sleep 1
    wait_count=$((wait_count + 1))
  done
  
  cp "$TMP_XML" "$OUT_DIR/voice_chat_active.xml"
  shot "06_voice_chat_final_state"
fi

# Step 4: End Voice Chat
if [ "$voice_chat_initialized" = "PASS" ]; then
  log "Step 4: Ending Voice Chat session..."
  
  dump_ui "$TMP_XML"
  
  if tap_by_any_rid "$TMP_XML" \
    "com.alibaba.mnnllm.android:id/button_end_call" \
    "com.alibaba.mnnllm.android.release:id/button_end_call"; then
    sleep 2
    dump_ui "$TMP_XML"
    
    # Verify we're back in chat (not in voice chat anymore)
    if exists_any_rid "$TMP_XML" \
      "com.alibaba.mnnllm.android:id/et_message" \
      "com.alibaba.mnnllm.android.release:id/et_message"; then
      end_voice_chat_status="PASS"
      log "Successfully ended voice chat and returned to chat"
    else
      log "End call pressed but chat UI not detected"
    fi
  # Also try back button or toolbar navigation
  elif tap_by_any_contains_text "$TMP_XML" "End Call" "结束通话"; then
    sleep 2
    end_voice_chat_status="PASS"
  else
    log "End call button not found, trying back press"
    adb shell input keyevent 4
    sleep 2
    dump_ui "$TMP_XML"
    if exists_any_rid "$TMP_XML" \
      "com.alibaba.mnnllm.android:id/et_message" \
      "com.alibaba.mnnllm.android.release:id/et_message"; then
      end_voice_chat_status="PASS"
    fi
  fi
  
  shot "07_after_end_voice_chat"
  cp "$TMP_XML" "$OUT_DIR/after_end_voice_chat.xml"
fi

# Determine overall status
overall_status="FAIL"
if [ "$enter_chat_status" = "PASS" ] && \
   [ "$open_voice_chat_status" = "PASS" ] && \
   [ "$voice_chat_initialized" = "PASS" ]; then
  overall_status="PASS"
fi

# Write summary
{
  echo "VOICE_UI_SMOKE=$overall_status"
  echo "ENTER_CHAT_STATUS=$enter_chat_status"
  echo "OPEN_VOICE_CHAT_STATUS=$open_voice_chat_status"
  echo "VOICE_CHAT_INITIALIZED=$voice_chat_initialized"
  echo "VOICE_CHAT_LISTENING=$voice_chat_listening"
  echo "END_VOICE_CHAT_STATUS=$end_voice_chat_status"
  echo "VOICE_STATUS_SEEN=${voice_status_seen:-none}"
  echo "SHOT_DIR=$SHOT_DIR"
  echo "ARTIFACT_DIR=$OUT_DIR"
} >"$SUMMARY_FILE"

log "=== Voice Chat UI Smoke Test Summary ==="
cat "$SUMMARY_FILE"

if [ "$overall_status" != "PASS" ]; then
  exit 1
fi
