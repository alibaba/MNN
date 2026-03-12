#!/usr/bin/env bash
set -euo pipefail

# Voice Smoke Test via dumpapp (TTS/ASR capabilities)
#
# This script tests the Voice Chat capabilities without UI interaction:
# - TTS model initialization and text-to-speech generation
# - TTS -> ASR roundtrip validation for Chinese output correctness
# - Voice model path configuration
#
# Prerequisites:
# - TTS model downloaded (e.g., ModelScope/MNN/BertVITS2-MNN)
# - App running with dumpapp connectivity
#
# Usage:
#   ./scripts/noui/15_regress_voice_dumpapp.sh
#
# Environment Variables:
#   TTS_TEST_TEXT      - Text to use for TTS test (default: "你好，今天天气不错。")
#   VOICE_TIMEOUT_SEC  - Timeout for voice operations (default: 60)
#   ROUNDTRIP_MIN_SIM  - Minimum roundtrip similarity (default: 0.50)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
ARTIFACT_DIR="${ARTIFACT_DIR:-$SMOKE_DIR/artifacts}"
OUT_DIR="$ARTIFACT_DIR/voice_dumpapp"
mkdir -p "$OUT_DIR"

PACKAGE_NAME="${PACKAGE_NAME:-com.alibaba.mnnllm.android}"
DUMPAPP="${DUMPAPP:-$SMOKE_DIR/../../tools/dumpapp}"
MAIN_ACTIVITY="${MAIN_ACTIVITY:-$PACKAGE_NAME/com.alibaba.mnnllm.android.main.MainActivity}"
TTS_TEST_TEXT="${TTS_TEST_TEXT:-你好，今天天气不错。}"
VOICE_TIMEOUT_SEC="${VOICE_TIMEOUT_SEC:-60}"
ROUNDTRIP_MIN_SIM="${ROUNDTRIP_MIN_SIM:-0.50}"

SUMMARY_FILE="$OUT_DIR/summary.txt"
UI_LOG="$OUT_DIR/ui_actions.log"

# Initialize log file
: >"$UI_LOG"

log() {
  echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$UI_LOG" >&2
}

ensure_app_foreground() {
  log "Ensuring app is in foreground..."
  adb shell am start -W -n "$MAIN_ACTIVITY" >"$OUT_DIR/launch_activity.log" 2>&1 || true
  sleep 2
}

run_dumpapp_voice() {
  local subcommand="$1"
  shift
  local log_file="$OUT_DIR/voice_${subcommand// /_}.log"
  
  log "Running: dumpapp voice $subcommand $*"
  "$DUMPAPP" -p "$PACKAGE_NAME" voice $subcommand "$@" >"$log_file" 2>&1 || true
  
  # Return the log file path for callers to check
  echo "$log_file"
}

check_voice_status() {
  local log_file
  log_file=$(run_dumpapp_voice status)
  
  if rg -q "READY=true" "$log_file"; then
    log "Voice models are ready"
    return 0
  else
    log "Voice models are NOT ready"
    cat "$log_file" >> "$UI_LOG"
    return 1
  fi
}

test_tts_init() {
  local log_file
  log_file=$(run_dumpapp_voice "tts init")
  
  if rg -q "TTS_INIT=SUCCESS" "$log_file"; then
    log "TTS initialization: SUCCESS"
    local init_time
    init_time=$(rg "TTS_INIT_TIME_MS=" "$log_file" | cut -d= -f2 || echo "unknown")
    log "TTS init time: ${init_time}ms"
    return 0
  elif rg -q "TTS_INIT=ALREADY_INITIALIZED" "$log_file"; then
    log "TTS initialization: ALREADY_INITIALIZED (OK)"
    return 0
  else
    log "TTS initialization: FAIL"
    cat "$log_file" >> "$UI_LOG"
    return 1
  fi
}

test_tts_process() {
  local text="$1"
  local log_file
  log_file=$(run_dumpapp_voice "tts test" "$text")
  
  if rg -q "TTS_TEST=SUCCESS" "$log_file"; then
    log "TTS test: SUCCESS"
    
    local samples duration rtf roundtrip_text roundtrip_has_chinese roundtrip_similarity roundtrip_status
    samples=$(rg "TTS_AUDIO_SAMPLES=" "$log_file" | cut -d= -f2 || echo "0")
    duration=$(rg "TTS_AUDIO_DURATION_SEC=" "$log_file" | cut -d= -f2 || echo "0")
    rtf=$(rg "TTS_RTF=" "$log_file" | cut -d= -f2 || echo "0")
    roundtrip_text=$(rg "TTS_ROUNDTRIP_TEXT=" "$log_file" | cut -d= -f2- || echo "")
    roundtrip_has_chinese=$(rg "TTS_ROUNDTRIP_HAS_CHINESE=" "$log_file" | cut -d= -f2 || echo "false")
    roundtrip_similarity=$(rg "TTS_ROUNDTRIP_SIMILARITY=" "$log_file" | cut -d= -f2 || echo "0")
    roundtrip_status=$(rg "TTS_ROUNDTRIP_STATUS=" "$log_file" | cut -d= -f2 || echo "FAIL")
    
    log "TTS samples: $samples, duration: ${duration}s, RTF: $rtf"
    log "Roundtrip text: ${roundtrip_text:-<empty>}"
    log "Roundtrip hasChinese: $roundtrip_has_chinese, similarity: $roundtrip_similarity, status: $roundtrip_status"
    
    echo "TTS_AUDIO_SAMPLES=$samples"
    echo "TTS_AUDIO_DURATION_SEC=$duration"
    echo "TTS_RTF=$rtf"
    echo "TTS_ROUNDTRIP_TEXT=$roundtrip_text"
    echo "TTS_ROUNDTRIP_HAS_CHINESE=$roundtrip_has_chinese"
    echo "TTS_ROUNDTRIP_SIMILARITY=$roundtrip_similarity"
    echo "TTS_ROUNDTRIP_STATUS=$roundtrip_status"

    # Validate audio output
    if [ "$samples" -gt 0 ] && \
       [ "$roundtrip_has_chinese" = "true" ] && \
       [ "$roundtrip_status" = "PASS" ] && \
       awk "BEGIN { exit !($roundtrip_similarity >= $ROUNDTRIP_MIN_SIM) }"; then
      return 0
    else
      log "TTS test: audio or roundtrip validation failed"
      cat "$log_file" >> "$UI_LOG"
      return 1
    fi
  else
    log "TTS test: FAIL"
    cat "$log_file" >> "$UI_LOG"
    return 1
  fi
}

test_tts_destroy() {
  local log_file
  log_file=$(run_dumpapp_voice "tts destroy")
  
  if rg -q "TTS_DESTROY=SUCCESS" "$log_file"; then
    log "TTS destroy: SUCCESS"
    return 0
  else
    log "TTS destroy: FAIL"
    return 1
  fi
}

# Status tracking
voice_status_check="FAIL"
tts_init_status="FAIL"
tts_test_status="FAIL"
tts_destroy_status="FAIL"
tts_audio_samples="0"
tts_audio_duration="0"
tts_rtf="0"
tts_roundtrip_text=""
tts_roundtrip_has_chinese="false"
tts_roundtrip_similarity="0"
tts_roundtrip_status="FAIL"

# Main test sequence
log "=== Voice dumpapp Smoke Test ==="
ensure_app_foreground

# Step 1: Check voice models status
log "Step 1: Checking voice models status..."
if check_voice_status; then
  voice_status_check="PASS"
else
  log "WARNING: Voice models not ready. TTS test may fail."
  # Continue anyway to see what error we get
fi

# Step 2: Initialize TTS
log "Step 2: Initializing TTS..."
if test_tts_init; then
  tts_init_status="PASS"
else
  log "TTS init failed, skipping remaining TTS tests"
fi

# Step 3: Test TTS processing
if [ "$tts_init_status" = "PASS" ]; then
  log "Step 3: Testing TTS processing..."
  tts_output=$(test_tts_process "$TTS_TEST_TEXT" || true)
  if [ -n "$tts_output" ]; then
    tts_audio_samples=$(echo "$tts_output" | rg "TTS_AUDIO_SAMPLES=" | cut -d= -f2 || echo "0")
    tts_audio_duration=$(echo "$tts_output" | rg "TTS_AUDIO_DURATION_SEC=" | cut -d= -f2 || echo "0")
    tts_rtf=$(echo "$tts_output" | rg "TTS_RTF=" | cut -d= -f2 || echo "0")
    tts_roundtrip_text=$(echo "$tts_output" | rg "TTS_ROUNDTRIP_TEXT=" | cut -d= -f2- || echo "")
    tts_roundtrip_has_chinese=$(echo "$tts_output" | rg "TTS_ROUNDTRIP_HAS_CHINESE=" | cut -d= -f2 || echo "false")
    tts_roundtrip_similarity=$(echo "$tts_output" | rg "TTS_ROUNDTRIP_SIMILARITY=" | cut -d= -f2 || echo "0")
    tts_roundtrip_status=$(echo "$tts_output" | rg "TTS_ROUNDTRIP_STATUS=" | cut -d= -f2 || echo "FAIL")
    if [ "$tts_roundtrip_status" = "PASS" ] && [ "$tts_roundtrip_has_chinese" = "true" ]; then
      tts_test_status="PASS"
    fi
  fi
fi

# Step 4: Cleanup - destroy TTS
if [ "$tts_init_status" = "PASS" ]; then
  log "Step 4: Destroying TTS..."
  if test_tts_destroy; then
    tts_destroy_status="PASS"
  fi
fi

# Determine overall status
overall_status="FAIL"
if [ "$voice_status_check" = "PASS" ] && [ "$tts_init_status" = "PASS" ] && [ "$tts_test_status" = "PASS" ]; then
  overall_status="PASS"
fi

# Write summary
{
  echo "VOICE_DUMPAPP_SMOKE=$overall_status"
  echo "VOICE_STATUS_CHECK=$voice_status_check"
  echo "TTS_INIT_STATUS=$tts_init_status"
  echo "TTS_TEST_STATUS=$tts_test_status"
  echo "TTS_DESTROY_STATUS=$tts_destroy_status"
  echo "TTS_AUDIO_SAMPLES=$tts_audio_samples"
  echo "TTS_AUDIO_DURATION_SEC=$tts_audio_duration"
  echo "TTS_RTF=$tts_rtf"
  echo "TTS_ROUNDTRIP_TEXT=$tts_roundtrip_text"
  echo "TTS_ROUNDTRIP_HAS_CHINESE=$tts_roundtrip_has_chinese"
  echo "TTS_ROUNDTRIP_SIMILARITY=$tts_roundtrip_similarity"
  echo "TTS_ROUNDTRIP_STATUS=$tts_roundtrip_status"
  echo "TTS_TEST_TEXT=$TTS_TEST_TEXT"
  echo "ARTIFACT_DIR=$OUT_DIR"
} >"$SUMMARY_FILE"

log "=== Voice dumpapp Smoke Test Summary ==="
cat "$SUMMARY_FILE"
cat "$SUMMARY_FILE" | tee -a "$UI_LOG"

if [ "$overall_status" != "PASS" ]; then
  exit 1
fi
