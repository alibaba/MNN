#!/bin/bash

set -euo pipefail

WORKTREE_PATH="${1:?usage: 17_bisect_mnn_tts_lib.sh <worktree_path>}"
DEVICE_ID="${DEVICE_ID:-1b4a0523}"
INPUT_TEXT="${INPUT_TEXT:-你好，今天天气不错。}"
GOOD_AUDIO_SHA256="${GOOD_AUDIO_SHA256:-04e25afde1c13e66c6412bf9389f927ad7f5b5c8ecc4c1e873bd2025c747f984}"
MODEL_PATH="${MODEL_PATH:-/data/local/tmp/tts_models/bert-vits2-MNN}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-30}"
POLL_INTERVAL_SECONDS="${POLL_INTERVAL_SECONDS:-2}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHAT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
MNN_ROOT="$(cd "$CHAT_ROOT/../../.." && pwd)"
DEMO_DIR="$MNN_ROOT/apps/frameworks/mnn_tts/demo/android"
MNN_TTS_ANDROID_DIR="$MNN_ROOT/apps/frameworks/mnn_tts/android"
TARGET_LIB_DIR="$MNN_ROOT/project/android/build_64/lib"
LOG_DIR="$CHAT_ROOT/tests/smoke/artifacts/mnn_tts_bisect"

mkdir -p "$LOG_DIR"

commit_hash="$(git -C "$WORKTREE_PATH" rev-parse --short HEAD)"
run_log="$LOG_DIR/${commit_hash}.log"

pick_built_libs() {
  local build_root="$1/project/android/build_64"
  if [[ -f "$build_root/lib/libMNN.so" ]]; then
    BUILT_LIB_MNN="$build_root/lib/libMNN.so"
    BUILT_LIB_EXPRESS="$build_root/lib/libMNN_Express.so"
  elif [[ -f "$build_root/libMNN.so" ]]; then
    BUILT_LIB_MNN="$build_root/libMNN.so"
    BUILT_LIB_EXPRESS="$build_root/libMNN_Express.so"
  else
    echo "missing libMNN.so under $build_root" >&2
    exit 125
  fi
}

capture_device_log() {
  adb -s "$DEVICE_ID" logcat -d TTS_TEST:V MNN_TTS:V AudioChunksPlayer:V AndroidRuntime:E '*:S' >"$run_log.device"
}

echo "[bisect] commit=$commit_hash worktree=$WORKTREE_PATH"

rm -rf "$WORKTREE_PATH/project/android/build_64"
mkdir -p "$WORKTREE_PATH/project/android/build_64"
(
  cd "$WORKTREE_PATH/project/android/build_64"
  ANDROID_NDK="${ANDROID_NDK:-/Users/songjinde/Library/Android/sdk/ndk/27.2.12479018}" ../build_64.sh
) >"$run_log" 2>&1 || {
  cat "$run_log"
  exit 125
}

pick_built_libs "$WORKTREE_PATH"
cp -R "$BUILT_LIB_MNN" "$TARGET_LIB_DIR/libMNN.so"
if [[ -f "$BUILT_LIB_EXPRESS" ]]; then
  cp -R "$BUILT_LIB_EXPRESS" "$TARGET_LIB_DIR/libMNN_Express.so"
else
  rm -f "$TARGET_LIB_DIR/libMNN_Express.so"
fi

rm -rf "$MNN_TTS_ANDROID_DIR/.cxx" "$DEMO_DIR/build"
(
  cd "$DEMO_DIR"
  ./gradlew assembleDebug
) >>"$run_log" 2>&1 || {
  cat "$run_log"
  exit 125
}

APK_PATH="$DEMO_DIR/build/outputs/apk/debug/MNNTTSDemo-arm64-v8a-debug.apk"
adb -s "$DEVICE_ID" install -r "$APK_PATH" >>"$run_log" 2>&1 || {
  cat "$run_log"
  exit 125
}

adb -s "$DEVICE_ID" logcat -c
adb -s "$DEVICE_ID" shell am force-stop com.alibaba.mnn.tts.demo
adb -s "$DEVICE_ID" shell am start -n com.alibaba.mnn.tts.demo/.MainActivity \
  --ez auto_run true \
  --es model_path "$MODEL_PATH" \
  --es input_text "$INPUT_TEXT" >>"$run_log" 2>&1 || {
  cat "$run_log"
  exit 125
}

audio_hash=""
generator_sum=""
for (( elapsed=0; elapsed<MAX_WAIT_SECONDS; elapsed+=POLL_INTERVAL_SECONDS )); do
  sleep "$POLL_INTERVAL_SECONDS"
  capture_device_log
  audio_hash="$(rg -o 'TTS_AUDIO_SHA256=([0-9a-f]+)' -r '$1' "$run_log.device" | tail -n 1 || true)"
  if [[ -n "$audio_hash" ]]; then
    generator_sum="$(rg -o 'generator output_sum: [-0-9.]+' "$run_log.device" | tail -n 1 || true)"
    break
  fi
done

if [[ -z "$audio_hash" ]]; then
  capture_device_log
fi

audio_hash="${audio_hash:-$(rg -o 'TTS_AUDIO_SHA256=([0-9a-f]+)' -r '$1' "$run_log.device" | tail -n 1 || true)}"
generator_sum="$(rg -o 'generator output_sum: [-0-9.]+' "$run_log.device" | tail -n 1 || true)"

if [[ -z "$audio_hash" ]]; then
  cat "$run_log.device"
  exit 125
fi

echo "[bisect] commit=$commit_hash audio_sha256=$audio_hash" | tee -a "$run_log"
if [[ -n "$generator_sum" ]]; then
  echo "[bisect] commit=$commit_hash $generator_sum" | tee -a "$run_log"
fi

if [[ "$audio_hash" == "$GOOD_AUDIO_SHA256" ]]; then
  exit 0
fi

exit 1
