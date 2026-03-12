#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CPP_FILE="$ROOT_DIR/app/src/main/cpp/mnn_wrapper_jni.cpp"
CMAKE_FILE="$ROOT_DIR/app/src/main/cpp/CMakeLists.txt"
HEADER_FILE="$ROOT_DIR/app/src/main/cpp/git_version.h"
SUMMARY_DIR="${ARTIFACT_DIR:-$ROOT_DIR/tests/smoke/artifacts/git_version_integration}"
SUMMARY_FILE="$SUMMARY_DIR/summary.txt"
BUILD_LOG="$SUMMARY_DIR/build.log"
HEADER_BACKUP=""

mkdir -p "$SUMMARY_DIR"

cleanup() {
  if [ -n "$HEADER_BACKUP" ] && [ -f "$HEADER_BACKUP" ]; then
    mv "$HEADER_BACKUP" "$HEADER_FILE"
  fi
}

trap cleanup EXIT

fail() {
  {
    echo "GIT_VERSION_INTEGRATION=FAIL"
    echo "REASON=$1"
  } >"$SUMMARY_FILE"
  cat "$SUMMARY_FILE"
  exit 1
}

rg -q 'GIT_COMMIT_ID' "$CPP_FILE" || fail "CPP_MISSING_GIT_COMMIT_ID"
rg -q '#include "git_version.h"' "$CPP_FILE" && fail "CPP_STILL_INCLUDES_GIT_VERSION_HEADER"
rg -q 'GIT_COMMIT_ID' "$CMAKE_FILE" || fail "CMAKE_MISSING_GIT_COMMIT_ID_DEFINE"

if [ -f "$HEADER_FILE" ]; then
  HEADER_BACKUP="$SUMMARY_DIR/git_version.h.bak"
  mv "$HEADER_FILE" "$HEADER_BACKUP"
fi

if ! ./gradlew :app:externalNativeBuildStandardDebug --no-daemon >"$BUILD_LOG" 2>&1; then
  fail "NATIVE_BUILD_FAILED"
fi

{
  echo "GIT_VERSION_INTEGRATION=PASS"
  echo "CPP_FILE=$CPP_FILE"
  echo "CMAKE_FILE=$CMAKE_FILE"
  echo "BUILD_LOG=$BUILD_LOG"
} >"$SUMMARY_FILE"

cat "$SUMMARY_FILE"
