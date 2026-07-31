#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ANDROID_BUILD_DIR="${REPO_ROOT}/project/android/build_64"
JOBS="${JOBS:-4}"

if [[ -z "${JAVA_HOME:-}" && -x "/Applications/Android Studio.app/Contents/jbr/Contents/Home/bin/java" ]]; then
    export JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home"
fi
if [[ -z "${ANDROID_HOME:-}" && -d "${HOME}/Library/Android/sdk" ]]; then
    export ANDROID_HOME="${HOME}/Library/Android/sdk"
fi

mkdir -p "${ANDROID_BUILD_DIR}"
cd "${ANDROID_BUILD_DIR}"

../build_64.sh "\
-DMNN_LOW_MEMORY=true \
-DMNN_CPU_WEIGHT_DEQUANT_GEMM=true \
-DMNN_BUILD_LLM=true \
-DMNN_SUPPORT_TRANSFORMER_FUSE=true \
-DMNN_ARM82=true \
-DMNN_USE_LOGCAT=true \
-DMNN_BUILD_AUDIO=true \
-DMNN_SEP_BUILD=OFF \
-DCMAKE_SHARED_LINKER_FLAGS='-Wl,-z,max-page-size=16384' \
-DCMAKE_INSTALL_PREFIX=."

make -j"${JOBS}" install

cd "${SCRIPT_DIR}"
../MnnLlmChat/gradlew -p "${SCRIPT_DIR}" :app:assembleDebug

echo "APK: ${SCRIPT_DIR}/app/build/outputs/apk/debug/app-debug.apk"
