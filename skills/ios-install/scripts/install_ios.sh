#!/bin/bash
# Build MNN.framework, bundle a model, build MNN Chat and install it on an iOS device.
# IMPORTANT: all compile steps run under env -i. This machine's shell exports
# SDKROOT/CPATH pointing at the MacOSX SDK, which breaks iOS builds on Xcode 26
# ("libc++ tried including <stddef.h> but didn't find..." / modulemap errors).
set -uo pipefail

SRC_ROOT="$HOME/work/AliNNPrivate"
APP_DIR=""
DEVICE=""
MODEL=""
TEAM="3TLG5LZ643"
BUNDLE_ID="com.jingbang.mnnchat.dev"
SKIP_FW=0

usage() {
    cat <<'EOF'
Usage: install_ios.sh --model <model-dir> [options]

Options:
  --model <dir>     (required) MNN model folder (config.json/llm.mnn/...)
  --device <id>     Device UDID (default: auto-detect the first online iPhone/iPad)
  --src <dir>       MNN source for the framework (default: ~/work/AliNNPrivate)
  --app <dir>       App project dir (default: apps/iOS/MNNLLMChat under this repo)
  --team <id>       Development team (default: 3TLG5LZ643, personal team)
  --bundle-id <id>  Bundle identifier (default: com.jingbang.mnnchat.dev)
  --skip-framework  Reuse existing MNN.framework, skip framework build

Backend: taken from the model's own config.json "backend_type" (the app reads
it first); edit that file to switch between cpu/metal.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2;;
        --device) DEVICE="$2"; shift 2;;
        --src) SRC_ROOT="$2"; shift 2;;
        --app) APP_DIR="$2"; shift 2;;
        --team) TEAM="$2"; shift 2;;
        --bundle-id) BUNDLE_ID="$2"; shift 2;;
        --skip-framework) SKIP_FW=1; shift;;
        -h|--help) usage; exit 0;;
        *) echo "unknown arg: $1"; usage; exit 1;;
    esac
done

[[ -n "$MODEL" ]] || { usage; exit 1; }
[[ -f "$MODEL/config.json" ]] || { echo "ERROR: $MODEL/config.json not found"; exit 1; }

if [[ -z "$APP_DIR" ]]; then
    APP_DIR="$(cd "$(dirname "$0")/../../.." && pwd)/apps/iOS/MNNLLMChat"
fi
[[ -d "$APP_DIR/MNNLLMiOS.xcodeproj" ]] || { echo "ERROR: no MNNLLMiOS.xcodeproj under $APP_DIR"; exit 1; }

if [[ -z "$DEVICE" ]]; then
    DEVICE=$(xcrun xctrace list devices 2>/dev/null | sed -n '/^== Devices ==/,/^== Devices Offline/p' \
        | grep -E '(iPhone|iPad)' | grep -Eo '\([0-9A-F-]{25,}\)$' | head -1 | tr -d '()')
    if [[ -z "$DEVICE" ]]; then
        DEVICE=$(xcrun devicectl list devices 2>/dev/null | grep -E '[[:space:]]available[[:space:]]' \
            | grep -Eo '[0-9A-F]{8}-([0-9A-F]{4}-){3}[0-9A-F]{12}' | head -1)
    fi
    [[ -n "$DEVICE" ]] || { echo "ERROR: no online iOS device found; pass --device <udid>"; exit 1; }
    echo "Auto-detected device: $DEVICE"
fi

MODEL_NAME=$(basename "$MODEL")
BUILD_DIR="$SRC_ROOT/build_ios_fw"
LOG="/tmp/mnnchat-ios-install.log"
SKILL_DIR="$(cd "$(dirname "$0")" && pwd)"

# Step 1: framework
if [[ "$SKIP_FW" -eq 1 ]]; then
    echo "[1/5] Skipping framework build (--skip-framework)"
elif [[ -d "$APP_DIR/MNN.framework" ]]; then
    echo "[1/5] MNN.framework already present in $APP_DIR, skipping build (delete it to force)"
else
    echo "[1/5] Configuring framework build"
    mkdir -p "$BUILD_DIR"
    env -i PATH=/usr/bin:/bin HOME="$HOME" sh -c "cd '$BUILD_DIR' && cmake '$SRC_ROOT' \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE='$SRC_ROOT/cmake/ios.toolchain.cmake' \
        -DARCHS=arm64 -DENABLE_BITCODE=0 -DMNN_AAPL_FMWK=1 -DMNN_SEP_BUILD=0 \
        -DMNN_BUILD_SHARED_LIBS=false -DMNN_USE_THREAD_POOL=OFF \
        -DMNN_ARM82=ON -DMNN_LOW_MEMORY=ON -DMNN_SUPPORT_TRANSFORMER_FUSE=ON \
        -DMNN_BUILD_LLM=ON -DMNN_CPU_WEIGHT_DEQUANT_GEMM=ON -DMNN_METAL=ON \
        -DMNN_BUILD_DIFFUSION=ON -DMNN_OPENCL=OFF -DLLM_SUPPORT_AUDIO=ON \
        -DMNN_BUILD_AUDIO=ON -DLLM_SUPPORT_VISION=ON -DMNN_BUILD_OPENCV=ON \
        -DMNN_IMGCODECS=ON -DMNN_BUILD_LLM_OMNI=ON" > "$LOG" 2>&1 \
        || { echo "ERROR: cmake configure failed, see $LOG"; exit 1; }
    echo "[2/5] Building framework (takes a few minutes)"
    env -i PATH=/usr/bin:/bin HOME="$HOME" make -C "$BUILD_DIR" MNN -j16 > "$LOG" 2>&1 \
        || { echo "ERROR: framework build failed, see $LOG"; exit 1; }
fi

# Step 2: framework into app
if [[ -d "$BUILD_DIR/MNN.framework" && "$SKIP_FW" -eq 0 && ! -d "$APP_DIR/MNN.framework" ]]; then
    cp -R "$BUILD_DIR/MNN.framework" "$APP_DIR/"
fi
[[ -d "$APP_DIR/MNN.framework" ]] || { echo "ERROR: $APP_DIR/MNN.framework missing"; exit 1; }

# Step 3: model
echo "[3/5] Bundling model $MODEL_NAME"
rm -rf "$APP_DIR/MNNLLMiOS/LocalModel/$MODEL_NAME"
cp -R "$MODEL" "$APP_DIR/MNNLLMiOS/LocalModel/"

# Step 4: build app
echo "[4/5] Building and installing app (model is large, takes a while)"
cp "$SKILL_DIR/../assets/personal.entitlements" /tmp/mnnchat-personal.entitlements
env -i PATH=/usr/bin:/bin:/usr/sbin:/sbin HOME="$HOME" USER="${USER:-}" \
    xcodebuild -project "$APP_DIR/MNNLLMiOS.xcodeproj" -scheme MNNLLMiOS \
    -destination "id=$DEVICE" -configuration Debug -allowProvisioningUpdates \
    CODE_SIGN_ENTITLEMENTS=/tmp/mnnchat-personal.entitlements \
    DEVELOPMENT_TEAM="$TEAM" PRODUCT_BUNDLE_IDENTIFIER="$BUNDLE_ID" \
    build > "$LOG" 2>&1
if ! grep -q "BUILD SUCCEEDED" "$LOG"; then
    echo "ERROR: app build failed. Errors:"
    grep "error:" "$LOG" | sort -u | head -10
    echo "Full log: $LOG"
    exit 1
fi

APP_PRODUCT=$(find "$HOME/Library/Developer/Xcode/DerivedData" -maxdepth 5 \
    -path "*Debug-iphoneos/MNNLLMiOS.app" -type d 2>/dev/null | head -1)
[[ -n "$APP_PRODUCT" ]] || { echo "ERROR: built app not found in DerivedData"; exit 1; }

# Step 5: install + launch
echo "[5/5] Installing to $DEVICE"
if ! xcrun devicectl device install app --device "$DEVICE" "$APP_PRODUCT" > "$LOG" 2>&1; then
    if grep -q "FreeProfileValidatedAppTracker\|ApplicationVerificationFailed" "$LOG"; then
        echo "ERROR: free-team verification failed. Personal teams allow at most 3 apps:"
        xcrun devicectl device info apps --device "$DEVICE" 2>/dev/null | sed -n '3,20p'
        echo "Ask the user which installed apps to uninstall, then:"
        echo "  xcrun devicectl device uninstall app --device $DEVICE <bundle-id>"
    else
        echo "ERROR: install failed, see $LOG"; tail -5 "$LOG"
    fi
    exit 1
fi
xcrun devicectl device process launch --device "$DEVICE" "$BUNDLE_ID" > /dev/null 2>&1 \
    || echo "NOTE: launch failed. If this is a first install with this team, trust the developer first:
Settings > General > VPN & Device Management > Trust 'Apple Development', then re-run launch."
echo "DONE: $MODEL_NAME installed and launched on device $DEVICE (backend: $(python3 -c "import json;print(json.load(open('$MODEL/config.json')).get('backend_type','cpu'))" 2>/dev/null))"
