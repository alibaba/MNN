#!/bin/sh
# Regenerates the engine xcframeworks consumed by the two iOS demo apps and
# restages the compile-time headers for the MNNChatCore package.
#
# Engine topology (dual-engine, since the internal and public MNN schema
# OpType numberings are mutually incompatible at the flatbuffer level):
#   Vendor/public/    MNN built from public MNN master   -> MNNLLMChat app
#                     (model market models are exported with the public schema)
#   Vendor/internal/  MNN built from AliNNPrivate master -> MNNPocket app
#                     (bundled slm270M/dflash models use the internal schema)
#   Vendor/shared/    sherpa-mnn static lib + staged c-api headers (both apps)
#
# CMNNInference compiles header-only against Vendor/public/MNN.framework's
# headers (the common API subset; both engine binaries export those symbols).
# Apps link their own engine xcframework directly — the package never links
# an engine, so the two apps can carry different engines.
#
# Usage:
#   build-xcframeworks.sh --engine public   --framework /path/to/MNN.framework
#   build-xcframeworks.sh --engine internal --framework /path/to/MNN.framework
#   build-xcframeworks.sh                   # regenerate everything in place
#
# Run this after rebuilding an engine so the apps pick up the new binary.
# All of Vendor/ and the staged header dirs are gitignored.

set -eu

IOS_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PKG_DIR="$IOS_DIR/MNNChatCore"
VENDOR="$PKG_DIR/Vendor"
SHERPA_SRC="$IOS_DIR/../frameworks/sherpa-mnn/sherpa-mnn"

ENGINE=""
FRAMEWORK=""
while [ $# -gt 0 ]; do
    case "$1" in
        --engine) ENGINE="$2"; shift 2;;
        --framework) FRAMEWORK="$2"; shift 2;;
        *) echo "unknown arg: $1"; exit 1;;
    esac
done
if [ -n "$ENGINE" ] && [ "$ENGINE" != "public" ] && [ "$ENGINE" != "internal" ]; then
    echo "ERROR: --engine must be public or internal"; exit 1
fi

make_xcframework() { # $1=flavor
    local flavor="$1"
    if [ ! -d "$VENDOR/$flavor/MNN.framework" ]; then
        echo "ERROR: $VENDOR/$flavor/MNN.framework missing — pass --framework or drop one in"
        exit 1
    fi
    rm -rf "$VENDOR/$flavor/MNN.xcframework"
    xcodebuild -create-xcframework \
      -framework "$VENDOR/$flavor/MNN.framework" \
      -output "$VENDOR/$flavor/MNN.xcframework"
    echo "Wrote $VENDOR/$flavor/MNN.xcframework"
}

# Optional: install a freshly built engine framework into a flavor slot.
if [ -n "$ENGINE" ]; then
    [ -n "$FRAMEWORK" ] || { echo "ERROR: --framework required with --engine"; exit 1; }
    rm -rf "$VENDOR/$ENGINE/MNN.framework"
    cp -R "$FRAMEWORK" "$VENDOR/$ENGINE/MNN.framework"
    make_xcframework "$ENGINE"
    exit 0
fi

# Full regeneration.
make_xcframework public
make_xcframework internal

# sherpa-mnn static lib -> shared xcframework (headers keep the
# "sherpa-mnn/c-api/..." include prefix used by PresetZipformerASRWrapper.mm).
HEADERS_STAGING="$VENDOR/shared/sherpa-headers"
rm -rf "$HEADERS_STAGING"
mkdir -p "$HEADERS_STAGING/sherpa-mnn/c-api"
cp "$SHERPA_SRC/c-api/c-api.h" "$HEADERS_STAGING/sherpa-mnn/c-api/c-api.h"
rm -rf "$VENDOR/shared/SherpaMNN.xcframework"
xcodebuild -create-xcframework \
  -library "$VENDOR/shared/libsherpa-mnn.a" \
  -headers "$HEADERS_STAGING" \
  -output "$VENDOR/shared/SherpaMNN.xcframework"
echo "Wrote $VENDOR/shared/SherpaMNN.xcframework"

# Compile-time headers for CMNNInference (from the public engine = common API
# subset). Consumed via headerSearchPath, never shipped.
MNN_INCLUDE="$PKG_DIR/Sources/CMNNInference/mnn-include"
rm -rf "$MNN_INCLUDE"
mkdir -p "$MNN_INCLUDE/MNN"
cp -R "$VENDOR/public/MNN.framework/Headers/" "$MNN_INCLUDE/MNN/"
SHERPA_INCLUDE="$PKG_DIR/Sources/CMNNInference/sherpa-include"
rm -rf "$SHERPA_INCLUDE"
mkdir -p "$SHERPA_INCLUDE/sherpa-mnn/c-api"
cp "$SHERPA_SRC/c-api/c-api.h" "$SHERPA_INCLUDE/sherpa-mnn/c-api/c-api.h"
echo "Staged headers: $MNN_INCLUDE, $SHERPA_INCLUDE"
