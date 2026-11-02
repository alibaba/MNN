#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CLI_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
if [[ "$(uname -s)" != Darwin ]]; then
    echo "This package must be built on macOS." >&2
    exit 1
fi

OUTPUT="${1:-$CLI_DIR/dist/MNN-Studio-$(uname -m).dmg}"
if [[ -e "$OUTPUT" ]]; then
    echo "Output already exists: $OUTPUT" >&2
    exit 1
fi
mkdir -p "$(dirname "$OUTPUT")"
(cd "$CLI_DIR" && ./build.sh)

STAGING="$(mktemp -d "${TMPDIR:-/tmp}/mnn-studio-package.XXXXXX")"
trap 'rm -rf "$STAGING"' EXIT
IMAGE_ROOT="$STAGING/image"
mkdir -p "$IMAGE_ROOT"
APP="$IMAGE_ROOT/MNN Studio.app"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$SCRIPT_DIR/Info.plist" "$APP/Contents/Info.plist"
cp "$CLI_DIR/build_mnncli/mnncli" "$APP/Contents/Resources/mnncli"
ICONSET="$STAGING/MNNStudio.iconset"
mkdir "$ICONSET"
for size in 16 32 128 256 512; do
    sips -z "$size" "$size" "$CLI_DIR/../../doc/logo.png" --out "$ICONSET/icon_${size}x${size}.png" >/dev/null
    double=$((size * 2))
    sips -z "$double" "$double" "$CLI_DIR/../../doc/logo.png" --out "$ICONSET/icon_${size}x${size}@2x.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/MNNStudio.icns"
rm -rf "$ICONSET"
CLANG_MODULE_CACHE_PATH="$STAGING/swift-cache" swiftc -O -module-cache-path "$STAGING/swift-cache" \
    -target "$(uname -m)-apple-macosx13.0" -framework AppKit -framework WebKit \
    "$SCRIPT_DIR/MNNStudio.swift" -o "$APP/Contents/MacOS/MNNStudio"
codesign --force --deep --sign - "$APP"
codesign --verify --deep --strict "$APP"
ln -s /Applications "$IMAGE_ROOT/Applications"
hdiutil create -quiet -fs HFS+ -format UDZO -volname "MNN Studio" -srcfolder "$IMAGE_ROOT" "$OUTPUT"
hdiutil verify -quiet "$OUTPUT"
echo "Created $OUTPUT"
