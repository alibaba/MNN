#!/bin/bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <version_name> <version_code>"
    echo "Example: $0 0.7.7 752"
    exit 1
fi

VERSION_NAME="$1"
VERSION_CODE="$2"
BUILD_GRADLE_FILE="app/build.gradle"

if [[ ! "$VERSION_CODE" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] version_code must be an integer"
    exit 1
fi

if [[ ! -f "$BUILD_GRADLE_FILE" ]]; then
    echo "[ERROR] $BUILD_GRADLE_FILE not found. Run this script in apps/Android/MnnLlmChat"
    exit 1
fi

CURRENT_VERSION_NAME=$(grep "versionName" "$BUILD_GRADLE_FILE" | head -1 | sed 's/.*versionName "\(.*\)"/\1/')
CURRENT_VERSION_CODE=$(grep "versionCode" "$BUILD_GRADLE_FILE" | head -1 | sed 's/.*versionCode \([0-9]*\)/\1/')

# Use perl for stable first-match replacement across GNU/BSD environments.
perl -0pi -e "s/versionCode\\s+\\d+/versionCode $VERSION_CODE/" "$BUILD_GRADLE_FILE"
perl -0pi -e "s/versionName\\s+\"[^\"]+\"/versionName \"$VERSION_NAME\"/" "$BUILD_GRADLE_FILE"

UPDATED_VERSION_NAME=$(grep "versionName" "$BUILD_GRADLE_FILE" | head -1 | sed 's/.*versionName "\(.*\)"/\1/')
UPDATED_VERSION_CODE=$(grep "versionCode" "$BUILD_GRADLE_FILE" | head -1 | sed 's/.*versionCode \([0-9]*\)/\1/')

if [[ "$UPDATED_VERSION_NAME" != "$VERSION_NAME" || "$UPDATED_VERSION_CODE" != "$VERSION_CODE" ]]; then
    echo "[ERROR] Failed to update app/build.gradle"
    echo "  - expected versionName=$VERSION_NAME, versionCode=$VERSION_CODE"
    echo "  - actual   versionName=$UPDATED_VERSION_NAME, versionCode=$UPDATED_VERSION_CODE"
    exit 1
fi

echo "[INFO] Updated version:"
echo "  - versionName: $CURRENT_VERSION_NAME -> $VERSION_NAME"
echo "  - versionCode: $CURRENT_VERSION_CODE -> $VERSION_CODE"
