#!/bin/bash
# This script downloads and prepares QNN dependencies, mimicking the behavior of
# project/android/qnnprepare.gradle for use in non-Android build environments.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# URL for QNN libraries zip file
QNN_LIBS_URL='http://meta.alicdn.com/data/mnn/libs/qnn_inc_libs_2_37.zip'
# Project root is the current directory where the script is run
PROJECT_ROOT=$(pwd)
# Temporary directory for downloads and extraction
BUILD_DIR="$PROJECT_ROOT/build_qnn_temp"
QNN_ZIP_NAME='qnn_inc_libs.zip'
QNN_ZIP_FILE="$BUILD_DIR/$QNN_ZIP_NAME"
# Temporary directory for unzipping
QNN_TMP_DIR="$BUILD_DIR/qnn_tmp"
# Final destination for QNN headers and libraries
QNN_DEST_DIR="$PROJECT_ROOT/source/backend/qnn/3rdParty"

echo "BUILD_QNN is ON. Preparing QNN dependencies..."

# Fast-path: if destination already prepared, skip download/unpack/copy
DEST_INCLUDE_DIR="$QNN_DEST_DIR/include"
DEST_LIB_DIR="$QNN_DEST_DIR/lib"
if [ -d "$DEST_INCLUDE_DIR" ] && [ -n "$(find "$DEST_INCLUDE_DIR" -mindepth 1 -print -quit 2>/dev/null)" ]; then
    echo "Detected existing QNN SDK at: $QNN_DEST_DIR"
    # Ensure env vars are set even when skipping work
    QNN_SDK_ROOT_PATH="$(cd "$QNN_DEST_DIR" && pwd)"
    export QNN_SDK_ROOT="$QNN_SDK_ROOT_PATH"
    ENV_FILE="$PROJECT_ROOT/.qnn_env"
    echo "export QNN_SDK_ROOT=\"$QNN_SDK_ROOT_PATH\"" > "$ENV_FILE"
    echo "Set QNN_SDK_ROOT to: $QNN_SDK_ROOT_PATH"
    echo "You can add it to your shell by running: source $ENV_FILE"
    exit 0
fi

# 1. Download the QNN zip file if it doesn't exist or is corrupted
mkdir -p "$BUILD_DIR"

download_qnn_zip() {
    echo "Downloading QNN dependencies from ${QNN_LIBS_URL}"
    curl -fL --retry 3 --retry-delay 2 -o "$QNN_ZIP_FILE" "$QNN_LIBS_URL"
    echo "Downloaded to: ${QNN_ZIP_FILE}"
}

validate_zip() {
    unzip -tq "$QNN_ZIP_FILE" >/dev/null 2>&1
}

if [ ! -f "$QNN_ZIP_FILE" ]; then
    download_qnn_zip
else
    echo "Using cached zip: ${QNN_ZIP_FILE}"
    if ! validate_zip; then
        echo "Cached zip appears to be invalid or corrupted. Re-downloading..."
        rm -f "$QNN_ZIP_FILE"
        download_qnn_zip
        if ! validate_zip; then
            echo "Error: Downloaded zip is invalid. Please try again later." >&2
            exit 1
        fi
    fi
fi

# 2. Unpack the zip file into a clean temporary directory
echo "Cleaning temp directory and unpacking..."
rm -rf "$QNN_TMP_DIR"
mkdir -p "$QNN_TMP_DIR"
unzip -q "$QNN_ZIP_FILE" -d "$QNN_TMP_DIR"

# Remove macOS-specific resource fork directory if it exists, which causes issues.
if [ -d "$QNN_TMP_DIR/__MACOSX" ]; then
    echo "Removing __MACOSX directory..."
    rm -rf "$QNN_TMP_DIR/__MACOSX"
fi

# 3. Find the main directory inside the unzipped archive
# We look for the directory that contains the 'include' folder.
# This handles archives that may have a nested top-level directory.
INCLUDE_DIR=$(find "$QNN_TMP_DIR" -type d -name "include" | head -n 1)
if [ -z "$INCLUDE_DIR" ]; then
    echo "Error: Failed to find 'include' directory in the extracted archive." >&2
    exit 1
fi
# The source directory is the parent of the 'include' directory
EXTRACTED_QNN_DIR=$(dirname "$INCLUDE_DIR")
echo "Found QNN content in: $EXTRACTED_QNN_DIR"

# 4. Copy headers and libraries to their final destination

echo "Creating destination directories..."
mkdir -p "$DEST_INCLUDE_DIR"
mkdir -p "$DEST_LIB_DIR"

# Copy include files
echo "Copying include files..."
# The trailing slash on the source is important for `cp -r` to copy contents
cp -r "$EXTRACTED_QNN_DIR/include/"* "$DEST_INCLUDE_DIR/"
echo "QNN includes copied to: $DEST_INCLUDE_DIR"

# Determine the source library directory (either 'jniLibs' or 'lib')
SOURCE_LIBS_DIR="$EXTRACTED_QNN_DIR/jniLibs"
if [ ! -d "$SOURCE_LIBS_DIR" ]; then
    SOURCE_LIBS_DIR="$EXTRACTED_QNN_DIR/lib"
fi

# Copy library files if the directory exists
if [ -d "$SOURCE_LIBS_DIR" ]; then
    echo "Copying library files from $SOURCE_LIBS_DIR..."
    cp -r "$SOURCE_LIBS_DIR/"* "$DEST_LIB_DIR/"
    echo "QNN libs copied to: $DEST_LIB_DIR"
else
    echo "Warning: No 'lib' or 'jniLibs' directory found in $EXTRACTED_QNN_DIR"
fi

# 5. Clean up temporary unzip directory but keep the cached zip for future runs
echo "Cleaning up temporary files..."
rm -rf "$QNN_TMP_DIR"

echo "QNN dependencies preparation completed successfully."

# 6. Export QNN_SDK_ROOT for current shell and persist to .qnn_env for future shells
QNN_SDK_ROOT_PATH="$(cd "$QNN_DEST_DIR" && pwd)"
export QNN_SDK_ROOT="$QNN_SDK_ROOT_PATH"
ENV_FILE="$PROJECT_ROOT/.qnn_env"
echo "export QNN_SDK_ROOT=\"$QNN_SDK_ROOT_PATH\"" > "$ENV_FILE"
echo "Set QNN_SDK_ROOT to: $QNN_SDK_ROOT_PATH"
echo "You can add it to your shell by running: source $ENV_FILE"
