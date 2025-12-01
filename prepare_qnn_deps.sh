#!/bin/bash
# This script downloads and prepares QNN dependencies, mimicking the behavior of
# project/android/qnnprepare.gradle for use in non-Android build environments.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# URL for QNN libraries zip file
QNN_LIBS_URL='http://meta.alicdn.com/data/mnn/libs/qnn_inc_libs.zip'
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

# 1. Download the QNN zip file if it doesn't exist
mkdir -p "$BUILD_DIR"
if [ ! -f "$QNN_ZIP_FILE" ]; then
    echo "Downloading QNN dependencies from ${QNN_LIBS_URL}"
    # Use curl to download, following redirects (-L) and showing progress
    curl -L -o "$QNN_ZIP_FILE" "$QNN_LIBS_URL"
    echo "Downloaded to: ${QNN_ZIP_FILE}"
else
    echo "Using cached zip: ${QNN_ZIP_FILE}"
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
DEST_INCLUDE_DIR="$QNN_DEST_DIR/include"
DEST_LIB_DIR="$QNN_DEST_DIR/lib"

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

# 5. Clean up temporary build directory
echo "Cleaning up temporary files..."
rm -rf "$BUILD_DIR"

echo "QNN dependencies preparation completed successfully."
