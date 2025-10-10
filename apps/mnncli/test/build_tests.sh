#!/bin/bash

# Build script for MNN CLI tests
# This script builds all tests in the build directory

echo "Building MNN CLI tests..."

# Create build directory if it doesn't exist
mkdir -p build

# Configure CMake in build directory
echo "Configuring CMake..."
cd build
cmake .. -DBUILD_MNNCLI_TEST=ON

if [ $? -ne 0 ]; then
    echo "CMake configuration failed!"
    exit 1
fi

# Build all tests
echo "Building all tests..."
make -j$(nproc)

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo "All tests built successfully!"
echo "Binaries are located in the build directory."

cd ..