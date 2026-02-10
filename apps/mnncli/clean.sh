#!/bin/bash

# MNNCLI Clean Script
# This script cleans build files and cache

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Cleaning MNNCLI build files...${NC}"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build_mnncli"

# Clean build directory
if [ -d "$BUILD_DIR" ]; then
    echo -e "${YELLOW}Removing build directory: $BUILD_DIR${NC}"
    rm -rf "$BUILD_DIR"
    echo -e "${GREEN}Build directory removed${NC}"
else
    echo -e "${YELLOW}Build directory not found: $BUILD_DIR${NC}"
fi

# Clean any object files in source directory
echo -e "${YELLOW}Cleaning source directory...${NC}"
find "$SCRIPT_DIR/src" -name "*.o" -delete 2>/dev/null || true
find "$SCRIPT_DIR/src" -name "*.d" -delete 2>/dev/null || true

# Clean any temporary files
echo -e "${YELLOW}Cleaning temporary files...${NC}"
find "$SCRIPT_DIR" -name "*.tmp" -delete 2>/dev/null || true
find "$SCRIPT_DIR" -name "*.log" -delete 2>/dev/null || true

# Clean CMake cache files
if [ -f "$PROJECT_ROOT/CMakeCache.txt" ]; then
    echo -e "${YELLOW}Removing CMake cache...${NC}"
    rm -f "$PROJECT_ROOT/CMakeCache.txt"
fi

if [ -d "$PROJECT_ROOT/CMakeFiles" ]; then
    echo -e "${YELLOW}Removing CMake files...${NC}"
    rm -rf "$PROJECT_ROOT/CMakeFiles"
fi

echo -e "${GREEN}Clean completed!${NC}"
echo -e "${YELLOW}To rebuild, run: ./build.sh${NC}"
