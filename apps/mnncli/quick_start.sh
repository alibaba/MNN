#!/bin/bash

# MNNCLI Quick Start Script
# This script quickly builds and tests mnncli

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}MNNCLI Quick Start${NC}"
echo "=================="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if build script exists
if [ ! -f "$SCRIPT_DIR/build.sh" ]; then
    echo -e "${RED}Error: build.sh not found${NC}"
    exit 1
fi

# Check if test script exists
if [ ! -f "$SCRIPT_DIR/test.sh" ]; then
    echo -e "${RED}Error: test.sh not found${NC}"
    exit 1
fi

# Make scripts executable
echo -e "${YELLOW}Making scripts executable...${NC}"
chmod +x "$SCRIPT_DIR/build.sh"
chmod +x "$SCRIPT_DIR/test.sh"
chmod +x "$SCRIPT_DIR/clean.sh"

echo -e "${GREEN}Scripts made executable${NC}"
echo ""

# Build the project
echo -e "${YELLOW}Building MNNCLI...${NC}"
if "$SCRIPT_DIR/build.sh"; then
    echo -e "${GREEN}Build successful!${NC}"
else
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi
echo ""

# Test the project
echo -e "${YELLOW}Testing MNNCLI...${NC}"
if "$SCRIPT_DIR/test.sh"; then
    echo -e "${GREEN}Tests passed!${NC}"
else
    echo -e "${RED}Tests failed!${NC}"
    exit 1
fi
echo ""

echo -e "${GREEN}Quick start completed successfully!${NC}"
echo ""
echo -e "${BLUE}What's next?${NC}"
echo "1. Try using mnncli: ./build_mnncli/apps/mnncli/mnncli --help"
echo "2. Check the README.md for usage examples"
echo "3. Run ./clean.sh to clean build files"
echo "4. Run ./build.sh to rebuild"
echo "5. Run ./test.sh to test again"
echo ""
echo -e "${GREEN}Happy coding! 🚀${NC}"
