#!/bin/bash

# Run Unit Tests Script
# This script builds and runs all unit tests in the unit_tests directory

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "========================================="
echo "Unit Tests Runner"
echo "========================================="
echo ""

# Step 1: Configure (creates build system)
echo -e "${YELLOW}Step 1: Configuring tests...${NC}"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: cmake configuration failed${NC}"
    exit 1
fi
echo ""

# Step 2: Build (compiles tests)
echo -e "${YELLOW}Step 2: Building tests...${NC}"
cmake --build build -j
if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: build failed${NC}"
    exit 1
fi
echo ""

# Step 3: Run all tests
echo -e "${YELLOW}Step 3: Running all tests...${NC}"
echo "========================================="
echo ""

TOTAL_PASSED=0
TOTAL_FAILED=0

# List of test executables
TEST_EXECUTABLES=(
    "model_name_utils_tests"
    "file_utils_tests"
    "local_model_utils_tests"
)

# Run each test executable
for test_exe in "${TEST_EXECUTABLES[@]}"; do
    test_path="build/$test_exe"
    
    if [ -f "$test_path" ]; then
        echo -e "${GREEN}Running $test_exe...${NC}"
        echo ""
        
        if "$test_path"; then
            echo -e "${GREEN}✓ $test_exe passed${NC}"
            ((TOTAL_PASSED++))
        else
            echo -e "${RED}✗ $test_exe failed${NC}"
            ((TOTAL_FAILED++))
        fi
        echo ""
        echo "========================================="
        echo ""
    else
        echo -e "${RED}ERROR: Test executable not found: $test_path${NC}"
        ((TOTAL_FAILED++))
        echo ""
    fi
done

# Summary
echo ""
echo "========================================="
echo "Overall Test Summary"
echo "========================================="
echo -e "${GREEN}Total Passed: ${TOTAL_PASSED}${NC}"
if [ $TOTAL_FAILED -eq 0 ]; then
    echo -e "${GREEN}Total Failed: ${TOTAL_FAILED}${NC}"
    echo ""
    echo -e "${GREEN}All tests passed! ✓${NC}"
    exit 0
else
    echo -e "${RED}Total Failed: ${TOTAL_FAILED}${NC}"
    echo ""
    echo -e "${RED}Some tests failed! ✗${NC}"
    exit 1
fi

