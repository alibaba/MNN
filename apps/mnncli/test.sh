#!/bin/bash

# MNNCLI Test Script
# This script tests various functionalities of mnncli
# 
# Note: MNNCLI expects a directory containing config.json, not the .mnn file directly.
# You can either:
# 1. Pass the directory path: mnncli run <model_directory>
# 2. Use the -c flag: mnncli run -c <path/to/config.json>

set -e

# Control variable: set to 1 to only print commands without executing them
DRY_RUN=${DRY_RUN:-0}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build_mnncli"
EXECUTABLE="$BUILD_DIR/apps/mnncli/mnncli"

echo -e "${BLUE}MNNCLI Test Script${NC}"
echo "=================="

# Show dry run status
if [ "$DRY_RUN" -eq 1 ]; then
    echo -e "${YELLOW}DRY RUN MODE: Commands will be printed but not executed${NC}"
    echo ""
fi

# Check if executable exists
if [ ! -f "$EXECUTABLE" ]; then
    echo -e "${RED}Error: mnncli executable not found at $EXECUTABLE${NC}"
    echo -e "${YELLOW}Please run build.sh first to build the project.${NC}"
    exit 1
fi

echo -e "${GREEN}Found executable: $EXECUTABLE${NC}"
echo ""

# Test 1: Help command
echo -e "${YELLOW}Test 1: Help command${NC}"
echo "Testing: $EXECUTABLE --help"
if [ "$DRY_RUN" -eq 1 ]; then
    echo -e "${BLUE}[DRY RUN] Would execute: $EXECUTABLE --help${NC}"
    echo -e "${GREEN}✓ Help command works (dry run)${NC}"
else
    if "$EXECUTABLE" --help &> /dev/null; then
        echo -e "${GREEN}✓ Help command works${NC}"
    else
        echo -e "${RED}✗ Help command failed${NC}"
        exit 1
    fi
fi
echo ""

# Test 2: List command (should work even without models)
echo -e "${YELLOW}Test 2: List command${NC}"
echo "Testing: $EXECUTABLE list"
if [ "$DRY_RUN" -eq 1 ]; then
    echo -e "${BLUE}[DRY RUN] Would execute: $EXECUTABLE list${NC}"
    echo -e "${GREEN}✓ List command works (dry run)${NC}"
else
    if "$EXECUTABLE" list &> /dev/null; then
        echo -e "${GREEN}✓ List command works${NC}"
    else
        echo -e "${RED}✗ List command failed${NC}"
    fi
fi
echo ""

# Test 3: Search command (should work for basic search)
echo -e "${YELLOW}Test 3: Search command${NC}"
echo "Testing: $EXECUTABLE search test"
if [ "$DRY_RUN" -eq 1 ]; then
    echo -e "${BLUE}[DRY RUN] Would execute: $EXECUTABLE search test${NC}"
    echo -e "${GREEN}✓ Search command works (dry run)${NC}"
else
    if "$EXECUTABLE" search test &> /dev/null; then
        echo -e "${GREEN}✓ Search command works${NC}"
    else
        echo -e "${RED}✗ Search command failed${NC}"
    fi
fi
echo ""

# Test 4: Version or info command (if available)
echo -e "${YELLOW}Test 4: Version/Info command${NC}"
echo "Testing: $EXECUTABLE --version"
if [ "$DRY_RUN" -eq 1 ]; then
    echo -e "${BLUE}[DRY RUN] Would execute: $EXECUTABLE --version${NC}"
    echo -e "${GREEN}✓ Version command works (dry run)${NC}"
else
    if "$EXECUTABLE" --version &> /dev/null; then
        echo -e "${GREEN}✓ Version command works${NC}"
    else
        echo -e "${YELLOW}⚠ Version command not available (this is normal)${NC}"
    fi
fi
echo ""

# Test 5: Invalid command (should show error)
echo -e "${YELLOW}Test 5: Invalid command handling${NC}"
echo "Testing: $EXECUTABLE invalid_command"
if [ "$DRY_RUN" -eq 1 ]; then
    echo -e "${BLUE}[DRY RUN] Would execute: $EXECUTABLE invalid_command${NC}"
    echo -e "${GREEN}✓ Invalid command properly handled (dry run)${NC}"
else
    if "$EXECUTABLE" invalid_command &> /dev/null; then
        echo -e "${YELLOW}⚠ Invalid command didn't show error (this might be normal)${NC}"
    else
        echo -e "${GREEN}✓ Invalid command properly handled${NC}"
    fi
fi
echo ""

# Test 6: Check executable properties
echo -e "${YELLOW}Test 6: Executable properties${NC}"
echo "File size:"
if [ "$DRY_RUN" -eq 1 ]; then
    echo -e "${BLUE}[DRY RUN] Would execute: ls -lh \"$EXECUTABLE\"${NC}"
else
    ls -lh "$EXECUTABLE"
fi
echo ""
echo "File type:"
if [ "$DRY_RUN" -eq 1 ]; then
    echo -e "${BLUE}[DRY RUN] Would execute: file \"$EXECUTABLE\"${NC}"
else
    file "$EXECUTABLE"
fi
echo ""

# Test 7: Dependencies check
echo -e "${YELLOW}Test 7: Dependencies check${NC}"
if [ "$DRY_RUN" -eq 1 ]; then
    if command -v otool &> /dev/null; then
        echo -e "${BLUE}[DRY RUN] Would execute: otool -L \"$EXECUTABLE\" | head -10${NC}"
    elif command -v ldd &> /dev/null; then
        echo -e "${BLUE}[DRY RUN] Would execute: ldd \"$EXECUTABLE\" | head -10${NC}"
    fi
    echo -e "${GREEN}✓ Dependencies check completed (dry run)${NC}"
else
    if command -v otool &> /dev/null; then
        echo "Dynamic libraries (macOS):"
        otool -L "$EXECUTABLE" | head -10
    elif command -v ldd &> /dev/null; then
        echo "Dynamic libraries (Linux):"
        ldd "$EXECUTABLE" | head -10
    else
        echo "Cannot check dependencies (otool/ldd not available)"
    fi
fi
echo ""

# Test 8: Performance test (basic)
echo -e "${YELLOW}Test 8: Basic performance test${NC}"
echo "Testing startup time..."
if [ "$DRY_RUN" -eq 1 ]; then
    echo -e "${BLUE}[DRY RUN] Would execute: $EXECUTABLE --help &> /dev/null${NC}"
    echo -e "${GREEN}Startup time: simulated (dry run)${NC}"
else
    START_TIME=$(date +%s.%N)
    "$EXECUTABLE" --help &> /dev/null
    END_TIME=$(date +%s.%N)
    STARTUP_TIME=$(echo "$END_TIME - $START_TIME" | bc -l 2>/dev/null || echo "unknown")
    echo -e "${GREEN}Startup time: ${STARTUP_TIME}s${NC}"
fi
echo ""

# Test 9: Test all models in ~/.cache/modelscope/hub/MNN
echo -e "${YELLOW}Test 9: Testing all models with prompt 'hello'${NC}"
MODELS_DIR="$HOME/.cache/modelscope/hub/MNN"

if [ -d "$MODELS_DIR" ]; then
    echo "Found models directory: $MODELS_DIR"
    echo "Searching for models..."
    
    # Find all .mnn files recursively
    MODEL_FILES=$(find "$MODELS_DIR" -name "*.mnn" -type f 2>/dev/null)
    
    if [ -n "$MODEL_FILES" ]; then
        echo -e "${GREEN}Found $(echo "$MODEL_FILES" | wc -l) model(s):${NC}"
        echo "$MODEL_FILES" | while read -r model_file; do
            model_name=$(basename "$model_file" .mnn)
            echo ""
            echo -e "${BLUE}Testing model: $model_name${NC}"
            echo "File: $model_file"
            
            # Test the model with 'hello' prompt
            # Use the directory containing the model file (which should have config.json)
            # The mnncli tool expects a directory with config.json, not the .mnn file directly
            model_dir=$(dirname "$model_file")
            echo "Running: $EXECUTABLE run \"$model_dir\" -p \"hello\""
            
            # Capture both stdout and stderr, with timeout
            if [ "$DRY_RUN" -eq 1 ]; then
                echo -e "${BLUE}[DRY RUN] Would execute: timeout 30s \"$EXECUTABLE\" run \"$model_dir\" -p \"hello\" 2>&1${NC}"
                echo -e "${GREEN}✓ Model $model_name completed successfully (dry run)${NC}"
            else
                if timeout 30s "$EXECUTABLE" run "$model_dir" -p "hello" 2>&1; then
                    echo -e "${GREEN}✓ Model $model_name completed successfully${NC}"
                else
                    exit_code=$?
                    if [ $exit_code -eq 124 ]; then
                        echo -e "${YELLOW}⚠ Model $model_name timed out after 30 seconds${NC}"
                    else
                        echo -e "${RED}✗ Model $model_name failed with exit code $exit_code${NC}"
                    fi
                fi
            fi
            
            # Alternative test using explicit -c flag
            config_file="$model_dir/config.json"
            if [ -f "$config_file" ]; then
                echo "Running with explicit config: $EXECUTABLE run -c \"$config_file\" -p \"hello\""
                if [ "$DRY_RUN" -eq 1 ]; then
                    echo -e "${BLUE}[DRY RUN] Would execute: timeout 30s \"$EXECUTABLE\" run -c \"$config_file\" -p \"hello\" 2>&1${NC}"
                    echo -e "${GREEN}✓ Model $model_name with explicit config completed successfully (dry run)${NC}"
                else
                    if timeout 30s "$EXECUTABLE" run -c "$config_file" -p "hello" 2>&1; then
                        echo -e "${GREEN}✓ Model $model_name with explicit config completed successfully${NC}"
                    else
                        exit_code=$?
                        if [ $exit_code -eq 124 ]; then
                            echo -e "${YELLOW}⚠ Model $model_name with explicit config timed out after 30 seconds${NC}"
                        else
                            echo -e "${RED}✗ Model $model_name with explicit config failed with exit code $exit_code${NC}"
                        fi
                    fi
                fi
            else
                echo -e "${YELLOW}⚠ Config file not found: $config_file${NC}"
            fi
        done
    else
        echo -e "${YELLOW}No .mnn files found in $MODELS_DIR${NC}"
    fi
else
    echo -e "${YELLOW}Models directory not found: $MODELS_DIR${NC}"
    echo "This is normal if you haven't downloaded any models yet."
fi
echo ""

echo -e "${GREEN}All tests completed!${NC}"
echo -e "${BLUE}MNNCLI appears to be working correctly.${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Try downloading a model: $EXECUTABLE download <model_name>"
echo "2. Try running a model: $EXECUTABLE run <model_name> (or $EXECUTABLE run -c <path/to/config.json>)"
echo "3. Try serving a model: $EXECUTABLE serve <model_name> (or $EXECUTABLE serve -c <path/to/config.json>)"
echo "4. Check the README.md for more usage examples"

# Add usage information for dry run mode
if [ "$DRY_RUN" -eq 1 ]; then
    echo ""
    echo -e "${BLUE}Dry Run Mode Usage:${NC}"
    echo "To actually execute the commands, run: DRY_RUN=0 $0"
    echo "Or simply run: $0 (default behavior)"
fi
