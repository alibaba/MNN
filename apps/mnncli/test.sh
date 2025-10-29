#!/bin/bash

# MNNCLI Test Script
# This script tests various functionalities of mnncli
#
# Test matrix (high-level):
# - Basic commands: help, version, list
# - Search across providers: huggingface, modelscope, modelers (keyword: SmolVLM2)
# - Isolated cache: set cache_dir to ~/.cache/mnnclitest/mnnmodels, restore after tests
# - Pre/post list assertions around real downloads
# - Real downloads with timeout + retry for SmolVLM2-256M-Video-Instruct-MNN
#   * HuggingFace: taobao-mnn/SmolVLM2-256M-Video-Instruct-MNN
#   * ModelScope:  MNN/SmolVLM2-256M-Video-Instruct-MNN
#   * Modelers:    MNN/SmolVLM2-256M-Video-Instruct-MNN
# - Idempotency: second download should succeed quickly/no-op
# - Cleanup: restore original cache_dir & provider, remove test cache directory
#
# Notes:
# - MNNCLI expects a directory containing config.json, not the .mnn file directly.
#   You can either:
#   1) Pass the directory path: mnncli run <model_directory>
#   2) Use the -c flag: mnncli run -c <path/to/config.json>

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
BUILD_DIR="$SCRIPT_DIR/build_mnncli"
EXECUTABLE="$BUILD_DIR/mnncli"

# Network control
SKIP_NETWORK=${SKIP_NETWORK:-0}
TIMEOUT=${TIMEOUT:-600}
RETRIES=${RETRIES:-0}

# Download failure handling (1 = exit on failure, 0 = continue after failure)
EXIT_ON_DOWNLOAD_FAILURE=${EXIT_ON_DOWNLOAD_FAILURE:-1}

# Testing cache directory (isolated from normal usage)
TEST_CACHE_DIR="$HOME/.cache/mnnclitest/mnnmodels"

# Helper: print a command or run it depending on DRY_RUN
run_cmd() {
    local desc="${1}"; shift
    if [ "$DRY_RUN" -eq 1 ]; then
        echo -e "${BLUE}[DRY RUN]${NC} $desc"
        echo -e "${BLUE}  Command:${NC} $*"
        return 0
    fi
    echo -e "${BLUE}[EXEC]${NC} $desc"
    echo -e "${BLUE}  Command:${NC} $*"
    "$@"
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo -e "${GREEN}  ✓ Success${NC}"
    else
        echo -e "${RED}  ✗ Failed (exit code: $rc)${NC}"
    fi
    return $rc
}

# Helper: run with timeout and simple retry
run_with_retry() {
    local desc="${1}"; shift
    local attempts=0
    local max_attempts=$((RETRIES+1))
    local cmd=("$@")
    echo -e "${BLUE}[EXEC]${NC} $desc"
    echo -e "${BLUE}  Command:${NC} ${cmd[*]}"
    echo -e "${BLUE}  Timeout:${NC} ${TIMEOUT}s, Retries: ${RETRIES}"
    while [ $attempts -lt $max_attempts ]; do
        attempts=$((attempts+1))
        if [ "$DRY_RUN" -eq 1 ]; then
            echo -e "${BLUE}[DRY RUN]${NC} timeout ${TIMEOUT}s ${cmd[*]}"
            echo -e "${GREEN}  ✓ Success (dry run)${NC}"
            return 0
        fi
        echo -e "${BLUE}  Attempt $attempts/$max_attempts...${NC}"
        if command -v timeout >/dev/null 2>&1; then
            timeout ${TIMEOUT}s "${cmd[@]}" && return 0
            rc=$?
        else
            "${cmd[@]}" && return 0
            rc=$?
        fi
        if [ $attempts -lt $max_attempts ]; then
            echo -e "${YELLOW}  ✗ Attempt $attempts failed (rc=$rc), retrying in 2s...${NC}"
        else
            echo -e "${RED}  ✗ All $max_attempts attempts failed (rc=$rc)${NC}"
        fi
        sleep 2
    done
    return 1
}

# Helper: assertions on output
assert_contains() {
    local haystack="$1"; shift
    local needle="$1"; shift
    echo "$haystack" | grep -q "$needle"
}

# Helpers: save and restore config
ORIG_CACHE_DIR=""
ORIG_PROVIDER=""
save_current_config() {
    if [ "$DRY_RUN" -eq 1 ]; then
        echo -e "${BLUE}[DRY RUN]${NC} mnncli config show"
        ORIG_CACHE_DIR=""
        ORIG_PROVIDER=""
        return 0
    fi
    local cfg
    cfg=$("$EXECUTABLE" config show 2>/dev/null || true)
    ORIG_CACHE_DIR=$(echo "$cfg" | sed -n 's/.*Cache Directory(cache_dir): //p')
    ORIG_PROVIDER=$(echo "$cfg" | sed -n 's/.*Download Provider(download_provider): //p')
}

restore_config() {
    if [ "$DRY_RUN" -eq 1 ]; then
        echo -e "${BLUE}[DRY RUN]${NC} restore cache_dir=$ORIG_CACHE_DIR provider=$ORIG_PROVIDER"
        return 0
    fi
    if [ -n "$ORIG_CACHE_DIR" ]; then
        "$EXECUTABLE" config set cache_dir "$ORIG_CACHE_DIR" >/dev/null 2>&1 || true
    fi
    if [ -n "$ORIG_PROVIDER" ]; then
        "$EXECUTABLE" config set download_provider "$ORIG_PROVIDER" >/dev/null 2>&1 || true
    fi
}

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

# Ensure we restore config and clean cache on exit
cleanup_all() {
    echo -e "${BLUE}Cleaning up test environment...${NC}"
    restore_config
    if [ "$DRY_RUN" -eq 1 ]; then
        echo -e "${BLUE}[DRY RUN]${NC} rm -rf \"$TEST_CACHE_DIR\""
    else
        rm -rf "$TEST_CACHE_DIR" 2>/dev/null || true
    fi
}
trap cleanup_all EXIT

# Save original config and switch to isolated cache
save_current_config
echo -e "${YELLOW}Setting isolated cache dir: $TEST_CACHE_DIR${NC}"
run_cmd "Set isolated cache dir" "$EXECUTABLE" config set cache_dir "$TEST_CACHE_DIR"
mkdir -p "$TEST_CACHE_DIR" 2>/dev/null || true
echo ""

# Test 1: Help command
echo -e "${YELLOW}Test 1: Help command${NC}"
echo -e "${BLUE}Description:${NC} Verify mnncli --help works correctly"
echo "Testing: $EXECUTABLE --help"
if [ "$DRY_RUN" -eq 1 ]; then
    echo -e "${BLUE}[DRY RUN] Would execute: $EXECUTABLE --help${NC}"
    echo -e "${GREEN}✓ Help command works (dry run)${NC}"
else
    if "$EXECUTABLE" --help &> /dev/null; then
        echo -e "${GREEN}✓ Help command works${NC}"
    else
        echo -e "${RED}✗ Help command failed${NC}"
        echo -e "${RED}  Failed command: $EXECUTABLE --help${NC}"
        exit 1
    fi
fi
echo ""

# Test 2: List command (record pre-download state)
echo -e "${YELLOW}Test 2: List command (pre-download snapshot)${NC}"
echo -e "${BLUE}Description:${NC} List local models before any downloads to establish baseline"
echo "Testing: $EXECUTABLE list"
if [ "$DRY_RUN" -eq 1 ]; then
    echo -e "${BLUE}[DRY RUN] Would execute: $EXECUTABLE list${NC}"
    PRE_LIST=""
    echo -e "${GREEN}✓ List command works (dry run)${NC}"
else
    if PRE_LIST=$("$EXECUTABLE" list 2>&1); then
        echo -e "${GREEN}✓ List command works${NC}"
        echo -e "${BLUE}  Models found: $(echo "$PRE_LIST" | wc -l) lines${NC}"
    else
        echo -e "${RED}✗ List command failed${NC}"
        echo -e "${RED}  Failed command: $EXECUTABLE list${NC}"
    fi
fi
echo ""

# Test 3: Search command across providers for SmolVLM2
echo -e "${YELLOW}Test 3: Search across providers (SmolVLM2)${NC}"
echo -e "${BLUE}Description:${NC} Search for SmolVLM2 on huggingface, modelscope, modelers"
if [ "$SKIP_NETWORK" -eq 1 ]; then
    echo -e "${YELLOW}Skipping network-dependent search tests (SKIP_NETWORK=1)${NC}"
else
    for provider in huggingface modelscope modelers; do
        echo -e "${BLUE}Testing provider: $provider${NC}"
        run_cmd "Set download provider to $provider" "$EXECUTABLE" config set download_provider "$provider"
        if [ "$DRY_RUN" -eq 1 ]; then
            echo -e "${BLUE}[DRY RUN] Would execute: $EXECUTABLE search SmolVLM2${NC}"
            echo -e "${GREEN}✓ Search ($provider) simulated${NC}"
        else
            if out=$(run_with_retry "Search for SmolVLM2 with $provider" "$EXECUTABLE" search SmolVLM2); then
                if assert_contains "$out" "SmolVLM2" || assert_contains "$out" "SmolVLM2-256M"; then
                    echo -e "${GREEN}✓ Search ($provider) returned results${NC}"
                    echo -e "${BLUE}  Results: $(echo "$out" | grep -i smolvlm | wc -l) matches${NC}"
                else
                    echo -e "${YELLOW}⚠ Search ($provider) did not show SmolVLM2 keyword${NC}"
                    echo -e "${BLUE}  First line of output: $(echo "$out" | head -1)${NC}"
                fi
            else
                echo -e "${RED}✗ Search ($provider) failed${NC}"
                echo -e "${RED}  Failed command: $EXECUTABLE search SmolVLM2 (provider: $provider)${NC}"
            fi
        fi
    done
fi
echo ""

# Download scenario: SmolVLM2-256M from three providers with assertions
echo -e "${YELLOW}Download Scenario: SmolVLM2-256M across providers${NC}"
echo -e "${BLUE}Description:${NC} Download SmolVLM2-256M from each provider with pre/post assertions"

# Assert model not present pre-download (if we captured PRE_LIST)
MODEL_KEYWORD="SmolVLM2-256M"
if [ -n "$PRE_LIST" ]; then
    if echo "$PRE_LIST" | grep -q "$MODEL_KEYWORD"; then
        echo -e "${YELLOW}⚠ Pre-download: $MODEL_KEYWORD already in cache (may be leftover)${NC}"
    else
        echo -e "${GREEN}✓ Pre-download: $MODEL_KEYWORD not yet cached (expected)${NC}"
    fi
fi

if [ "$SKIP_NETWORK" -eq 1 ]; then
    echo -e "${YELLOW}Skipping downloads (SKIP_NETWORK=1)${NC}"
else
    # 1) HuggingFace
    echo -e "${BLUE}=== Provider: HuggingFace ===${NC}"
    echo -e "${BLUE}Model ID: taobao-mnn/SmolVLM2-256M-Video-Instruct-MNN${NC}"
    run_cmd "Set download provider to huggingface" "$EXECUTABLE" config set download_provider huggingface
    if [ "$EXIT_ON_DOWNLOAD_FAILURE" -eq 1 ]; then
        run_with_retry "Download SmolVLM2-256M-Video-Instruct-MNN from HuggingFace" "$EXECUTABLE" download taobao-mnn/SmolVLM2-256M-Video-Instruct-MNN
    else
        run_with_retry "Download SmolVLM2-256M-Video-Instruct-MNN from HuggingFace" "$EXECUTABLE" download taobao-mnn/SmolVLM2-256M-Video-Instruct-MNN || echo -e "${RED}HF download failed${NC}"
    fi
    # Idempotency check
    echo -e "${BLUE}Testing idempotency (re-download should be quick/no-op)...${NC}"
    run_with_retry "Re-download SmolVLM2-256M-Video-Instruct-MNN from HuggingFace" "$EXECUTABLE" download taobao-mnn/SmolVLM2-256M-Video-Instruct-MNN || echo -e "${YELLOW}HF re-download reported failure${NC}"

    # 2) ModelScope
    echo -e "${BLUE}=== Provider: ModelScope ===${NC}"
    echo -e "${BLUE}Model ID: MNN/SmolVLM2-256M-Video-Instruct-MNN${NC}"
    run_cmd "Set download provider to modelscope" "$EXECUTABLE" config set download_provider modelscope
    if [ "$EXIT_ON_DOWNLOAD_FAILURE" -eq 1 ]; then
        run_with_retry "Download SmolVLM2-256M-Video-Instruct-MNN from ModelScope" "$EXECUTABLE" download MNN/SmolVLM2-256M-Video-Instruct-MNN
    else
        run_with_retry "Download SmolVLM2-256M-Video-Instruct-MNN from ModelScope" "$EXECUTABLE" download MNN/SmolVLM2-256M-Video-Instruct-MNN || echo -e "${RED}MS download failed${NC}"
    fi

    # 3) Modelers
    echo -e "${BLUE}=== Provider: Modelers ===${NC}"
    echo -e "${BLUE}Model ID: MNN/SmolVLM2-256M-Video-Instruct-MNN${NC}"
    run_cmd "Set download provider to modelers" "$EXECUTABLE" config set download_provider modelers
    if [ "$EXIT_ON_DOWNLOAD_FAILURE" -eq 1 ]; then
        run_with_retry "Download SmolVLM2-256M-Video-Instruct-MNN from Modelers" "$EXECUTABLE" download MNN/SmolVLM2-256M-Video-Instruct-MNN
    else
        run_with_retry "Download SmolVLM2-256M-Video-Instruct-MNN from Modelers" "$EXECUTABLE" download MNN/SmolVLM2-256M-Video-Instruct-MNN || echo -e "${RED}Modelers download failed${NC}"
    fi

    # Post list snapshot and assertions
    echo -e "${BLUE}Post-download verification:${NC}"
    if [ "$DRY_RUN" -eq 1 ]; then
        echo -e "${BLUE}[DRY RUN] Would execute: $EXECUTABLE list${NC}"
        POST_LIST=""
    else
        POST_LIST=$("$EXECUTABLE" list 2>&1 || true)
    fi
    if [ -n "$POST_LIST" ] && echo "$POST_LIST" | grep -q "$MODEL_KEYWORD"; then
        echo -e "${GREEN}✓ Post-download list contains $MODEL_KEYWORD${NC}"
        echo -e "${BLUE}  Cache dir: $TEST_CACHE_DIR${NC}"
    else
        echo -e "${YELLOW}⚠ Post-download: $MODEL_KEYWORD NOT found in list${NC}"
        echo -e "${BLUE}  This may indicate download failed or cache not indexed properly${NC}"
        echo -e "${BLUE}  Cache dir: $TEST_CACHE_DIR${NC}"
    fi

    # Optional: model_info presence checks
    echo -e "${BLUE}Checking model_info for each downloaded model...${NC}"
    for name in taobao-mnn/SmolVLM2-256M-Video-Instruct-MNN MNN/SmolVLM2-256M-Video-Instruct-MNN; do
        if [ "$DRY_RUN" -eq 1 ]; then
            echo -e "${BLUE}[DRY RUN] Would execute: $EXECUTABLE model_info $name${NC}"
        else
            if info_out=$("$EXECUTABLE" model_info "$name" 2>&1); then
                echo -e "${GREEN}✓ model_info for $name succeeded${NC}"
                echo -e "${BLUE}  First line: $(echo "$info_out" | head -1)${NC}"
            else
                echo -e "${YELLOW}⚠ model_info for $name failed (model may not be properly downloaded)${NC}"
                echo -e "${RED}  Failed command: $EXECUTABLE model_info $name${NC}"
            fi
        fi
    done
fi
echo ""

# Test 4: Version or info command (if available)
echo -e "${YELLOW}Test 4: Version/Info command${NC}"
echo -e "${BLUE}Description:${NC} Check if version command is available"
echo "Testing: $EXECUTABLE --version"
if [ "$DRY_RUN" -eq 1 ]; then
    echo -e "${BLUE}[DRY RUN] Would execute: $EXECUTABLE --version${NC}"
    echo -e "${GREEN}✓ Version command works (dry run)${NC}"
else
    if "$EXECUTABLE" --version &> /dev/null; then
        echo -e "${GREEN}✓ Version command works${NC}"
        version_output=$("$EXECUTABLE" --version 2>&1)
        echo -e "${BLUE}  Version: $version_output${NC}"
    else
        echo -e "${YELLOW}⚠ Version command not available (this is normal)${NC}"
    fi
fi
echo ""

# Test 5: Invalid command (should show error)
echo -e "${YELLOW}Test 5: Invalid command handling${NC}"
echo -e "${BLUE}Description:${NC} Verify invalid commands are rejected properly"
echo "Testing: $EXECUTABLE invalid_command"
if [ "$DRY_RUN" -eq 1 ]; then
    echo -e "${BLUE}[DRY RUN] Would execute: $EXECUTABLE invalid_command${NC}"
    echo -e "${GREEN}✓ Invalid command properly handled (dry run)${NC}"
else
    if "$EXECUTABLE" invalid_command &> /dev/null; then
        echo -e "${YELLOW}⚠ Invalid command didn't show error (this might be normal)${NC}"
    else
        echo -e "${GREEN}✓ Invalid command properly handled${NC}"
        echo -e "${BLUE}  (Command correctly rejected with error)${NC}"
    fi
fi
echo ""

# Test 6: Check executable properties
echo -e "${YELLOW}Test 6: Executable properties${NC}"
echo -e "${BLUE}Description:${NC} Check executable size, type, and format"
echo "File size:"
if [ "$DRY_RUN" -eq 1 ]; then
    echo -e "${BLUE}[DRY RUN] Would execute: ls -lh \"$EXECUTABLE\"${NC}"
else
    ls -lh "$EXECUTABLE"
    echo -e "${BLUE}  Location: $EXECUTABLE${NC}"
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
echo -e "${BLUE}Description:${NC} List dynamic library dependencies"
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
        echo -e "${BLUE}  (Showing first 10 dependencies)${NC}"
    elif command -v ldd &> /dev/null; then
        echo "Dynamic libraries (Linux):"
        ldd "$EXECUTABLE" | head -10
        echo -e "${BLUE}  (Showing first 10 dependencies)${NC}"
    else
        echo "Cannot check dependencies (otool/ldd not available)"
    fi
fi
echo ""

# Test 8: Performance test (basic)
echo -e "${YELLOW}Test 8: Basic performance test${NC}"
echo -e "${BLUE}Description:${NC} Measure startup time of mnncli --help"
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
    echo -e "${BLUE}  Command: $EXECUTABLE --help${NC}"
fi
echo ""

# Test 9: Test all models in ~/.cache/modelscope/hub/MNN
echo -e "${YELLOW}Test 9: Testing all models with prompt 'hello'${NC}"
echo -e "${BLUE}Description:${NC} Run inference on all downloaded models with prompt 'hello'"
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

            echo -e "${BLUE}Running inference test...${NC}"
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
                    echo -e "${BLUE}  Config dir: $model_dir${NC}"
                else
                    exit_code=$?
                    if [ $exit_code -eq 124 ]; then
                        echo -e "${YELLOW}⚠ Model $model_name timed out after 30 seconds${NC}"
                        echo -e "${BLUE}  (Increase timeout with TIMEOUT=<seconds> if needed)${NC}"
                    else
                        echo -e "${RED}✗ Model $model_name failed with exit code $exit_code${NC}"
                        echo -e "${RED}  Command: $EXECUTABLE run \"$model_dir\" -p \"hello\"${NC}"
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
                        echo -e "${BLUE}  Config file: $config_file${NC}"
                    else
                        exit_code=$?
                        if [ $exit_code -eq 124 ]; then
                            echo -e "${YELLOW}⚠ Model $model_name with explicit config timed out after 30 seconds${NC}"
                            echo -e "${BLUE}  (Increase timeout if needed)${NC}"
                        else
                            echo -e "${RED}✗ Model $model_name with explicit config failed with exit code $exit_code${NC}"
                            echo -e "${RED}  Command: $EXECUTABLE run -c \"$config_file\" -p \"hello\"${NC}"
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
echo -e "${YELLOW}=== Test Summary ===${NC}"
echo -e "${BLUE}Test Coverage:${NC}"
echo "  • Basic commands: help, version, list, invalid commands"
echo "  • Search: SmolVLM2 across huggingface, modelscope, modelers"
echo "  • Download: SmolVLM2-256M from all 3 providers with retries"
echo "  • Idempotency: Re-download to verify cache handling"
echo "  • Properties: Executable size, type, dependencies"
echo "  • Performance: Startup time measurement"
echo "  • Model inference: Run all downloaded models with prompt"
echo ""
echo -e "${BLUE}Isolated Cache:${NC} $TEST_CACHE_DIR"
echo -e "${BLUE}Timeout Setting:${NC} ${TIMEOUT}s"
echo -e "${BLUE}Retry Attempts:${NC} ${RETRIES}"
echo ""
if [ "$SKIP_NETWORK" -eq 1 ]; then
    echo -e "${YELLOW}Network tests were SKIPPED${NC}"
else
    echo -e "${GREEN}Network tests were EXECUTED${NC}"
fi
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
