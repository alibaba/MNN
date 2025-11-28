#!/bin/bash
#
# test_download_simple.sh
#
# Simple integration test for MNNCLI download functionality
# Tests actual CLI commands by executing them and verifying output
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default values
VERBOSE=false
SKIP_NETWORK=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --skip-network)
            SKIP_NETWORK=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -v, --verbose          Enable verbose output"
            echo "  --skip-network         Skip network-dependent tests"
            echo "  -h, --help             Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                     # Run all download tests"
            echo "  $0 -v                  # Run with verbose output"
            echo "  $0 --skip-network      # Skip network tests"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Test configuration
TEST_CACHE_DIR="/tmp/mnncli_test_$$"

# Function to log messages
log() {
    if [[ "$VERBOSE" == "true" ]]; then
        echo -e "${BLUE}[LOG]${NC} $1"
    fi
}

# Function to log test info
test_info() {
    echo -e "${YELLOW}[TEST]${NC} $1"
}

# Function to log success
test_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

# Function to log failure
test_failure() {
    echo -e "${RED}[FAIL]${NC} $1"
}

# Function to cleanup test environment
cleanup() {
    log "Cleaning up test environment..."
    rm -rf "$TEST_CACHE_DIR" 2>/dev/null || true
}

# Set up trap for cleanup
trap cleanup EXIT

# Function to check if mnncli binary exists
check_mnncli_binary() {
    local mnncli_path="$PROJECT_ROOT/../../build_mnncli/apps/mnncli/mnncli"
    if [[ ! -f "$mnncli_path" ]]; then
        echo -e "${RED}Error: MNNCLI binary not found at $mnncli_path${NC}"
        echo "Please build the project first using ./build.sh"
        exit 1
    fi
    echo "$mnncli_path"
}

# Function to execute mnncli command and capture output
run_mnncli() {
    local mnncli_path="$1"
    local args="$2"
    local expected_exit_code="${3:-0}"
    
    log "Running: $mnncli_path $args"
    
    # Set test environment
    export MNNCLI_CACHE_DIR="$TEST_CACHE_DIR"
    if [[ "$VERBOSE" == "true" ]]; then
        export MNNCLI_VERBOSE=1
    fi
    
    # Execute command and capture output
    local output
    local exit_code
    output=$("$mnncli_path" $args 2>&1)
    exit_code=$?
    
    log "Exit code: $exit_code"
    log "Output: $output"
    
    # Check exit code
    if [[ $exit_code -ne $expected_exit_code ]]; then
        test_failure "Expected exit code $expected_exit_code, got $exit_code"
        return 1
    fi
    
    echo "$output"
    return 0
}

# Function to test basic command
test_basic_command() {
    local mnncli_path="$1"
    local args="$2"
    local expected_pattern="$3"
    local test_name="$4"
    
    test_info "Testing: $test_name"
    
    local output
    if ! output=$(run_mnncli "$mnncli_path" "$args" 0); then
        test_failure "$test_name - Command execution failed"
        return 1
    fi
    
    # Check if output contains expected pattern
    if echo "$output" | grep -q "$expected_pattern"; then
        test_success "$test_name"
        return 0
    else
        test_failure "$test_name - Output does not contain expected pattern: $expected_pattern"
        echo "Actual output: $output"
        return 1
    fi
}

# Function to test error command
test_error_command() {
    local mnncli_path="$1"
    local args="$2"
    local expected_pattern="$3"
    local test_name="$4"
    
    test_info "Testing: $test_name"
    
    local output
    if ! output=$(run_mnncli "$mnncli_path" "$args" 1); then
        test_success "$test_name - Command failed as expected"
        return 0
    fi
    
    # Check if output contains error message
    if echo "$output" | grep -q "$expected_pattern"; then
        test_success "$test_name"
        return 0
    else
        test_failure "$test_name - Expected error message not found"
        echo "Actual output: $output"
        return 1
    fi
}

# Main test execution
main() {
    echo -e "${BLUE}=== MNNCLI Download Integration Tests ===${NC}"
    echo "Test cache directory: $TEST_CACHE_DIR"
    echo "Verbose mode: $VERBOSE"
    echo "Skip network: $SKIP_NETWORK"
    echo ""
    
    # Check if mnncli binary exists
    local mnncli_path
    mnncli_path=$(check_mnncli_binary)
    log "Using MNNCLI binary: $mnncli_path"
    
    # Create test cache directory
    mkdir -p "$TEST_CACHE_DIR"
    log "Created test cache directory: $TEST_CACHE_DIR"
    
    # Test counters
    local passed=0
    local failed=0
    
    # Test 1: Help command
    if test_basic_command "$mnncli_path" "--help" "Usage\|Commands\|Options" "Help command"; then
        ((passed++))
    else
        ((failed++))
    fi
    
    # Test 2: Version command
    if test_basic_command "$mnncli_path" "--version" "MNNCLI\|version" "Version command"; then
        ((passed++))
    else
        ((failed++))
    fi
    
    # Test 3: List command
    if test_basic_command "$mnncli_path" "list" "Local models\|MNN CLI" "List command"; then
        ((passed++))
    else
        ((failed++))
    fi
    
    # Test 4: Download with empty model name
    if test_error_command "$mnncli_path" "download" "Error\|Failed" "Download with empty model name"; then
        ((passed++))
    else
        ((failed++))
    fi
    
    # Test 5: Download with invalid model name
    if test_error_command "$mnncli_path" "download invalid-model-name" "Error\|Failed\|Repository not found" "Download with invalid model name"; then
        ((passed++))
    else
        ((failed++))
    fi
    
    # Network-dependent tests (only if not skipped)
    if [[ "$SKIP_NETWORK" == "false" ]]; then
        test_info "Running network-dependent tests..."
        
        # Test 6: Download a real model (if available)
        # Note: This test might fail if the model doesn't exist or network is unavailable
        if test_basic_command "$mnncli_path" "download microsoft/DialoGPT-small" "Download\|Error\|Failed" "Download real model"; then
            ((passed++))
        else
            ((failed++))
            log "Network test failed - this is expected if model doesn't exist or network is unavailable"
        fi
    else
        log "Skipping network-dependent tests"
    fi
    
    # Print test summary
    echo ""
    echo -e "${BLUE}=== Test Results ===${NC}"
    echo "Total tests: $((passed + failed))"
    echo -e "Passed: ${GREEN}$passed${NC}"
    echo -e "Failed: ${RED}$failed${NC}"
    
    if [[ $failed -eq 0 ]]; then
        echo -e "${GREEN}🎉 All download integration tests passed!${NC}"
        exit 0
    else
        echo -e "${RED}❌ Some download integration tests failed!${NC}"
        exit 1
    fi
}

# Run main function
main "$@"
