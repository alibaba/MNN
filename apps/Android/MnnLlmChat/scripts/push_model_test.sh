#!/bin/bash

# Model Test Script for MnnLlmChat
# This script pushes model files and test binaries to the phone and runs a test
#
# Usage:
#   ./scripts/push_model_test.sh                    # Normal verbose output
#   VERBOSE=false ./scripts/push_model_test.sh      # Quiet mode, less output

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="MnnLlmChat"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ROOT_PROJECT_DIR="$(dirname "$(dirname "$(dirname "$PROJECT_DIR")")")"

# Control verbosity (set to false to reduce output)
VERBOSE=${VERBOSE:-true}

# Paths
MODELSCOPE_CACHE_DIR="$HOME/.cache/modelscope"
MNN_MODELS_DIR="$MODELSCOPE_CACHE_DIR/hub/models/MNN"
BUILD_64_DIR="$ROOT_PROJECT_DIR/project/android/build_64"
PHONE_TEST_DIR="/data/local/tmp/test_model"

# Functions
log_info() {
    if [[ "$VERBOSE" == "true" ]]; then
        echo -e "${BLUE}[INFO]${NC} $1"
    fi
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    log_info "Checking requirements..."
    
    # Check if we're in the right directory
    if [[ ! -f "app/build.gradle" ]]; then
        log_error "This script must be run from the project root directory"
        exit 1
    fi
    
    # Check if ADB is available
    if ! command -v adb &> /dev/null; then
        log_error "ADB is not installed or not in PATH"
        exit 1
    fi
    
    # Check if modelscope cache directory exists
    if [[ ! -d "$MODELSCOPE_CACHE_DIR" ]]; then
        log_error "ModelScope cache directory not found: $MODELSCOPE_CACHE_DIR"
        exit 1
    fi
    
    # Check if models directory exists (either MNN subdirectory or main models directory)
    if [[ ! -d "$MNN_MODELS_DIR" ]] && [[ ! -d "$MODELSCOPE_CACHE_DIR/hub/models" ]]; then
        log_error "Neither MNN models directory nor main models directory found"
        log_info "Available directories in $MODELSCOPE_CACHE_DIR:"
        ls -la "$MODELSCOPE_CACHE_DIR" 2>/dev/null || true
        exit 1
    fi
    
    # Check if build_64 directory exists
    if [[ ! -d "$BUILD_64_DIR" ]]; then
        log_error "Build 64 directory not found: $BUILD_64_DIR"
        exit 1
    fi
    
    log_success "Requirements check completed"
}

check_device_connection() {
    log_info "Checking device connection..."
    
    # Check if any device is connected
    if ! adb devices | grep -q "device$"; then
        log_error "No Android device connected. Please connect a device and enable USB debugging."
        exit 1
    fi
    
    # Get device info
    DEVICE_ID=$(adb devices | grep "device$" | head -n1 | cut -f1)
    log_info "Connected device: $DEVICE_ID"
    
    # Check if device is rooted (needed for /data/local/tmp access)
    if ! adb shell "su -c 'ls /data/local/tmp'" &> /dev/null; then
        log_warning "Device may not be rooted. Some operations might fail."
    fi
    
    log_success "Device connection verified"
}

setup_phone_directory() {
    log_info "Setting up phone directory..."
    
    # Create test directory on phone
    adb shell "mkdir -p $PHONE_TEST_DIR"
    adb shell "chmod 755 $PHONE_TEST_DIR"
    
    log_success "Phone directory setup completed"
}

select_and_push_model() {
    log_info "Selecting model from models directory..."
    
    # First try MNN subdirectory, then fall back to main models directory
    if [[ -d "$MNN_MODELS_DIR" ]]; then
        MODELS_DIR="$MNN_MODELS_DIR"
        log_info "Using MNN models directory: $MODELS_DIR"
    else
        MODELS_DIR="$MODELSCOPE_CACHE_DIR/hub/models"
        log_info "MNN subdirectory not found, using main models directory: $MODELS_DIR"
    fi
    
    # Debug: show what's actually in the models directory
    if [[ "$VERBOSE" == "true" ]]; then
        log_info "Contents of $MODELS_DIR:"
        ls -la "$MODELS_DIR" 2>/dev/null || true
        echo "----------------------------------------"
    fi
    
    # List available models
    log_info "Available models in $MODELS_DIR:"
    echo "----------------------------------------"
    
    # Get list of model directories (excluding . and .. and the parent directory)
    MODEL_DIRS=()
    
    # Get top-level directories, excluding . and .. and the parent directory itself
    while IFS= read -r -d '' dir; do
        if [[ "$dir" != "$MODELS_DIR" ]]; then
            MODEL_DIRS+=("$dir")
        fi
    done < <(find "$MODELS_DIR" -maxdepth 1 -type d ! -name "." ! -name ".." -print0 | sort -z)
    
    # Debug: show what we found
    if [[ "$VERBOSE" == "true" ]]; then
        log_info "Found ${#MODEL_DIRS[@]} model directories:"
        for dir in "${MODEL_DIRS[@]}"; do
            log_info "  Found: $dir"
        done
    fi
    
    if [[ ${#MODEL_DIRS[@]} -eq 0 ]]; then
        log_error "No model directories found in $MODELS_DIR. Please check your ModelScope cache structure."
        return 1
    fi
    
    # Display models with numbers
    for i in "${!MODEL_DIRS[@]}"; do
        model_name=$(basename "${MODEL_DIRS[$i]}")
        echo "$((i+1)). $model_name"
    done
    echo "----------------------------------------"
    
    # Get user selection
    while true; do
        read -p "Select a model (1-${#MODEL_DIRS[@]}): " selection
        if [[ "$selection" =~ ^[0-9]+$ ]] && [[ "$selection" -ge 1 ]] && [[ "$selection" -le "${#MODEL_DIRS[@]}" ]]; then
            SELECTED_MODEL_DIR="${MODEL_DIRS[$((selection-1))]}"
            break
        else
            echo "Please enter a valid number between 1 and ${#MODEL_DIRS[@]}"
        fi
    done
    
    SELECTED_MODEL_NAME=$(basename "$SELECTED_MODEL_DIR")
    log_info "Selected model: $SELECTED_MODEL_NAME"
    
    # Push the selected model files
    log_info "Pushing all files for $SELECTED_MODEL_NAME..."
    
    # Push all files from the selected directory
    MODEL_FILES=$(find "$SELECTED_MODEL_DIR" -type f 2>/dev/null)
    
    if [[ -z "$MODEL_FILES" ]]; then
        log_warning "No files found in $SELECTED_MODEL_NAME directory"
        return 1
    fi
    
    # Push each file
    for file in $MODEL_FILES; do
        filename=$(basename "$file")
        log_info "Pushing $filename..."
        adb push "$file" "$PHONE_TEST_DIR/"
    done
    
    log_success "Model files for $SELECTED_MODEL_NAME pushed to phone"
    
    # Show what files were pushed
    if [[ "$VERBOSE" == "true" ]]; then
        log_info "Files pushed for $SELECTED_MODEL_NAME:"
        adb shell "ls -la $PHONE_TEST_DIR/*" 2>/dev/null || true
    fi
}

push_test_binaries() {
    log_info "Pushing test binaries from build_64 directory..."
    
    # Check for llm_demo binary
    if [[ -f "$BUILD_64_DIR/llm_demo" ]]; then
        log_info "Pushing llm_demo binary..."
        adb push "$BUILD_64_DIR/llm_demo" "$PHONE_TEST_DIR/"
        adb shell "chmod +x $PHONE_TEST_DIR/llm_demo"
    else
        log_warning "llm_demo binary not found in $BUILD_64_DIR"
    fi
    
    # Push .so files
    SO_FILES=$(find "$BUILD_64_DIR" -name "*.so" 2>/dev/null)
    if [[ -n "$SO_FILES" ]]; then
        log_info "Pushing .so files..."
        for file in $SO_FILES; do
            filename=$(basename "$file")
            log_info "Pushing $filename..."
            adb push "$file" "$PHONE_TEST_DIR/"
        done
    else
        log_warning "No .so files found in $BUILD_64_DIR"
    fi
    
    # Push other important files
    for file in "config.json" "tokenizer.json" "tokenizer_config.json" "special_tokens_map.json"; do
        if [[ -f "$BUILD_64_DIR/$file" ]]; then
            log_info "Pushing $file..."
            adb push "$BUILD_64_DIR/$file" "$PHONE_TEST_DIR/"
        fi
    done
    
    # Push libMNN.so from the correct location
    if [[ -f "$BUILD_64_DIR/libMNN.so" ]]; then
        log_info "Found libMNN.so in build directory"
    else
        log_error "libMNN.so not found at $BUILD_64_DIR/libMNN.so. Please ensure it's built."
        return 1
    fi
    
    log_success "Test binaries pushed to phone"
}

run_test() {
    log_info "Running test on phone..."
    
    # Create prompt file
    adb shell "echo 'hello' > $PHONE_TEST_DIR/prompt"
    
    # Find config file
    CONFIG_FILE=""
    for config in "config.json" "model_config.json" "config.ini"; do
        if adb shell "test -f $PHONE_TEST_DIR/$config" 2>/dev/null; then
            CONFIG_FILE="$config"
            break
        fi
    done
    
    # Set LD_LIBRARY_PATH to include current directory for .so files
    if [[ -z "$CONFIG_FILE" ]]; then
        log_warning "No config file found, trying to run without config..."
        adb shell "cd $PHONE_TEST_DIR && export LD_LIBRARY_PATH=./ && ./llm_demo prompt"
    else
        log_info "Using config file: $CONFIG_FILE"
        adb shell "cd $PHONE_TEST_DIR && export LD_LIBRARY_PATH=./ && ./llm_demo $CONFIG_FILE prompt"
    fi
    
    log_success "Test execution completed"
}

list_phone_files() {
    if [[ "$VERBOSE" == "true" ]]; then
        log_info "Listing files on phone..."
        adb shell "ls -la $PHONE_TEST_DIR"
    fi
}

verify_library_dependencies() {
    log_info "Verifying library dependencies..."
    
    # Check if libMNN.so is present
    if adb shell "test -f $PHONE_TEST_DIR/libMNN.so" 2>/dev/null; then
        log_success "libMNN.so found on phone"
    else
        log_error "libMNN.so not found on phone - this will cause execution to fail"
        return 1
    fi
    
    # Check if llm_demo is executable
    if adb shell "test -x $PHONE_TEST_DIR/llm_demo" 2>/dev/null; then
        log_success "llm_demo is executable on phone"
    else
        log_error "llm_demo is not executable on phone"
        return 1
    fi
    
    log_success "Library dependencies verified"
}

cleanup() {
    log_info "Cleaning up..."
    # Optionally remove test directory from phone
    read -p "Do you want to remove the test directory from phone? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        adb shell "rm -rf $PHONE_TEST_DIR"
        log_success "Phone test directory removed"
    fi
}

main() {
    log_info "Starting Model Test Script for $PROJECT_NAME"
    
    check_requirements
    check_device_connection
    setup_phone_directory
    select_and_push_model
    push_test_binaries
    list_phone_files
    verify_library_dependencies
    run_test
    
    log_success "Model test script completed successfully"
    
    cleanup
}

# Handle script interruption
trap 'log_error "Script interrupted"; exit 1' INT TERM

# Run main function
main "$@"
