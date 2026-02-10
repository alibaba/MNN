#!/bin/bash

# QNN Optimized Cleanup Script
# Simplified version that cleans QNN libraries and verifies each deletion immediately

set -e

# Configuration
PACKAGE_NAME="com.alibaba.mnnllm.android"
APP_DATA_DIR="/data/data/${PACKAGE_NAME}"

# Directories to clean up
CLEANUP_DIRS=(
    "files/.mnnmodels/modelscope/models--MNN--qnn_arm64_libs"
    "files/.mnnmodels/modelscope/qnn_arm64_libs"
    "files/.mnnmodels/modelers/qnn_arm64_libs"
    "files/.mnnmodels/modelers/models--MNN--qnn_arm64_libs"
    "files/.mnnmodels/qnn_arm64_libs"
    "files/.mnnmodels/models--taobao-mnn--qnn_arm64_libs"
)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Helper functions
log_step() { echo -e "${YELLOW}[STEP]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }

# Check prerequisites
check_prerequisites() {
    log_step "Checking prerequisites..."
    
    # Check ADB
    if ! command -v adb &> /dev/null; then
        log_error "ADB not found. Please install Android SDK platform-tools"
        exit 1
    fi
    
    # Check device connection
    local devices=$(adb devices | grep -v "List of devices" | grep -v "^$" | grep -v "unauthorized" | wc -l)
    if [ "$devices" -eq 0 ]; then
        log_error "No authorized Android devices connected"
        exit 1
    fi
    
    # Check app installation
    if ! adb shell pm list packages | grep -q "^package:${PACKAGE_NAME}$"; then
        log_error "App ${PACKAGE_NAME} not installed"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Delete directory and verify immediately
delete_and_verify() {
    local dir="$1"
    log_info "Processing: $dir"
    
    # Check if directory exists
    if ! adb shell "run-as ${PACKAGE_NAME} [ -d '$dir' ]" 2>/dev/null; then
        log_info "Directory does not exist: $dir"
        return 0
    fi
    
    # Count files before deletion
    local files_before=$(adb shell "run-as ${PACKAGE_NAME} find '$dir' -type f 2>/dev/null | wc -l" 2>/dev/null || echo "0")
    log_info "Found $files_before files to delete"
    
    # Delete directory
    if adb shell "run-as ${PACKAGE_NAME} rm -rf '$dir'" 2>/dev/null; then
        log_success "Deleted directory: $dir"
    else
        log_error "Failed to delete directory: $dir"
        return 1
    fi
    
    # Verify deletion immediately
    if adb shell "run-as ${PACKAGE_NAME} [ -d '$dir' ]" 2>/dev/null; then
        log_error "Verification failed: directory still exists"
        return 1
    else
        log_success "Verification passed: directory deleted"
        return 0
    fi
}

# Clean QNN libraries
clean_qnn_libraries() {
    log_step "Cleaning QNN libraries..."
    
    local total_deleted=0
    local failed_dirs=()
    
    for dir in "${CLEANUP_DIRS[@]}"; do
        if delete_and_verify "$dir"; then
            total_deleted=$((total_deleted + 1))
        else
            failed_dirs+=("$dir")
        fi
        echo
    done
    
    # Summary
    if [ ${#failed_dirs[@]} -eq 0 ]; then
        log_success "All directories cleaned successfully ($total_deleted/$total_deleted)"
    else
        log_error "Failed to clean ${#failed_dirs[@]} directories"
        for dir in "${failed_dirs[@]}"; do
            log_error "  - $dir"
        done
    fi
}

# Reset installation flag
reset_installation_flag() {
    log_step "Resetting installation flag..."
    
    if adb shell "run-as ${PACKAGE_NAME} [ -f 'shared_prefs/qnn_libs.xml' ]" 2>/dev/null; then
        if adb shell "run-as ${PACKAGE_NAME} rm -f 'shared_prefs/qnn_libs.xml'" 2>/dev/null; then
            log_success "Installation flag reset"
        else
            log_error "Failed to reset installation flag"
        fi
    else
        log_info "Installation flag file does not exist"
    fi
}

# Check QNN status
check_qnn_status() {
    log_step "Checking QNN status..."
    
    local total_files=0
    local existing_dirs=0
    
    for dir in "${CLEANUP_DIRS[@]}"; do
        if adb shell "run-as ${PACKAGE_NAME} [ -d '$dir' ]" 2>/dev/null; then
            existing_dirs=$((existing_dirs + 1))
            local files_count=$(adb shell "run-as ${PACKAGE_NAME} find '$dir' -type f 2>/dev/null | wc -l" 2>/dev/null || echo "0")
            total_files=$((total_files + files_count))
            log_success "Found $files_count files in $dir"
        fi
    done
    
    # Check installation flag
    local flag_exists=false
    if adb shell "run-as ${PACKAGE_NAME} [ -f 'shared_prefs/qnn_libs.xml' ]" 2>/dev/null; then
        flag_exists=true
        log_success "Installation flag exists"
    else
        log_info "Installation flag does not exist"
    fi
    
    # Status summary
    echo
    if [ "$existing_dirs" -gt 0 ] && [ "$total_files" -gt 0 ] && [ "$flag_exists" = true ]; then
        log_success "QNN setup: COMPLETE ($existing_dirs dirs, $total_files files)"
    elif [ "$existing_dirs" -gt 0 ] && [ "$total_files" -gt 0 ]; then
        log_error "QNN setup: PARTIAL ($existing_dirs dirs, $total_files files, no flag)"
    elif [ "$existing_dirs" -gt 0 ]; then
        log_error "QNN setup: INCOMPLETE ($existing_dirs dirs, no files)"
    else
        log_error "QNN setup: NOT INSTALLED"
    fi
}

# Show usage
show_usage() {
    echo -e "${BLUE}Usage:${NC} $0 <command>"
    echo
    echo -e "${YELLOW}Commands:${NC}"
    echo -e "  ${GREEN}check${NC}  - Check QNN libraries status"
    echo -e "  ${GREEN}clean${NC}  - Clean QNN libraries and reset flag"
    echo
    echo -e "${YELLOW}Examples:${NC}"
    echo -e "  $0 check"
    echo -e "  $0 clean"
}

# Main function
main() {
    if [ $# -eq 0 ]; then
        show_usage
        exit 1
    fi
    
    local command="$1"
    
    case "$command" in
        "check")
            check_prerequisites
            echo
            check_qnn_status
            ;;
        "clean")
            check_prerequisites
            echo
            clean_qnn_libraries
            reset_installation_flag
            echo
            log_success "QNN cleanup completed!"
            ;;
        *)
            log_error "Unknown command: $command"
            echo
            show_usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
