#!/bin/bash

# Test script to verify build commands work correctly
# This script tests the build commands without uploading

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
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

# Check if we're in the right directory
if [[ ! -f "app/build.gradle" ]]; then
    log_error "This script must be run from the project root directory"
    exit 1
fi

log_info "Testing build commands..."

# Test standard debug build
log_info "Testing standard debug build..."
if ./gradlew assembleStandardDebug; then
    log_success "Standard debug build completed successfully"
else
    log_error "Standard debug build failed"
    exit 1
fi

# Test Google Play release build (if signing is configured)
if [[ -n "$KEYSTORE_FILE" && -n "$KEYSTORE_PASSWORD" && -n "$KEY_ALIAS" && -n "$KEY_PASSWORD" ]]; then
    log_info "Testing Google Play release build..."
    if ./gradlew assembleGoogleplayRelease; then
        log_success "Google Play release build completed successfully"
    else
        log_error "Google Play release build failed"
        exit 1
    fi
    
    log_info "Testing Google Play release bundle..."
    if ./gradlew bundleGoogleplayRelease; then
        log_success "Google Play release bundle completed successfully"
    else
        log_error "Google Play release bundle failed"
        exit 1
    fi
else
    log_warning "Skipping Google Play build tests due to missing signing configuration"
fi

log_success "All build tests completed successfully!" 