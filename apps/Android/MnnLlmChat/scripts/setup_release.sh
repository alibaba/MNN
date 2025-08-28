#!/bin/bash

# Setup script for release environment
# This script helps users configure the release environment

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

log_info "Setting up release environment..."

# Check if release.config already exists
if [[ -f "scripts/release.config" ]]; then
    log_warning "release.config already exists. Do you want to overwrite it? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        log_info "Setup cancelled."
        exit 0
    fi
fi

# Create release.config from example
if [[ -f "scripts/release.config.example" ]]; then
    cp scripts/release.config.example scripts/release.config
    log_success "Created scripts/release.config from example"
else
    log_error "scripts/release.config.example not found"
    exit 1
fi

log_info "Please edit scripts/release.config with your actual values:"
log_info "1. Signing configuration (for Google Play releases)"
log_info "2. CDN configuration (for Aliyun OSS uploads)"
log_info "3. Google Play configuration (for Google Play uploads)"

log_info ""
log_info "You can edit the file with:"
log_info "  nano scripts/release.config"
log_info "  # or"
log_info "  vim scripts/release.config"
log_info "  # or"
log_info "  code scripts/release.config"

log_info ""
log_info "After editing the configuration, you can run:"
log_info "  source scripts/release.config"
log_info "  ./scripts/release.sh"

log_success "Setup completed!" 