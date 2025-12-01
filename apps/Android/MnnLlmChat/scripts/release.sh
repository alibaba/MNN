#!/bin/bash

# Release script for MnnLlmChat
# This script builds and publishes:
# 1. Standard flavor debug version for CDN upload
# 2. Google Play flavor release APK/AAB for Google Play Store

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="MnnLlmChat"
VERSION_NAME=$(grep "versionName" app/build.gradle | head -1 | sed 's/.*versionName "\(.*\)"/\1/')
VERSION_CODE=$(grep "versionCode" app/build.gradle | head -1 | sed 's/.*versionCode \([0-9]*\)/\1/')
BUILD_DATE=$(date +"%Y%m%d_%H%M%S")

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/app/build"
OUTPUT_DIR="$PROJECT_DIR/release_outputs"
CDN_UPLOAD_DIR="$OUTPUT_DIR/cdn"
GOOGLE_PLAY_DIR="$OUTPUT_DIR/googleplay"

# Environment variables for signing (should be set in CI/CD or locally)
KEYSTORE_FILE="${KEYSTORE_FILE:-}"
KEYSTORE_PASSWORD="${KEYSTORE_PASSWORD:-}"
KEY_ALIAS="${KEY_ALIAS:-}"
KEY_PASSWORD="${KEY_PASSWORD:-}"

# CDN configuration
CDN_ENDPOINT="${CDN_ENDPOINT:-}"
CDN_ACCESS_KEY="${CDN_ACCESS_KEY:-}"
CDN_SECRET_KEY="${CDN_SECRET_KEY:-}"
CDN_BUCKET="${CDN_BUCKET:-}"

# Google Play configuration
GOOGLE_PLAY_SERVICE_ACCOUNT="${GOOGLE_PLAY_SERVICE_ACCOUNT:-}"
GOOGLE_PLAY_PACKAGE_NAME="${GOOGLE_PLAY_PACKAGE_NAME:-com.alibaba.mnnllm.android.googleplay}"

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

check_requirements() {
    log_info "Checking requirements..."
    
    # Check if we're in the right directory
    if [[ ! -f "app/build.gradle" ]]; then
        log_error "This script must be run from the project root directory"
        exit 1
    fi
    
    # Check Java version
    java_version=$(java -version 2>&1 | head -n 1 | cut -d'"' -f2 | cut -d'.' -f1)
    if [[ "$java_version" -lt 11 ]]; then
        log_error "Java 11 or higher is required. Current version: $java_version"
        exit 1
    fi
    
    # Check if Gradle wrapper exists
    if [[ ! -f "gradlew" ]]; then
        log_error "Gradle wrapper not found. Please run 'gradle wrapper' first."
        exit 1
    fi
    
    # Check signing configuration for Google Play
    if [[ -z "$KEYSTORE_FILE" || -z "$KEYSTORE_PASSWORD" || -z "$KEY_ALIAS" || -z "$KEY_PASSWORD" ]]; then
        log_warning "Signing configuration not found. Google Play release will be skipped."
        SKIP_GOOGLE_PLAY=true
    else
        SKIP_GOOGLE_PLAY=false
    fi
    
    log_success "Requirements check completed"
}

clean_build() {
    log_info "Cleaning previous builds..."
    ./gradlew clean
    rm -rf "$OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR" "$CDN_UPLOAD_DIR" "$GOOGLE_PLAY_DIR"
    log_success "Clean completed"
}

build_standard_debug() {
    log_info "Building standard flavor debug version..."
    
    ./gradlew assembleStandardDebug
    
    # Generate version-based filename (replace dots with underscores)
    VERSION_FILENAME=$(echo "$VERSION_NAME" | sed 's/\./_/g')
    APK_FILENAME="mnn_chat_${VERSION_FILENAME}.apk"
    
    # Copy APK to output directory with version-based name
    APK_PATH="$BUILD_DIR/outputs/apk/standard/debug/app-standard-debug.apk"
    if [[ -f "$APK_PATH" ]]; then
        cp "$APK_PATH" "$CDN_UPLOAD_DIR/$APK_FILENAME"
        log_success "Standard debug APK built: $CDN_UPLOAD_DIR/$APK_FILENAME"
    else
        log_error "Standard debug APK not found at $APK_PATH"
        exit 1
    fi
}

build_googleplay_release() {
    if [[ "$SKIP_GOOGLE_PLAY" == "true" ]]; then
        log_warning "Skipping Google Play release build due to missing signing configuration"
        return
    fi
    
    log_info "Building Google Play flavor release version..."
    
    # Build the release APK
    ./gradlew assembleGoogleplayRelease
    
    # Copy APK to output directory
    APK_PATH="$BUILD_DIR/outputs/apk/googleplay/release/app-googleplay-release.apk"
    if [[ -f "$APK_PATH" ]]; then
        cp "$APK_PATH" "$GOOGLE_PLAY_DIR/"
        log_success "Google Play release APK built: $GOOGLE_PLAY_DIR/app-googleplay-release.apk"
    else
        log_error "Google Play release APK not found at $APK_PATH"
        exit 1
    fi
    
    # Also try to build AAB if possible
    log_info "Building Google Play flavor release AAB..."
    ./gradlew bundleGoogleplayRelease
    
    # Copy AAB to output directory
    AAB_PATH="$BUILD_DIR/outputs/bundle/googleplayRelease/app-googleplay-release.aab"
    if [[ -f "$AAB_PATH" ]]; then
        cp "$AAB_PATH" "$GOOGLE_PLAY_DIR/"
        log_success "Google Play release AAB built: $GOOGLE_PLAY_DIR/app-googleplay-release.aab"
    else
        log_warning "Google Play release AAB not found at $AAB_PATH"
    fi
}

upload_to_cdn() {
    if [[ -z "$CDN_ENDPOINT" || -z "$CDN_ACCESS_KEY" || -z "$CDN_SECRET_KEY" || -z "$CDN_BUCKET" ]]; then
        log_warning "CDN configuration not found. Skipping CDN upload."
        return
    fi
    
    log_info "Uploading to CDN..."
    
    # Check if ossutil is available (Aliyun OSS CLI tool)
    if ! command -v ossutil &> /dev/null; then
        log_warning "ossutil not found. Please install it to upload to CDN."
        log_info "You can install ossutil from: https://www.alibabacloud.com/help/en/object-storage-service/latest/ossutil-installation"
        return
    fi
    
    # Configure ossutil
    ossutil config -e "$CDN_ENDPOINT" -i "$CDN_ACCESS_KEY" -k "$CDN_SECRET_KEY"
    
    # Generate version-based filename for upload
    VERSION_FILENAME=$(echo "$VERSION_NAME" | sed 's/\./_/g')
    APK_FILENAME="mnn_chat_${VERSION_FILENAME}.apk"
    
    # Upload APK to CDN
    APK_FILE="$CDN_UPLOAD_DIR/$APK_FILENAME"
    if [[ -f "$APK_FILE" ]]; then
        ossutil cp "$APK_FILE" "oss://$CDN_BUCKET/releases/$VERSION_NAME/$APK_FILENAME"
        log_success "APK uploaded to CDN: oss://$CDN_BUCKET/releases/$VERSION_NAME/$APK_FILENAME"
    else
        log_error "APK file not found for CDN upload: $APK_FILE"
    fi
}

upload_to_google_play() {
    if [[ "$SKIP_GOOGLE_PLAY" == "true" ]]; then
        log_warning "Skipping Google Play upload due to missing signing configuration"
        return
    fi
    
    if [[ -z "$GOOGLE_PLAY_SERVICE_ACCOUNT" ]]; then
        log_warning "Google Play service account not configured. Skipping Google Play upload."
        return
    fi
    
    log_info "Uploading to Google Play..."
    
    # Check if fastlane is available
    if ! command -v fastlane &> /dev/null; then
        log_warning "fastlane not found. Please install it to upload to Google Play."
        log_info "You can install fastlane with: gem install fastlane"
        return
    fi
    
    # Create fastlane configuration if it doesn't exist
    FASTLANE_DIR="$PROJECT_DIR/fastlane"
    if [[ ! -d "$FASTLANE_DIR" ]]; then
        mkdir -p "$FASTLANE_DIR"
        cat > "$FASTLANE_DIR/Appfile" << EOF
json_key_file("$GOOGLE_PLAY_SERVICE_ACCOUNT")
package_name("$GOOGLE_PLAY_PACKAGE_NAME")
EOF
        
        cat > "$FASTLANE_DIR/Fastfile" << EOF
default_platform(:android)

platform :android do
  desc "Upload to Google Play"
  lane :upload do
    upload_to_play_store(
      track: 'internal',
      aab: '../release_outputs/googleplay/app-googleplay-release.aab',
      apk: '../release_outputs/googleplay/app-googleplay-release.apk'
    )
  end
end
EOF
    fi
    
    # Upload to Google Play
    cd "$FASTLANE_DIR"
    fastlane upload
    cd "$PROJECT_DIR"
    
    log_success "App uploaded to Google Play"
}

generate_release_notes() {
    log_info "Generating release notes..."
    
    # Generate version-based filename for documentation
    VERSION_FILENAME=$(echo "$VERSION_NAME" | sed 's/\./_/g')
    APK_FILENAME="mnn_chat_${VERSION_FILENAME}.apk"
    
    RELEASE_NOTES_FILE="$OUTPUT_DIR/release_notes.md"
    cat > "$RELEASE_NOTES_FILE" << EOF
# Release Notes - $PROJECT_NAME v$VERSION_NAME

## Version Information
- **Version Name**: $VERSION_NAME
- **Version Code**: $VERSION_CODE
- **Build Date**: $BUILD_DATE
- **Build Type**: Release

## Build Outputs

### Standard Flavor (Debug)
- **APK**: \`$APK_FILENAME\`
- **Purpose**: CDN distribution
- **Location**: \`$CDN_UPLOAD_DIR/\`

### Google Play Flavor (Release)
- **APK**: \`app-googleplay-release.apk\`
- **AAB**: \`app-googleplay-release.aab\`
- **Purpose**: Google Play Store distribution
- **Location**: \`$GOOGLE_PLAY_DIR/\`

## Build Configuration
- **Min SDK**: 26
- **Target SDK**: 35
- **Compile SDK**: 35
- **ABI**: arm64-v8a

## Notes
- Standard flavor includes debug features and is suitable for testing
- Google Play flavor is optimized for production and follows Google Play guidelines
- Both builds include native libraries and are signed appropriately

EOF
    
    log_success "Release notes generated: $RELEASE_NOTES_FILE"
}

main() {
    log_info "Starting release process for $PROJECT_NAME v$VERSION_NAME"
    
    # Check requirements
    check_requirements
    
    # Clean previous builds
    clean_build
    
    # Build standard debug version
    build_standard_debug
    
    # Build Google Play release version
    build_googleplay_release
    
    # Upload to CDN
    upload_to_cdn
    
    # Upload to Google Play
    upload_to_google_play
    
    # Generate release notes
    generate_release_notes
    
    log_success "Release process completed successfully!"
    log_info "Output directory: $OUTPUT_DIR"
    log_info "CDN uploads: $CDN_UPLOAD_DIR"
    log_info "Google Play uploads: $GOOGLE_PLAY_DIR"
}

# Run main function
main "$@" 