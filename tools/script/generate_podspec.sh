#!/bin/bash
# =============================================================================
# generate_podspec.sh
# Auto-generate MNN.podspec from CMake targets and optionally validate it
#
# Uses the same CMake configuration as production build
#
# Usage:
#   ./tools/script/generate_podspec.sh              # Generate podspec
#   ./tools/script/generate_podspec.sh -c           # Check if podspec is in sync
#   ./tools/script/generate_podspec.sh -v 3.2.0     # Override version
#   ./tools/script/generate_podspec.sh --validate   # Generate + validate syntax
#   ./tools/script/generate_podspec.sh --validate-full  # Generate + full validation (requires Xcode)
#   ./tools/script/generate_podspec.sh --ci         # CI mode (no colors, strict exit)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MNN_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Options
VERSION=""
CHECK_MODE=false
VALIDATE_MODE=false
VALIDATE_FULL=false
CI_MODE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --version|-v)
            VERSION="$2"
            shift 2
            ;;
        --check|-c)
            CHECK_MODE=true
            shift
            ;;
        --validate)
            VALIDATE_MODE=true
            shift
            ;;
        --validate-full)
            VALIDATE_FULL=true
            VALIDATE_MODE=true
            shift
            ;;
        --ci)
            CI_MODE=true
            RED=''
            GREEN=''
            YELLOW=''
            BLUE=''
            NC=''
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Generate MNN.podspec from CMake targets"
            echo "Version defaults to MNN_VERSION from CMakeLists.txt"
            echo ""
            echo "Options:"
            echo "  --check, -c             Check if podspec is in sync (for CI)"
            echo "  --version, -v VERSION   Override podspec version"
            echo "  --validate              Generate + validate syntax"
            echo "  --validate-full         Generate + full validation (requires Xcode)"
            echo "  --ci                    CI mode (no colors, strict exit)"
            echo "  --help, -h              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Helper functions for logging
log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[PASS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()   { echo -e "${RED}[FAIL]${NC} $1"; }

# =============================================================================
# Check mode: backup current podspec first
# =============================================================================
if [ "$CHECK_MODE" = true ]; then
    echo "=========================================="
    echo "Checking MNN.podspec is in sync"
    echo "=========================================="

    if [ ! -f "$MNN_ROOT/MNN.podspec" ]; then
        echo "❌ Error: MNN.podspec not found!"
        echo ""
        echo "Please run: ./tools/script/generate_podspec.sh"
        exit 1
    fi

    cp "$MNN_ROOT/MNN.podspec" "$MNN_ROOT/MNN.podspec.backup"
else
    echo "=========================================="
    echo "Generating MNN.podspec from CMake targets"
    echo "=========================================="
fi

echo "MNN Root: $MNN_ROOT"
if [ -n "$VERSION" ]; then
    echo "Version: $VERSION (override)"
else
    echo "Version: (from MNN_VERSION in CMakeLists.txt)"
fi
echo ""

# =============================================================================
# CMake configuration
# =============================================================================
BUILD_DIR="$MNN_ROOT/build_podspec_gen"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# COMMON flags (same as online build script)
COMMON_FLAGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_TOOLCHAIN_FILE="$MNN_ROOT/cmake/ios.toolchain.cmake"
    -DENABLE_BITCODE=0
    -DMNN_AAPL_FMWK=ON
    -DMNN_SEP_BUILD=OFF
    -DMNN_BUILD_SHARED_LIBS=OFF
    -DMNN_USE_THREAD_POOL=OFF
    -DMNN_KLEIDIAI=OFF
    -DMNN_REDUCE_SIZE=OFF
)

# EXTRA flags (same as online build script)
EXTRA_FLAGS=(
    -DCMAKE_SYSTEM_NAME=iOS
    -DMNN_BUILD_LLM=ON
    -DMNN_SUPPORT_TRANSFORMER_FUSE=ON
    -DMNN_LOW_MEMORY=ON
    -DMNN_BUILD_AUDIO=OFF
    -DLLM_SUPPORT_VISION=OFF
    -DMNN_BUILD_OPENCV=ON
    -DMNN_IMGCODECS=OFF
    -DMNN_SME2=OFF
    -DLLM_SUPPORT_HTTP_RESOURCE=OFF
)

# iOS arm64 flags (same as online build script - NO CoreML!)
IOS_FLAGS=(
    -DMNN_METAL=ON
    -DARCHS="arm64"
    # Optimization flags same as online build
    '-DCMAKE_CXX_FLAGS_RELEASE=-Oz'
    '-DCMAKE_C_FLAGS_RELEASE=-Oz'
)

# Podspec generation flags
PODSPEC_FLAGS=(
    -DMNN_GENERATE_PODSPEC=ON
)
if [ -n "$VERSION" ]; then
    PODSPEC_FLAGS+=(-DMNN_PODSPEC_VERSION="$VERSION")
fi

echo "Configuring CMake..."
cmake "$MNN_ROOT" \
    "${COMMON_FLAGS[@]}" \
    "${EXTRA_FLAGS[@]}" \
    "${IOS_FLAGS[@]}" \
    "${PODSPEC_FLAGS[@]}" \
    > /dev/null 2>&1

# Cleanup
cd "$MNN_ROOT"
rm -rf "$BUILD_DIR"

# =============================================================================
# Check mode: compare and report
# =============================================================================
if [ "$CHECK_MODE" = true ]; then
    echo ""
    if diff -q "$MNN_ROOT/MNN.podspec.backup" "$MNN_ROOT/MNN.podspec" > /dev/null 2>&1; then
        echo "✅ MNN.podspec is in sync!"
        rm -f "$MNN_ROOT/MNN.podspec.backup"
        exit 0
    else
        echo "❌ MNN.podspec is out of sync!"
        echo ""
        echo "Differences:"
        echo "=========================================="
        diff -u "$MNN_ROOT/MNN.podspec.backup" "$MNN_ROOT/MNN.podspec" | head -30 || true
        echo "=========================================="
        echo ""
        echo "To fix, run:"
        echo "  ./tools/script/generate_podspec.sh"
        echo "  git add MNN.podspec"
        echo "  git commit -m 'Update MNN.podspec'"

        # Restore original
        mv "$MNN_ROOT/MNN.podspec.backup" "$MNN_ROOT/MNN.podspec"
        exit 1
    fi
fi

# =============================================================================
# Validation functions
# =============================================================================

# Check if podspec file exists
validate_file_exists() {
    log_info "Checking podspec file exists..."
    
    if [ ! -f "$MNN_ROOT/MNN.podspec" ]; then
        log_error "MNN.podspec not found at: $MNN_ROOT/MNN.podspec"
        return 1
    fi
    
    log_success "Podspec file exists: $MNN_ROOT/MNN.podspec"
    return 0
}

# Validate podspec syntax
validate_syntax() {
    log_info "Validating podspec syntax..."
    
    if ! command -v pod &> /dev/null; then
        log_warning "CocoaPods not installed, skipping syntax validation"
        echo "  Install CocoaPods to enable validation: sudo gem install cocoapods"
        return 0
    fi
    
    # Use pod spec lint for quick syntax check (no download/compile)
    if pod spec lint "$MNN_ROOT/MNN.podspec" --quick --no-clean --allow-warnings 2>&1; then
        log_success "Podspec syntax is valid"
        return 0
    else
        log_error "Podspec syntax validation failed"
        return 1
    fi
}

# Validate file references in podspec
validate_file_references() {
    log_info "Checking file references..."
    
    local errors=0
    
    # Extract source_files patterns and check if they exist
    while IFS= read -r pattern; do
        # Remove quotes and commas
        pattern=$(echo "$pattern" | sed "s/['',]//g" | xargs)
        
        # Skip empty lines and comments
        if [ -z "$pattern" ] || [[ "$pattern" == \#* ]]; then
            continue
        fi
        
        # Check if pattern contains glob
        if [[ "$pattern" == *"*"* ]]; then
            # Expand glob and check if any files match
            expanded=$(eval echo "$pattern" 2>/dev/null)
            if [ -z "$expanded" ]; then
                log_warning "Pattern matches no files: $pattern"
            fi
        else
            # Check if file exists
            if [ ! -e "$MNN_ROOT/$pattern" ]; then
                log_error "File not found: $pattern"
                errors=$((errors + 1))
            fi
        fi
    done < <(grep -E "^\s*'" "$MNN_ROOT/MNN.podspec" | grep -v "^\s*#" | head -100)
    
    if [ $errors -eq 0 ]; then
        log_success "File references are valid"
        return 0
    else
        log_error "Found $errors invalid file references"
        return 1
    fi
}

# Full validation using pod lib lint (requires Xcode)
validate_full() {
    log_info "Running full validation (this may take a while)..."
    
    if ! command -v pod &> /dev/null; then
        log_error "CocoaPods not installed"
        echo "  Install CocoaPods to enable full validation: sudo gem install cocoapods"
        return 1
    fi
    
    if [[ "$(uname)" != "Darwin" ]]; then
        log_warning "Full validation requires macOS, skipping"
        return 0
    fi
    
    # Run pod lib lint
    if pod lib lint "$MNN_ROOT/MNN.podspec" --allow-warnings --verbose 2>&1; then
        log_success "Full validation passed"
        return 0
    else
        log_error "Full validation failed"
        return 1
    fi
}

# =============================================================================
# Validation mode: run validation after generation
# =============================================================================
if [ "$VALIDATE_MODE" = true ]; then
    echo ""
    echo "=========================================="
    echo "Validating MNN.podspec"
    echo "=========================================="
    
    VALIDATE_FAILED=0
    
    # Step 1: Check file exists
    if ! validate_file_exists; then
        VALIDATE_FAILED=1
    fi
    
    # Step 2: Validate syntax (only if file exists)
    if [ $VALIDATE_FAILED -eq 0 ]; then
        if ! validate_syntax; then
            VALIDATE_FAILED=1
        fi
    fi
    
    # Step 3: Validate file references
    if ! validate_file_references; then
        VALIDATE_FAILED=1
    fi
    
    # Step 4: Full validation (if requested)
    if [ "$VALIDATE_FULL" = true ]; then
        if ! validate_full; then
            VALIDATE_FAILED=1
        fi
    fi
    
    # Report results
    echo ""
    echo "=========================================="
    if [ $VALIDATE_FAILED -eq 0 ]; then
        log_success "All validations passed!"
        echo "=========================================="
    else
        log_error "Validation failed!"
        echo "=========================================="
        echo ""
        echo "To fix issues:"
        echo "  1. Review the errors above"
        echo "  2. Regenerate podspec if needed"
        echo "  3. Commit the updated podspec:"
        echo "     git add MNN.podspec"
        echo "     git commit -m 'Update MNN.podspec'"
        exit 1
    fi
fi

# =============================================================================
# Normal mode: show success message
# =============================================================================
if [ "$VALIDATE_MODE" = false ] && [ "$CHECK_MODE" = false ]; then
    echo ""
    echo "=========================================="
    echo "Done! Generated: $MNN_ROOT/MNN.podspec"
    echo "=========================================="
    echo ""
    echo "To validate the generated podspec:"
    echo "  ./tools/script/generate_podspec.sh --validate"
    echo ""
    echo "For full validation (requires Xcode):"
    echo "  ./tools/script/generate_podspec.sh --validate-full"
fi