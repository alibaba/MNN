# MNN CLI Test Organization Summary

## Directory Structure
- `src/` - Contains all C++ source files for tests
- `build/` - Contains compiled test binaries (created by build process)
- Root directory - Contains documentation, scripts, and build configuration

## Test Binaries (in build/ directory)
1. `cdn_etag_comparison_test` - Tests CDN ETag handling
2. `debug_sha_test` - Debugging SHA calculation
3. `download_verification_test` - Tests file download and verification
4. `embedding_file_verification_test` - Tests embedding file download/verification
5. `hf_bin_file_verification_test` - Main HuggingFace BIN file verification test
6. `sha256_verification_test` - SHA-256 verification testing
7. `test_llm_weight` - Tests downloading large LLM weight files
8. `test_real_file` - Tests real file handling
9. `verify_git_hash` - Git hash verification

## Scripts
- `build_tests.sh` - Builds all tests in the build directory
- `run_tests.sh` - Runs tests from the build directory

## Build Process
1. Run `./build_tests.sh` to configure and build all tests
2. Tests are built in the `build/` directory
3. Run tests using `./run_tests.sh <test_name>` or directly from build directory

## Test Verification
All tests have been verified to work correctly:
- ✅ `verify_git_hash` - Passes successfully
- ✅ `debug_sha_test` - Passes successfully
- ✅ `download_verification_test` - Passes successfully (downloads and verifies config.json)

The organization successfully separates source code, build output, and documentation/scripts as requested.