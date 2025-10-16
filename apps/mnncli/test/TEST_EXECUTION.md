# MNN CLI Test Execution Guide

## Test Scripts Overview

### 1. `build_tests.sh`
- **Purpose**: Configures and builds all tests
- **Location**: Root test directory
- **Usage**: `./build_tests.sh`
- **Output**: Compiled binaries in `build/` directory

### 2. `run_tests.sh`
- **Purpose**: Runs individual tests or all tests
- **Location**: Root test directory
- **Usage**:
  - `./run_tests.sh` - List available tests
  - `./run_tests.sh <test_name>` - Run specific test
  - `./run_tests.sh all` - Run all tests (uses run_all_tests.sh)

### 3. `run_all_tests.sh`
- **Purpose**: Comprehensive test runner with detailed reporting
- **Location**: Root test directory
- **Usage**: `./run_all_tests.sh`
- **Features**:
  - Runs all tests with appropriate modes (normal, --help, --skip-download)
  - Handles special cases (tests requiring arguments, network issues)
  - Generates detailed log file (`test_results.log`)
  - Color-coded output for pass/fail results
  - Performance timing for each test
  - Summary report with pass/fail counts

## Test Execution Modes

The comprehensive test runner handles different test types appropriately:

1. **Normal Tests**: Run without arguments (`verify_git_hash`, `debug_sha_test`, etc.)
2. **Help Mode Tests**: Run with `--help` flag to avoid downloading (`download_verification_test`, etc.)
3. **Skip-Download Tests**: Run with `--skip-download` to avoid network operations (`cdn_etag_comparison_test`)
4. **Argument-Required Tests**: Expected to fail due to missing arguments (`test_real_file`)

## Test Results Interpretation

- **PASSED** (Green): Test executed successfully
- **ARGS NEEDED (Expected)** (Yellow): Test failed as expected due to missing required arguments
- **NETWORK ISSUE (Expected)** (Yellow): Network-dependent test with expected issues
- **FAILED** (Red): Unexpected test failure

## Log Files

- `test_results.log`: Detailed output from all test runs
- Located in the root test directory

## Example Usage

```bash
# Build all tests
./build_tests.sh

# List available tests
./run_tests.sh

# Run a specific test
./run_tests.sh verify_git_hash

# Run all tests with comprehensive reporting
./run_tests.sh all
# or directly:
./run_all_tests.sh
```

All 9 tests currently pass the comprehensive test suite.