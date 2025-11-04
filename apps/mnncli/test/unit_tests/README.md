# Unit Tests - Quick Start

## One-Liner to Run All Tests

```bash
cd apps/mnncli/test/unit_tests && cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug && cmake --build build -j && ./build/model_name_utils_tests
```

## Step-by-Step

### 1. Navigate to tests directory
```bash
cd apps/mnncli/test/unit_tests
```

### 2. Configure (creates build system)
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
```

### 3. Build (compiles tests)
```bash
cmake --build build -j
```

### 4. Run (executes tests)
```bash
./build/model_name_utils_tests
```

## Output Example

```
========================================
Model Name Utils Test Suite
========================================

  GetFullModelId - ModelScope provider ... PASSED
  GetFullModelId - HuggingFace provider ... PASSED
  ...
  GetFullModelId - Special characters ... PASSED

========================================
Test Results:
  Passed:  18
  Failed:  0
========================================
```

## Key Points

- **No main project required**: Tests are completely standalone
- **Fast build**: Only 3 source files compiled (< 1 second)
- **No external dependencies**: Uses C++ standard library only
- **Exit code**: Returns 0 if all pass, 1 if any fail (CI-friendly)

## Common Issues

### Error: "CMake not found"
Install CMake: `brew install cmake`

### Build errors with missing headers
Make sure you're running from the correct directory (`test/unit_tests`)

### Test executable not found
You may have skipped the build step. Run: `cmake --build build -j`

## Notes

- See `README_UNIT_TESTS.md` for detailed documentation
- See `model_name_utils_case_test.cpp` to add new tests
- Build directory can be safely deleted: `rm -rf build`
