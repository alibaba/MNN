The quadratic fuzzer generates long sequences of repeated characters, such as `<?x<?x<?x<?x...`,
to detect quadratic complexity performance issues.

To build and run the quadratic fuzzer:

```bash
mkdir build-fuzz
cd build-fuzz
cmake -DCMARK_FUZZ_QUADRATIC=ON -DCMAKE_C_COMPILER=$(which clang) -DCMAKE_CXX_COMPILER=$(which clang++) -DCMAKE_BUILD_TYPE=Release ..
make
../fuzz/fuzzloop.sh
```
