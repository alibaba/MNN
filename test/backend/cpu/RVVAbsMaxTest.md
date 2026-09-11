# RVV AbsMax regression test

Build this standalone test from the repository root on an RVV 1.0 machine:

```sh
g++ -std=c++11 -O3 -march=rv64gcv -pthread \
  -DMNN_RVV_ABSMAX_TEST_MAIN \
  source/backend/cpu/riscv/rvv/MNNAbsMaxFP32.cpp \
  test/backend/cpu/RVVAbsMaxTest.cpp -o rvv_absmax_test
./rvv_absmax_test 1
./rvv_absmax_test 4
```

The test compares every output bit with an independent scalar oracle, checks
output guards, and covers depths 0 through 257, position counts 0/1/3/33,
packs 4/8/16, finite values, negative zero, all-NaN input, infinity, and maxima
before and inside the final tail. Each worker runs 18,576 comparisons with
independent buffers. The runtime layout remains C4; additional packs exercise
the kernel's existing parameter contract.

Cross-compile with the same flags plus `-static` and run under QEMU for VLEN
128, 256, 512 and 1024, for example:

```sh
qemu-riscv64 -cpu max,vlen=256 ./rvv_absmax_test 4
```

Performance must be measured on hardware. Compare the registered scalar
`CoreFunctions::MNNAbsMax`, the original PR kernel, and the repaired kernel
using identical inputs and build flags. Include `realSize=1` with 1024 and
4096 channels, as well as multiple-position inputs. Match the library's
conditional macros when compiling code that accesses `CoreFunctions`.
Warm up first, pin a core, alternate measurement order across independent
processes, retain all runs, and report medians rather than the best run.
