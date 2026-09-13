# RVV INT8 sum tail regression

The general `MNNSumByAxisLForMatmul_A_RVV` path accumulates with a variable VL,
then reduces the entire accumulator at VLMAX. Both additions must preserve
inactive lanes: they contain either zero or contributions from earlier channel
blocks. Using the default tail-agnostic addition can corrupt the final sum.

The test covers `(channels, positions)` values `(16,1)`, `(16,2)`, `(17,1)`,
`(17,2)`, `(17,3)`, `(31,2)`, `(32,2)` and `(33,2)`. It uses the packed
`[LU,E,LP]` input layout with LP=16, distinct values per output position,
poisoned channel padding and output guards. The independent reference is
`channels * (position + 1)`, exactly representable in FP32. The cases include
the separate single-position fast path, both accumulators in the general path,
an unpaired position and partial channel blocks.

In an RVV build with `MNN_BUILD_TEST=ON`, CMake includes the test automatically.
Run it through the normal test suite:

```sh
qemu-riscv64 -cpu rv64,v=true,vlen=128,elen=64,rvv_ta_all_1s=true \
  ./run_test.out backend/cpu/rvv/sum_by_axis_tail
qemu-riscv64 -cpu rv64,v=true,vlen=256,elen=64,rvv_ta_all_1s=true \
  ./run_test.out backend/cpu/rvv/sum_by_axis_tail
```

It also has a standalone entry point that calls the RVV kernel directly:

```sh
riscv64-linux-gnu-g++ -std=c++11 -O2 -static \
  -march=rv64gcv -mabi=lp64d -fno-tree-vectorize -fno-lto \
  -DMNN_USE_RVV -DMNN_RVV_SUM_TAIL_TEST_MAIN \
  -Iinclude -Isource -Ischema/current -I3rd_party/flatbuffers/include \
  test/backend/cpu/RVVSumByAxisTailTest.cpp \
  source/backend/cpu/riscv/rvv/MNNSumByAxisLForMatmul_A.cpp \
  -o rvv-sum-tail-test
qemu-riscv64 -cpu rv64,v=true,vlen=128,elen=64,rvv_ta_all_1s=true ./rvv-sum-tail-test
qemu-riscv64 -cpu rv64,v=true,vlen=256,elen=64,rvv_ta_all_1s=true ./rvv-sum-tail-test
```

Use a compiler supporting RVV 1.0 intrinsics and a QEMU version exposing
`rvv_ta_all_1s`. For dynamic binaries, supply the RISC-V Linux sysroot with
QEMU's `-L`. System-mode QEMU can run the same static binaries in a RISC-V
Linux guest with the same CPU options.

The tail-ones option selects an ISA-permitted behavior for tail-agnostic
instructions. With it disabled, QEMU can preserve the old tail values and hide
the bug. Test both settings; passing with the default setting alone is not a
regression check for this issue.

Standalone validation against master `a03b005cf6f888ebf092e4753840f935827f9c36`:
Zig 0.15.2 / Clang 20.1.2, `-O2`, static RISC-V Linux binaries, Windows QEMU
11.1.0 TCG system emulation and Debian Linux 6.12.107. Clang used
`-fno-vectorize -fno-slp-vectorize -fno-lto` and RV64IMAFDCV. The original
kernel failed 5/8 cases at VLEN=128 and 7/8 at VLEN=256 with tail-ones enabled;
both corrected configurations passed all eight cases. With tail-ones disabled,
both versions passed at both VLENs. For example, C=17/E=2 at VLEN=128 produced
`[-13,-11]` instead of `[17,34]` before the fix.

This validates the specified kernel paths, not a complete operator/model
regression or hardware performance. The GCC and full-suite commands above are
reproduction instructions; the measured A/B results used the standalone
Clang/system-QEMU configuration.
