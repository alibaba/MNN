# RISC-V CPU implementations

| Directory | Responsibility |
| --- | --- |
| `common/` | Shared helpers without RVV intrinsics or vendor instructions. |
| `rvv/` | Standard RVV kernels, execution implementations, and fallback registration. |
| `spacemit_ime2/` | SpacemiT IME2 kernels, execution implementations, and vendor registration. |
| `torq-tile/` | TORQ-Tile FP32 packed MatMul adapter and registration, using standard RVV. |

The IME2 implementation can reuse RVV kernels and execution interfaces. Keep vendor instructions and
platform resources in `spacemit_ime2/` so that standard RVV remains independently buildable.

## Build targets

`CMakeLists.txt` separates the implementation into object libraries:

- `MNNRVV`: standard RVV sources, compiled with the base ISA string.
- `MNNSpacemitIme2Runtime`: vendor execution and registration sources, compiled with the base ISA string.
- `MNNSpacemitIme2`: vendor kernels, compiled with the additional `_xsmtvdotii` extension.
- `MNNTorqTile`: optional MNN adapter and registration, using the base ISA and C++11.
- `MNNTorqTileKernels`: the vendored FP32 kernel, using the base ISA and C++17.

The existing build options remain `MNN_USE_RVV` and `MNN_RVV_SPACEMIT_IME2`. The latter selects the IME2
registration source instead of `rvv/MNNRvvFastPathRegistration.cpp`; both sources define the same
registration entry points and must remain mutually exclusive. Standard RVV kernels remain available
as fallbacks in the IME2 build.

Add vendor sources to the explicit runtime or kernel source list according to their ISA requirements.
The vendor low-memory convolution executor is built only when `MNN_LOW_MEMORY` is enabled.

## IME2 ISA configuration

The default `MNN_RVV_MARCH=rv64gcv` is insufficient for the current IME2 implementation because it lacks
`zfh` and `zihintpause`. This build limitation predates the directory reorganization. The
following base ISA was validated on K3 with GCC 15.2.0; use a toolchain that supports `_xsmtvdotii`:

```bash
cmake -S . -B build-k3-ime2 \
  -DMNN_USE_RVV=ON \
  -DMNN_RVV_SPACEMIT_IME2=ON \
  -DMNN_RVV_MARCH=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs_zicbop_zihintpause_zicond_zvbb_zvbc_zvkb
```

Run this command from the repository root and select the base ISA for your target CPU and toolchain.
CMake appends `_xsmtvdotii` only to the `MNNSpacemitIme2` kernel target.

## Experimental TORQ-Tile integration

The MNN adapter and registration live in `torq-tile/`, alongside `spacemit_ime2/`.
The imported TORQ-Tile kernels and license live in `3rd_party/torq-tile/`.

`MNN_RVV_TORQ_TILE` defaults to `OFF` and requires `MNN_USE_RVV=ON` on riscv64.
It replaces the FP32 `MNNPackedMatMul` and `MNNPackedMatMulRemain` functions
after RVV capability detection. Existing packers, `(eP,lP,hP)=(16,1,4)`, C4 output,
thread scheduling and public APIs are unchanged. Empty reductions and E=1 retain
the original RVV implementation. FP16, BF16, quantized kernels and GEMV are not
replaced. The separate registration function coexists with the existing IME2
registration without defining another mutually exclusive entry point.

The source revision, license and local changes are recorded in
[`3rd_party/torq-tile/README.MNN.md`](../../../../3rd_party/torq-tile/README.MNN.md).
Build on an RVV 1.0 target, or supply a RISC-V Linux toolchain file:

```bash
cmake -S . -B build-torq \
  -DCMAKE_BUILD_TYPE=Release \
  -DMNN_USE_RVV=ON -DMNN_RVV_MARCH=rv64gcv \
  -DMNN_RVV_TORQ_TILE=ON -DMNN_RVV_FAST_MATH=OFF \
  -DMNN_BUILD_TEST=ON -DMNN_BUILD_SHARED_LIBS=OFF
cmake --build build-torq -j4
```

Run on the target, checking that each filter reports a nonzero `passed` count:

```bash
./build-torq/run_test.out op/torq_tile_matmul 0 0 1
./build-torq/run_test.out op/matmul 0 0 1
./build-torq/run_test.out op/matmul 0 0 4
./build-torq/run_test.out op/convolution/conv2d 0 0 1
./build-torq/run_test.out op/convolution/conv2d 0 0 4
```

The dedicated test uses the registered functions and covers packed strides,
empty dimensions, all E tails, C4 padding, bias, clamp, nonfinite values and
threaded slices. It is available in static TORQ-Tile builds. Existing MatMul
tests additionally exercise input transposes and the executor's pack/unpack.
For Attention consumers, also enable `MNN_SUPPORT_TRANSFORMER_FUSE` and run
`op/attention` plus a real model smoke test before deployment.

Use a separate `build-rvv` directory with `MNN_RVV_TORQ_TILE=OFF` for correctness
and performance comparison. The compatibility adapter submits one C4 block at
a time and does not use TORQ-Tile's full N tile; it has no guaranteed speedup.
Bias is added after the dot product, so ordinary FP32 rounding differences are
expected. Validate model outputs and benchmark the intended shapes, VLEN and
thread counts on hardware before enabling this experimental path.
