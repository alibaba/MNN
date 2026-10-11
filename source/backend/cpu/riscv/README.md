# RISC-V CPU implementations

| Directory | Responsibility |
| --- | --- |
| `common/` | Shared helpers without RVV intrinsics or vendor instructions. |
| `rvv/` | Standard RVV kernels, execution implementations, and fallback registration. |
| `spacemit_ime2/` | SpacemiT IME2 kernels, execution implementations, and vendor registration. |

The IME2 implementation can reuse RVV kernels and execution interfaces. Keep vendor instructions and
platform resources in `spacemit_ime2/` so that standard RVV remains independently buildable.

## Build targets

`CMakeLists.txt` separates the implementation into three object libraries:

- `MNNRVV`: standard RVV sources, compiled with the base ISA string.
- `MNNSpacemitIme2Runtime`: vendor execution and registration sources, compiled with the base ISA string.
- `MNNSpacemitIme2`: vendor kernels, compiled with the additional `_xsmtvdotii` extension.

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
