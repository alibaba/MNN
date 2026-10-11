# TORQ-Tile FP32 subset

Upstream: https://github.com/XUANTIE-RV/torq-tile

Commit: `1e2185f025d8b51d45a560743c4bbe146ecb725c`

License: Apache-2.0, retained in `LICENSE` and the source file headers.

Only the FP32 transposed-A SGEMM (`sgemm_f32_f32t_f32_8x3vl_rvv`), its common
header and assembly helpers are included. No upstream build, tests, scheduling,
allocation, FP16/BF16 or quantized kernels are linked.

Local changes:

- Format the imported sources with MNN's clang-format 17 configuration.
- Query VLEN inside functions, after MNN's RVV capability check; remove dynamic
  namespace-scope initialization. A library load must not execute an RVV CSR read.
- Remove unused FP16/BF16 typedefs from the FP32-only common header so an FP32
  build does not require those compiler scalar types.
- Remove the common header include from the public SGEMM declarations, which
  only need `size_t`.
- Mark the assembly helper functions `always_inline`. They communicate through
  fixed registers and must not become ABI calls in unoptimized builds.

The adapter keeps MNN's existing `(eP,lP,hP) = (16,1,4)` contract. Each C4 block
is submitted as a separate SGEMM, with bias/clamp handled by the adapter. This
preserves compatibility but does not exploit the full 3*VL output-channel tile.
The option is experimental and disabled by default; performance must be measured
on the intended CPU before enabling it in a product.
