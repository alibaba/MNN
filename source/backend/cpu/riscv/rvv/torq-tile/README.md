# TORQ-Tile: first FP32 RVV slice

This directory contains one internal FP32 8x4 kernel derived from the MR=8, NR=1
path of [XUANTIE-RV/torq-tile](https://github.com/XUANTIE-RV/torq-tile/tree/1e2185f025d8b51d45a560743c4bbe146ecb725c),
revision `1e2185f025d8b51d45a560743c4bbe146ecb725c`. The original copyright,
Apache-2.0 SPDX identifier and [license](LICENSE) are retained.

The MNN specialization fixes VL=4, alpha=1 and beta=0, preserves the eight-row
FMA loop and consolidates its register operations into one assembly block with
explicit clobbers. It uses standard RVV 1.0 with VLEN >= 128. No generic SGEMM
interface, common headers, template dispatch or other TORQ-Tile kernels are
imported. The kernel and adapter use C++11 with no dynamic global initialization.

With `MNN_RVV_TORQ_TILE=ON`, the adapter selects this kernel only for nonempty
reductions and E divisible by eight (E=8 or E=16 with MNN's current pack mode).
Every other E size and K=0 uses the existing RVV implementation for the whole
call. The adapter retains MNN's packed strides, C4 output, bias and clamp handling.
Bias is added after the dot product, so ordinary FP32 rounding differences are
possible. No other data types or operators are replaced.

The option remains experimental and disabled by default. Target execution,
model correctness, multiple VLEN values and hardware performance remain to be
validated. Cross-compilation and host scalar substitutes are not target validation.
