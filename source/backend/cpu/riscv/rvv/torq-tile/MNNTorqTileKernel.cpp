//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
//
// Derived from XUANTIE-RV/torq-tile, revision
// 1e2185f025d8b51d45a560743c4bbe146ecb725c:
// src/gemm/gemm_f32_f32/tqt_kernel_sgemm_f32_f32_8x3vl_rvv.h
// and sgemm_f32_f32t_f32/tqt_sgemm_f32_f32t_f32_8x3vl_rvv.cpp.
// MNN specialization: retain only MR=8, NR=1, VL=4, alpha=1, beta=0.
// A single asm block keeps fixed-register lifetimes independent of inlining.

#include "MNNTorqTileKernel.hpp"

#if !defined(__riscv) || !defined(__riscv_v)
#error This file requires RISC-V with the vector extension.
#endif

void MNNTorqTileMatMul8x4(float* dst, const float* a, const float* b, size_t depth, size_t aStride) {
    const size_t aStrideBytes = aStride * sizeof(float);
    // ft0-ft7: the eight A scalars for the current K; v0: four B columns.
    // v8/v11/v14/v17/v20/v23/v26/v29: accumulators live until the stores.
    // No next-K loads occur after the final FMA, including when depth == 1.
    asm volatile(
        "vsetivli zero, 4, e32, m1, ta, ma\n\t"
        "vmv.v.i v8, 0\n\t"
        "vmv.v.i v11, 0\n\t"
        "vmv.v.i v14, 0\n\t"
        "vmv.v.i v17, 0\n\t"
        "vmv.v.i v20, 0\n\t"
        "vmv.v.i v23, 0\n\t"
        "vmv.v.i v26, 0\n\t"
        "vmv.v.i v29, 0\n\t"
        "flw ft0, 0(%[a])\n\t"
        "flw ft1, 4(%[a])\n\t"
        "flw ft2, 8(%[a])\n\t"
        "flw ft3, 12(%[a])\n\t"
        "flw ft4, 16(%[a])\n\t"
        "flw ft5, 20(%[a])\n\t"
        "flw ft6, 24(%[a])\n\t"
        "flw ft7, 28(%[a])\n\t"
        "vle32.v v0, (%[b])\n\t"
        "addi %[depth], %[depth], -1\n\t"
        "beqz %[depth], 2f\n\t"
        "1:\n\t"
        "vfmacc.vf v8, ft0, v0\n\t"
        "vfmacc.vf v11, ft1, v0\n\t"
        "vfmacc.vf v14, ft2, v0\n\t"
        "vfmacc.vf v17, ft3, v0\n\t"
        "vfmacc.vf v20, ft4, v0\n\t"
        "vfmacc.vf v23, ft5, v0\n\t"
        "vfmacc.vf v26, ft6, v0\n\t"
        "vfmacc.vf v29, ft7, v0\n\t"
        "addi %[b], %[b], 16\n\t"
        "vle32.v v0, (%[b])\n\t"
        "add %[a], %[a], %[aStride]\n\t"
        "flw ft0, 0(%[a])\n\t"
        "flw ft1, 4(%[a])\n\t"
        "flw ft2, 8(%[a])\n\t"
        "flw ft3, 12(%[a])\n\t"
        "flw ft4, 16(%[a])\n\t"
        "flw ft5, 20(%[a])\n\t"
        "flw ft6, 24(%[a])\n\t"
        "flw ft7, 28(%[a])\n\t"
        "addi %[depth], %[depth], -1\n\t"
        "bnez %[depth], 1b\n\t"
        "2:\n\t"
        "vfmacc.vf v8, ft0, v0\n\t"
        "vfmacc.vf v11, ft1, v0\n\t"
        "vfmacc.vf v14, ft2, v0\n\t"
        "vfmacc.vf v17, ft3, v0\n\t"
        "vfmacc.vf v20, ft4, v0\n\t"
        "vfmacc.vf v23, ft5, v0\n\t"
        "vfmacc.vf v26, ft6, v0\n\t"
        "vfmacc.vf v29, ft7, v0\n\t"
        "vse32.v v8, (%[dst])\n\t"
        "addi %[dst], %[dst], 16\n\t"
        "vse32.v v11, (%[dst])\n\t"
        "addi %[dst], %[dst], 16\n\t"
        "vse32.v v14, (%[dst])\n\t"
        "addi %[dst], %[dst], 16\n\t"
        "vse32.v v17, (%[dst])\n\t"
        "addi %[dst], %[dst], 16\n\t"
        "vse32.v v20, (%[dst])\n\t"
        "addi %[dst], %[dst], 16\n\t"
        "vse32.v v23, (%[dst])\n\t"
        "addi %[dst], %[dst], 16\n\t"
        "vse32.v v26, (%[dst])\n\t"
        "addi %[dst], %[dst], 16\n\t"
        "vse32.v v29, (%[dst])\n\t"
        : [dst] "+&r"(dst), [a] "+&r"(a), [b] "+&r"(b), [depth] "+&r"(depth)
        : [aStride] "r"(aStrideBytes)
        : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7", "v0", "v8", "v11", "v14", "v17", "v20", "v23", "v26",
          "v29", "memory");
}
