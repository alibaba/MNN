//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if !defined(__riscv) || !defined(__riscv_v)
#error This file must be compiled for riscv, riscv_vector.
#endif // Architectural features check.

#include <stddef.h>

#include "src/common/tqt_common.h"

/// Register allocation for 8x3vl GEMM kernel:
///   ft0-ft7: A[0-7]
///   v0-v2:   B matrix columns (3 vector registers)
///   v8-v31:  Accumulator matrix D[0-7][0-2] (24 vector registers)

/// Macro to initialize a single vector register to zero
#define TQT_VZERO_FP32(reg_num) asm volatile("vmv.v.i v" #reg_num ", 0\n\t" : : : "v" #reg_num);

/// Initialize accumulator registers to zero
template <size_t MR, size_t NR>
static inline __attribute__((always_inline)) void tqt_init_zero_kernel_sgemm_f32_f32_8x3vl_rvv(size_t vl) {
    static_assert(MR >= 1 && MR <= 8, "MR must be in range [1, 8]");
    static_assert(NR >= 1 && NR <= 3, "NR must be in range [1, 3]");

    asm volatile("vsetvli zero, %0, e32, m1, ta, ma\n\t" : : "r"(vl) :);

    // Row 0: v8, v9, v10
    if constexpr (MR >= 1) {
        if constexpr (NR >= 1)
            TQT_VZERO_FP32(8);
        if constexpr (NR >= 2)
            TQT_VZERO_FP32(9);
        if constexpr (NR >= 3)
            TQT_VZERO_FP32(10);
    }
    // Row 1: v11, v12, v13
    if constexpr (MR >= 2) {
        if constexpr (NR >= 1)
            TQT_VZERO_FP32(11);
        if constexpr (NR >= 2)
            TQT_VZERO_FP32(12);
        if constexpr (NR >= 3)
            TQT_VZERO_FP32(13);
    }
    // Row 2: v14, v15, v16
    if constexpr (MR >= 3) {
        if constexpr (NR >= 1)
            TQT_VZERO_FP32(14);
        if constexpr (NR >= 2)
            TQT_VZERO_FP32(15);
        if constexpr (NR >= 3)
            TQT_VZERO_FP32(16);
    }
    // Row 3: v17, v18, v19
    if constexpr (MR >= 4) {
        if constexpr (NR >= 1)
            TQT_VZERO_FP32(17);
        if constexpr (NR >= 2)
            TQT_VZERO_FP32(18);
        if constexpr (NR >= 3)
            TQT_VZERO_FP32(19);
    }
    // Row 4: v20, v21, v22
    if constexpr (MR >= 5) {
        if constexpr (NR >= 1)
            TQT_VZERO_FP32(20);
        if constexpr (NR >= 2)
            TQT_VZERO_FP32(21);
        if constexpr (NR >= 3)
            TQT_VZERO_FP32(22);
    }
    // Row 5: v23, v24, v25
    if constexpr (MR >= 6) {
        if constexpr (NR >= 1)
            TQT_VZERO_FP32(23);
        if constexpr (NR >= 2)
            TQT_VZERO_FP32(24);
        if constexpr (NR >= 3)
            TQT_VZERO_FP32(25);
    }
    // Row 6: v26, v27, v28
    if constexpr (MR >= 7) {
        if constexpr (NR >= 1)
            TQT_VZERO_FP32(26);
        if constexpr (NR >= 2)
            TQT_VZERO_FP32(27);
        if constexpr (NR >= 3)
            TQT_VZERO_FP32(28);
    }
    // Row 7: v29, v30, v31
    if constexpr (MR >= 8) {
        if constexpr (NR >= 1)
            TQT_VZERO_FP32(29);
        if constexpr (NR >= 2)
            TQT_VZERO_FP32(30);
        if constexpr (NR >= 3)
            TQT_VZERO_FP32(31);
    }
}
/// Load A matrix elements (scalars) into floating-point registers
template <size_t MR>
static inline __attribute__((always_inline)) void tqt_load_a_kernel_sgemm_f32_f32_8x3vl_rvv(const float* A,
                                                                                            size_t stride_row) {
    static_assert(MR >= 1 && MR <= 8, "MR must be in range [1, 8]");

    if constexpr (MR >= 1)
        asm volatile("flw ft0, 0(%0)" : : "r"(A + 0 * stride_row) : "ft0", "memory");
    if constexpr (MR >= 2)
        asm volatile("flw ft1, 0(%0)" : : "r"(A + 1 * stride_row) : "ft1", "memory");
    if constexpr (MR >= 3)
        asm volatile("flw ft2, 0(%0)" : : "r"(A + 2 * stride_row) : "ft2", "memory");
    if constexpr (MR >= 4)
        asm volatile("flw ft3, 0(%0)" : : "r"(A + 3 * stride_row) : "ft3", "memory");
    if constexpr (MR >= 5)
        asm volatile("flw ft4, 0(%0)" : : "r"(A + 4 * stride_row) : "ft4", "memory");
    if constexpr (MR >= 6)
        asm volatile("flw ft5, 0(%0)" : : "r"(A + 5 * stride_row) : "ft5", "memory");
    if constexpr (MR >= 7)
        asm volatile("flw ft6, 0(%0)" : : "r"(A + 6 * stride_row) : "ft6", "memory");
    if constexpr (MR >= 8)
        asm volatile("flw ft7, 0(%0)" : : "r"(A + 7 * stride_row) : "ft7", "memory");
}

/// Load B matrix columns (vectors) into vector registers
template <size_t NR>
static inline __attribute__((always_inline)) void tqt_load_b_kernel_sgemm_f32_f32_8x3vl_rvv(size_t vl, const float* B,
                                                                                            size_t stride_col) {
    static_assert(NR >= 1 && NR <= 3, "NR must be in range [1, 3]");

    asm volatile("vsetvli zero, %0, e32, m1, ta, ma" : : "r"(vl) :);

    if constexpr (NR >= 1)
        asm volatile("vle32.v v0, (%0)" : : "r"(B + 0 * stride_col) : "v0", "memory");
    if constexpr (NR >= 2)
        asm volatile("vle32.v v1, (%0)" : : "r"(B + 1 * stride_col) : "v1", "memory");
    if constexpr (NR >= 3)
        asm volatile("vle32.v v2, (%0)" : : "r"(B + 2 * stride_col) : "v2", "memory");
}

/// Load transposed B matrix columns (vectors) into vector registers
template <size_t NR>
static inline __attribute__((always_inline)) void tqt_load_bt_kernel_sgemm_f32_f32_8x3vl_rvv(size_t vl, const float* B,
                                                                                             size_t stride_row) {
    static_assert(NR >= 1 && NR <= 3, "NR must be in range [1, 3]");

    size_t stride_bytes = stride_row * sizeof(float);
    asm volatile("vsetvli zero, %0, e32, m1, ta, ma" : : "r"(vl) :);

    if constexpr (NR >= 1)
        asm volatile("vlse32.v v0, (%0), %1" : : "r"(B + 0 * vl * stride_row), "r"(stride_bytes) : "v0", "memory");
    if constexpr (NR >= 2)
        asm volatile("vlse32.v v1, (%0), %1" : : "r"(B + 1 * vl * stride_row), "r"(stride_bytes) : "v1", "memory");
    if constexpr (NR >= 3)
        asm volatile("vlse32.v v2, (%0), %1" : : "r"(B + 2 * vl * stride_row), "r"(stride_bytes) : "v2", "memory");
}

/// Store accumulator registers to D matrix
template <size_t MR, size_t NR>
static inline __attribute__((always_inline)) void
tqt_store_kernel_sgemm_f32_f32_8x3vl_rvv(size_t vl, float* D, size_t stride_row, size_t stride_col) {
    static_assert(MR >= 1 && MR <= 8, "MR must be in range [1, 8]");
    static_assert(NR >= 1 && NR <= 3, "NR must be in range [1, 3]");

    asm volatile("vsetvli zero, %0, e32, m1, ta, ma" : : "r"(vl) :);

    if constexpr (MR >= 1) {
        float* d_row_0 = D + 0 * stride_row;
        if constexpr (NR >= 1)
            asm volatile("vse32.v v8, (%0)" : : "r"(d_row_0 + 0 * stride_col) : "memory");
        if constexpr (NR >= 2)
            asm volatile("vse32.v v9, (%0)" : : "r"(d_row_0 + 1 * stride_col) : "memory");
        if constexpr (NR >= 3)
            asm volatile("vse32.v v10, (%0)" : : "r"(d_row_0 + 2 * stride_col) : "memory");
    }
    if constexpr (MR >= 2) {
        float* d_row_1 = D + 1 * stride_row;
        if constexpr (NR >= 1)
            asm volatile("vse32.v v11, (%0)" : : "r"(d_row_1 + 0 * stride_col) : "memory");
        if constexpr (NR >= 2)
            asm volatile("vse32.v v12, (%0)" : : "r"(d_row_1 + 1 * stride_col) : "memory");
        if constexpr (NR >= 3)
            asm volatile("vse32.v v13, (%0)" : : "r"(d_row_1 + 2 * stride_col) : "memory");
    }
    if constexpr (MR >= 3) {
        float* d_row_2 = D + 2 * stride_row;
        if constexpr (NR >= 1)
            asm volatile("vse32.v v14, (%0)" : : "r"(d_row_2 + 0 * stride_col) : "memory");
        if constexpr (NR >= 2)
            asm volatile("vse32.v v15, (%0)" : : "r"(d_row_2 + 1 * stride_col) : "memory");
        if constexpr (NR >= 3)
            asm volatile("vse32.v v16, (%0)" : : "r"(d_row_2 + 2 * stride_col) : "memory");
    }
    if constexpr (MR >= 4) {
        float* d_row_3 = D + 3 * stride_row;
        if constexpr (NR >= 1)
            asm volatile("vse32.v v17, (%0)" : : "r"(d_row_3 + 0 * stride_col) : "memory");
        if constexpr (NR >= 2)
            asm volatile("vse32.v v18, (%0)" : : "r"(d_row_3 + 1 * stride_col) : "memory");
        if constexpr (NR >= 3)
            asm volatile("vse32.v v19, (%0)" : : "r"(d_row_3 + 2 * stride_col) : "memory");
    }
    if constexpr (MR >= 5) {
        float* d_row_4 = D + 4 * stride_row;
        if constexpr (NR >= 1)
            asm volatile("vse32.v v20, (%0)" : : "r"(d_row_4 + 0 * stride_col) : "memory");
        if constexpr (NR >= 2)
            asm volatile("vse32.v v21, (%0)" : : "r"(d_row_4 + 1 * stride_col) : "memory");
        if constexpr (NR >= 3)
            asm volatile("vse32.v v22, (%0)" : : "r"(d_row_4 + 2 * stride_col) : "memory");
    }
    if constexpr (MR >= 6) {
        float* d_row_5 = D + 5 * stride_row;
        if constexpr (NR >= 1)
            asm volatile("vse32.v v23, (%0)" : : "r"(d_row_5 + 0 * stride_col) : "memory");
        if constexpr (NR >= 2)
            asm volatile("vse32.v v24, (%0)" : : "r"(d_row_5 + 1 * stride_col) : "memory");
        if constexpr (NR >= 3)
            asm volatile("vse32.v v25, (%0)" : : "r"(d_row_5 + 2 * stride_col) : "memory");
    }
    if constexpr (MR >= 7) {
        float* d_row_6 = D + 6 * stride_row;
        if constexpr (NR >= 1)
            asm volatile("vse32.v v26, (%0)" : : "r"(d_row_6 + 0 * stride_col) : "memory");
        if constexpr (NR >= 2)
            asm volatile("vse32.v v27, (%0)" : : "r"(d_row_6 + 1 * stride_col) : "memory");
        if constexpr (NR >= 3)
            asm volatile("vse32.v v28, (%0)" : : "r"(d_row_6 + 2 * stride_col) : "memory");
    }
    if constexpr (MR >= 8) {
        float* d_row_7 = D + 7 * stride_row;
        if constexpr (NR >= 1)
            asm volatile("vse32.v v29, (%0)" : : "r"(d_row_7 + 0 * stride_col) : "memory");
        if constexpr (NR >= 2)
            asm volatile("vse32.v v30, (%0)" : : "r"(d_row_7 + 1 * stride_col) : "memory");
        if constexpr (NR >= 3)
            asm volatile("vse32.v v31, (%0)" : : "r"(d_row_7 + 2 * stride_col) : "memory");
    }
}

/// Fused load + vfmacc kernel (A*B) with column-first ordering and software pipelining.
/// B is loaded with vle32 (contiguous). Used for packed and non-packed B variants.
/// Computes current iteration's MR*NR vfmacc operations while prefetching
/// next iteration's A scalars (into ft0-ft7) and B vectors (into v0-v2).
/// Assumes ft0-ft7 and v0-v2 already hold the current iteration's data.
template <size_t MR, size_t NR>
static inline __attribute__((always_inline)) void
tqt_fused_load_vfmacc_axb_kernel_sgemm_f32_f32_8x3vl_rvv(const float* A_next, size_t a_stride_row, const float* B_next,
                                                         size_t b_col_stride) {
    static_assert(MR >= 1 && MR <= 8, "MR must be in range [1, 8]");
    static_assert(NR >= 1 && NR <= 3, "NR must be in range [1, 3]");

    // =================================================================
    // Column 0: all v0 uses
    // =================================================================
    if constexpr (MR >= 1)
        asm volatile("vfmacc.vf v8,  ft0, v0" ::: "v8");
    if constexpr (MR >= 2)
        asm volatile("vfmacc.vf v11, ft1, v0" ::: "v11");
    if constexpr (MR >= 3)
        asm volatile("vfmacc.vf v14, ft2, v0" ::: "v14");
    if constexpr (MR >= 4)
        asm volatile("vfmacc.vf v17, ft3, v0" ::: "v17");
    if constexpr (MR >= 5)
        asm volatile("vfmacc.vf v20, ft4, v0" ::: "v20");
    if constexpr (MR >= 6)
        asm volatile("vfmacc.vf v23, ft5, v0" ::: "v23");
    if constexpr (MR >= 7)
        asm volatile("vfmacc.vf v26, ft6, v0" ::: "v26");
    if constexpr (MR >= 8)
        asm volatile("vfmacc.vf v29, ft7, v0" ::: "v29");
    // v0 freed → prefetch next B col 0
    asm volatile("vle32.v v0, (%0)" : : "r"(B_next) : "v0", "memory");

    // =================================================================
    // Column 1: all v1 uses (NR >= 2)
    // =================================================================
    if constexpr (NR >= 2) {
        if constexpr (MR >= 1)
            asm volatile("vfmacc.vf v9,  ft0, v1" ::: "v9");
        if constexpr (MR >= 2)
            asm volatile("vfmacc.vf v12, ft1, v1" ::: "v12");
        if constexpr (MR >= 3)
            asm volatile("vfmacc.vf v15, ft2, v1" ::: "v15");
        if constexpr (MR >= 4)
            asm volatile("vfmacc.vf v18, ft3, v1" ::: "v18");
        if constexpr (MR >= 5)
            asm volatile("vfmacc.vf v21, ft4, v1" ::: "v21");
        if constexpr (MR >= 6)
            asm volatile("vfmacc.vf v24, ft5, v1" ::: "v24");
        if constexpr (MR >= 7)
            asm volatile("vfmacc.vf v27, ft6, v1" ::: "v27");
        if constexpr (MR >= 8)
            asm volatile("vfmacc.vf v30, ft7, v1" ::: "v30");
        // v1 freed → prefetch next B col 1
        asm volatile("vle32.v v1, (%0)" : : "r"(B_next + b_col_stride) : "v1", "memory");
    }

    // =================================================================
    // Last column: vfmacc interleaved with next iteration's A loads.
    // After each vfmacc using ftX, ftX is freed and reloaded for next K.
    // =================================================================
    if constexpr (NR == 3) {
        // Column 2: vfmacc with v2, interleaved with A prefetch
        if constexpr (MR >= 1) {
            asm volatile("vfmacc.vf v10, ft0, v2" ::: "v10");
            asm volatile("flw ft0, 0(%0)" : : "r"(A_next + 0 * a_stride_row) : "ft0", "memory");
        }
        if constexpr (MR >= 2) {
            asm volatile("vfmacc.vf v13, ft1, v2" ::: "v13");
            asm volatile("flw ft1, 0(%0)" : : "r"(A_next + 1 * a_stride_row) : "ft1", "memory");
        }
        if constexpr (MR >= 3) {
            asm volatile("vfmacc.vf v16, ft2, v2" ::: "v16");
            asm volatile("flw ft2, 0(%0)" : : "r"(A_next + 2 * a_stride_row) : "ft2", "memory");
        }
        if constexpr (MR >= 4) {
            asm volatile("vfmacc.vf v19, ft3, v2" ::: "v19");
            asm volatile("flw ft3, 0(%0)" : : "r"(A_next + 3 * a_stride_row) : "ft3", "memory");
        }
        if constexpr (MR >= 5) {
            asm volatile("vfmacc.vf v22, ft4, v2" ::: "v22");
            asm volatile("flw ft4, 0(%0)" : : "r"(A_next + 4 * a_stride_row) : "ft4", "memory");
        }
        if constexpr (MR >= 6) {
            asm volatile("vfmacc.vf v25, ft5, v2" ::: "v25");
            asm volatile("flw ft5, 0(%0)" : : "r"(A_next + 5 * a_stride_row) : "ft5", "memory");
        }
        if constexpr (MR >= 7) {
            asm volatile("vfmacc.vf v28, ft6, v2" ::: "v28");
            asm volatile("flw ft6, 0(%0)" : : "r"(A_next + 6 * a_stride_row) : "ft6", "memory");
        }
        if constexpr (MR >= 8) {
            asm volatile("vfmacc.vf v31, ft7, v2" ::: "v31");
            asm volatile("flw ft7, 0(%0)" : : "r"(A_next + 7 * a_stride_row) : "ft7", "memory");
        }
        // v2 freed → prefetch next B col 2
        asm volatile("vle32.v v2, (%0)" : : "r"(B_next + 2 * b_col_stride) : "v2", "memory");

    } else {
        // NR=1 or NR=2: last vfmacc column already emitted above.
        // A prefetch: load sequentially (no vfmacc to interleave with)
        if constexpr (MR >= 1)
            asm volatile("flw ft0, 0(%0)" : : "r"(A_next + 0 * a_stride_row) : "ft0", "memory");
        if constexpr (MR >= 2)
            asm volatile("flw ft1, 0(%0)" : : "r"(A_next + 1 * a_stride_row) : "ft1", "memory");
        if constexpr (MR >= 3)
            asm volatile("flw ft2, 0(%0)" : : "r"(A_next + 2 * a_stride_row) : "ft2", "memory");
        if constexpr (MR >= 4)
            asm volatile("flw ft3, 0(%0)" : : "r"(A_next + 3 * a_stride_row) : "ft3", "memory");
        if constexpr (MR >= 5)
            asm volatile("flw ft4, 0(%0)" : : "r"(A_next + 4 * a_stride_row) : "ft4", "memory");
        if constexpr (MR >= 6)
            asm volatile("flw ft5, 0(%0)" : : "r"(A_next + 5 * a_stride_row) : "ft5", "memory");
        if constexpr (MR >= 7)
            asm volatile("flw ft6, 0(%0)" : : "r"(A_next + 6 * a_stride_row) : "ft6", "memory");
        if constexpr (MR >= 8)
            asm volatile("flw ft7, 0(%0)" : : "r"(A_next + 7 * a_stride_row) : "ft7", "memory");
    }
}

/// Fused load + vfmacc kernel (A*B^T) with column-first ordering and software pipelining.
/// B is loaded with vlse32 (strided). Used for transposed B variants.
/// b_stride_row is the row stride of transposed B matrix (= ldb).
template <size_t MR, size_t NR>
static inline __attribute__((always_inline)) void
tqt_fused_load_vfmacc_axbt_kernel_sgemm_f32_f32_8x3vl_rvv(size_t vl, const float* A_next, size_t a_stride_row,
                                                          const float* B_next, size_t b_stride_row) {
    static_assert(MR >= 1 && MR <= 8, "MR must be in range [1, 8]");
    static_assert(NR >= 1 && NR <= 3, "NR must be in range [1, 3]");
    size_t stride_bytes = b_stride_row * sizeof(float);

    // =================================================================
    // Column 0: all v0 uses
    // =================================================================
    if constexpr (MR >= 1)
        asm volatile("vfmacc.vf v8,  ft0, v0" ::: "v8");
    if constexpr (MR >= 2)
        asm volatile("vfmacc.vf v11, ft1, v0" ::: "v11");
    if constexpr (MR >= 3)
        asm volatile("vfmacc.vf v14, ft2, v0" ::: "v14");
    if constexpr (MR >= 4)
        asm volatile("vfmacc.vf v17, ft3, v0" ::: "v17");
    if constexpr (MR >= 5)
        asm volatile("vfmacc.vf v20, ft4, v0" ::: "v20");
    if constexpr (MR >= 6)
        asm volatile("vfmacc.vf v23, ft5, v0" ::: "v23");
    if constexpr (MR >= 7)
        asm volatile("vfmacc.vf v26, ft6, v0" ::: "v26");
    if constexpr (MR >= 8)
        asm volatile("vfmacc.vf v29, ft7, v0" ::: "v29");
    // v0 freed → prefetch next B col 0 (strided)
    asm volatile("vlse32.v v0, (%0), %1" : : "r"(B_next + 0 * vl * b_stride_row), "r"(stride_bytes) : "v0", "memory");

    // =================================================================
    // Column 1: all v1 uses (NR >= 2)
    // =================================================================
    if constexpr (NR >= 2) {
        if constexpr (MR >= 1)
            asm volatile("vfmacc.vf v9,  ft0, v1" ::: "v9");
        if constexpr (MR >= 2)
            asm volatile("vfmacc.vf v12, ft1, v1" ::: "v12");
        if constexpr (MR >= 3)
            asm volatile("vfmacc.vf v15, ft2, v1" ::: "v15");
        if constexpr (MR >= 4)
            asm volatile("vfmacc.vf v18, ft3, v1" ::: "v18");
        if constexpr (MR >= 5)
            asm volatile("vfmacc.vf v21, ft4, v1" ::: "v21");
        if constexpr (MR >= 6)
            asm volatile("vfmacc.vf v24, ft5, v1" ::: "v24");
        if constexpr (MR >= 7)
            asm volatile("vfmacc.vf v27, ft6, v1" ::: "v27");
        if constexpr (MR >= 8)
            asm volatile("vfmacc.vf v30, ft7, v1" ::: "v30");
        // v1 freed → prefetch next B col 1 (strided)
        asm volatile("vlse32.v v1, (%0), %1"
                     :
                     : "r"(B_next + 1 * vl * b_stride_row), "r"(stride_bytes)
                     : "v1", "memory");
    }

    // =================================================================
    // Last column: vfmacc interleaved with next iteration's A loads
    // =================================================================
    if constexpr (NR == 3) {
        if constexpr (MR >= 1) {
            asm volatile("vfmacc.vf v10, ft0, v2" ::: "v10");
            asm volatile("flw ft0, 0(%0)" : : "r"(A_next + 0 * a_stride_row) : "ft0", "memory");
        }
        if constexpr (MR >= 2) {
            asm volatile("vfmacc.vf v13, ft1, v2" ::: "v13");
            asm volatile("flw ft1, 0(%0)" : : "r"(A_next + 1 * a_stride_row) : "ft1", "memory");
        }
        if constexpr (MR >= 3) {
            asm volatile("vfmacc.vf v16, ft2, v2" ::: "v16");
            asm volatile("flw ft2, 0(%0)" : : "r"(A_next + 2 * a_stride_row) : "ft2", "memory");
        }
        if constexpr (MR >= 4) {
            asm volatile("vfmacc.vf v19, ft3, v2" ::: "v19");
            asm volatile("flw ft3, 0(%0)" : : "r"(A_next + 3 * a_stride_row) : "ft3", "memory");
        }
        if constexpr (MR >= 5) {
            asm volatile("vfmacc.vf v22, ft4, v2" ::: "v22");
            asm volatile("flw ft4, 0(%0)" : : "r"(A_next + 4 * a_stride_row) : "ft4", "memory");
        }
        if constexpr (MR >= 6) {
            asm volatile("vfmacc.vf v25, ft5, v2" ::: "v25");
            asm volatile("flw ft5, 0(%0)" : : "r"(A_next + 5 * a_stride_row) : "ft5", "memory");
        }
        if constexpr (MR >= 7) {
            asm volatile("vfmacc.vf v28, ft6, v2" ::: "v28");
            asm volatile("flw ft6, 0(%0)" : : "r"(A_next + 6 * a_stride_row) : "ft6", "memory");
        }
        if constexpr (MR >= 8) {
            asm volatile("vfmacc.vf v31, ft7, v2" ::: "v31");
            asm volatile("flw ft7, 0(%0)" : : "r"(A_next + 7 * a_stride_row) : "ft7", "memory");
        }
        // v2 freed → prefetch next B col 2 (strided)
        asm volatile("vlse32.v v2, (%0), %1"
                     :
                     : "r"(B_next + 2 * vl * b_stride_row), "r"(stride_bytes)
                     : "v2", "memory");

    } else {
        // NR=1 or NR=2: A prefetch (sequential, no vfmacc to interleave with)
        if constexpr (MR >= 1)
            asm volatile("flw ft0, 0(%0)" : : "r"(A_next + 0 * a_stride_row) : "ft0", "memory");
        if constexpr (MR >= 2)
            asm volatile("flw ft1, 0(%0)" : : "r"(A_next + 1 * a_stride_row) : "ft1", "memory");
        if constexpr (MR >= 3)
            asm volatile("flw ft2, 0(%0)" : : "r"(A_next + 2 * a_stride_row) : "ft2", "memory");
        if constexpr (MR >= 4)
            asm volatile("flw ft3, 0(%0)" : : "r"(A_next + 3 * a_stride_row) : "ft3", "memory");
        if constexpr (MR >= 5)
            asm volatile("flw ft4, 0(%0)" : : "r"(A_next + 4 * a_stride_row) : "ft4", "memory");
        if constexpr (MR >= 6)
            asm volatile("flw ft5, 0(%0)" : : "r"(A_next + 5 * a_stride_row) : "ft5", "memory");
        if constexpr (MR >= 7)
            asm volatile("flw ft6, 0(%0)" : : "r"(A_next + 6 * a_stride_row) : "ft6", "memory");
        if constexpr (MR >= 8)
            asm volatile("flw ft7, 0(%0)" : : "r"(A_next + 7 * a_stride_row) : "ft7", "memory");
    }
}

/// Epilogue vfmacc kernel: column-first ordering, no prefetch.
/// Used for the last K iteration when there is no next iteration to prefetch.
template <size_t MR, size_t NR>
static inline __attribute__((always_inline)) void tqt_epilogue_vfmacc_kernel_sgemm_f32_f32_8x3vl_rvv() {
    static_assert(MR >= 1 && MR <= 8, "MR must be in range [1, 8]");
    static_assert(NR >= 1 && NR <= 3, "NR must be in range [1, 3]");

    // Column 0
    if constexpr (MR >= 1)
        asm volatile("vfmacc.vf v8,  ft0, v0" ::: "v8");
    if constexpr (MR >= 2)
        asm volatile("vfmacc.vf v11, ft1, v0" ::: "v11");
    if constexpr (MR >= 3)
        asm volatile("vfmacc.vf v14, ft2, v0" ::: "v14");
    if constexpr (MR >= 4)
        asm volatile("vfmacc.vf v17, ft3, v0" ::: "v17");
    if constexpr (MR >= 5)
        asm volatile("vfmacc.vf v20, ft4, v0" ::: "v20");
    if constexpr (MR >= 6)
        asm volatile("vfmacc.vf v23, ft5, v0" ::: "v23");
    if constexpr (MR >= 7)
        asm volatile("vfmacc.vf v26, ft6, v0" ::: "v26");
    if constexpr (MR >= 8)
        asm volatile("vfmacc.vf v29, ft7, v0" ::: "v29");

    // Column 1
    if constexpr (NR >= 2) {
        if constexpr (MR >= 1)
            asm volatile("vfmacc.vf v9,  ft0, v1" ::: "v9");
        if constexpr (MR >= 2)
            asm volatile("vfmacc.vf v12, ft1, v1" ::: "v12");
        if constexpr (MR >= 3)
            asm volatile("vfmacc.vf v15, ft2, v1" ::: "v15");
        if constexpr (MR >= 4)
            asm volatile("vfmacc.vf v18, ft3, v1" ::: "v18");
        if constexpr (MR >= 5)
            asm volatile("vfmacc.vf v21, ft4, v1" ::: "v21");
        if constexpr (MR >= 6)
            asm volatile("vfmacc.vf v24, ft5, v1" ::: "v24");
        if constexpr (MR >= 7)
            asm volatile("vfmacc.vf v27, ft6, v1" ::: "v27");
        if constexpr (MR >= 8)
            asm volatile("vfmacc.vf v30, ft7, v1" ::: "v30");
    }

    // Column 2
    if constexpr (NR >= 3) {
        if constexpr (MR >= 1)
            asm volatile("vfmacc.vf v10, ft0, v2" ::: "v10");
        if constexpr (MR >= 2)
            asm volatile("vfmacc.vf v13, ft1, v2" ::: "v13");
        if constexpr (MR >= 3)
            asm volatile("vfmacc.vf v16, ft2, v2" ::: "v16");
        if constexpr (MR >= 4)
            asm volatile("vfmacc.vf v19, ft3, v2" ::: "v19");
        if constexpr (MR >= 5)
            asm volatile("vfmacc.vf v22, ft4, v2" ::: "v22");
        if constexpr (MR >= 6)
            asm volatile("vfmacc.vf v25, ft5, v2" ::: "v25");
        if constexpr (MR >= 7)
            asm volatile("vfmacc.vf v28, ft6, v2" ::: "v28");
        if constexpr (MR >= 8)
            asm volatile("vfmacc.vf v31, ft7, v2" ::: "v31");
    }
}

/// Fused alpha/beta store: apply D = alpha * acc + beta * C directly from accumulator
/// registers (v8-v31), reading C BEFORE writing D. This is safe when C and D alias
/// the same memory, because C is loaded while the accumulator is still in registers
/// and D has not yet been written.
///
/// Must be called immediately after the K-loop epilogue, before any other code
/// touches v8-v31 or v0-v2 (v0-v2 are used as temporaries for loading C).
///
/// The caller should use store_kernel for the simple path (alpha==1, beta==0)
/// and this function for all other (alpha, beta) combinations.
template <size_t MR, size_t NR>
static inline __attribute__((always_inline)) void
tqt_store_alpha_beta_kernel_sgemm_f32_f32_8x3vl_rvv(size_t vl, float* D, const float* C, float alpha, float beta,
                                                    size_t stride_d_row, size_t stride_d_col, size_t ldc) {
    static_assert(MR >= 1 && MR <= 8, "MR must be in range [1, 8]");
    static_assert(NR >= 1 && NR <= 3, "NR must be in range [1, 3]");

    asm volatile("vsetvli zero, %0, e32, m1, ta, ma" : : "r"(vl) :);

    const bool has_c = (C != nullptr && beta != 0.0f);
    const bool need_alpha = (alpha != 1.0f);

    // Per-row macro: scale acc by alpha, load C into v0-v2, fmacc beta*C, store D.
    // V0/V1/V2 are the accumulator register names for the row.
#define TQT_STORE_AB_ROW(ROW, V0, V1, V2)                                                            \
    if constexpr (MR >= ROW + 1) {                                                                   \
        float* d_row = D + ROW * stride_d_row;                                                       \
        if (need_alpha) {                                                                            \
            if constexpr (NR >= 1)                                                                   \
                asm volatile("vfmul.vf " #V0 ", " #V0 ", %0" : : "f"(alpha) : #V0);                  \
            if constexpr (NR >= 2)                                                                   \
                asm volatile("vfmul.vf " #V1 ", " #V1 ", %0" : : "f"(alpha) : #V1);                  \
            if constexpr (NR >= 3)                                                                   \
                asm volatile("vfmul.vf " #V2 ", " #V2 ", %0" : : "f"(alpha) : #V2);                  \
        }                                                                                            \
        if (has_c) {                                                                                 \
            const float* c_row = C + ROW * ldc;                                                      \
            if constexpr (NR >= 1) {                                                                 \
                asm volatile("vle32.v v0, (%0)" : : "r"(c_row + 0 * stride_d_col) : "v0", "memory"); \
                asm volatile("vfmacc.vf " #V0 ", %0, v0" : : "f"(beta) : #V0);                       \
            }                                                                                        \
            if constexpr (NR >= 2) {                                                                 \
                asm volatile("vle32.v v1, (%0)" : : "r"(c_row + 1 * stride_d_col) : "v1", "memory"); \
                asm volatile("vfmacc.vf " #V1 ", %0, v1" : : "f"(beta) : #V1);                       \
            }                                                                                        \
            if constexpr (NR >= 3) {                                                                 \
                asm volatile("vle32.v v2, (%0)" : : "r"(c_row + 2 * stride_d_col) : "v2", "memory"); \
                asm volatile("vfmacc.vf " #V2 ", %0, v2" : : "f"(beta) : #V2);                       \
            }                                                                                        \
        }                                                                                            \
        if constexpr (NR >= 1)                                                                       \
            asm volatile("vse32.v " #V0 ", (%0)" : : "r"(d_row + 0 * stride_d_col) : "memory");      \
        if constexpr (NR >= 2)                                                                       \
            asm volatile("vse32.v " #V1 ", (%0)" : : "r"(d_row + 1 * stride_d_col) : "memory");      \
        if constexpr (NR >= 3)                                                                       \
            asm volatile("vse32.v " #V2 ", (%0)" : : "r"(d_row + 2 * stride_d_col) : "memory");      \
    }

    TQT_STORE_AB_ROW(0, v8, v9, v10)
    TQT_STORE_AB_ROW(1, v11, v12, v13)
    TQT_STORE_AB_ROW(2, v14, v15, v16)
    TQT_STORE_AB_ROW(3, v17, v18, v19)
    TQT_STORE_AB_ROW(4, v20, v21, v22)
    TQT_STORE_AB_ROW(5, v23, v24, v25)
    TQT_STORE_AB_ROW(6, v26, v27, v28)
    TQT_STORE_AB_ROW(7, v29, v30, v31)

#undef TQT_STORE_AB_ROW
}

#undef TQT_VZERO_FP32
