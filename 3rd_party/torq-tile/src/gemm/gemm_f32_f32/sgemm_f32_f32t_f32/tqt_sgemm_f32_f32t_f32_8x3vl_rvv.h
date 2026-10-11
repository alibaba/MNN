//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/// Micro-kernel: SGEMM (D = alpha * A^T * B + beta * C) for FP32
///
/// This is a rvv (8 x 3*vl tile) implementation where vl = csrr_vlenb() / sizeof(float).
/// Computes: D = alpha * A^T * B + beta * C
/// where A is in transposed format [k, m] and B is in row-major format [k, n].

/// --------------------------------------------------

/// Gets m step value.
///
/// The starting row index must be divisible by `m_step`.
///
/// @return The m step value (8).
size_t tqt_get_m_step_sgemm_f32_f32t_f32_8x3vl_rvv(void);

/// Gets n step value.
///
/// The starting column index must be divisible by `n_step`.
///
/// @return The n step value (3 * vl, where vl = csrr_vlenb() / sizeof(float)).
size_t tqt_get_n_step_sgemm_f32_f32t_f32_8x3vl_rvv(void);

/// Gets the offset in bytes to the data element in the A matrix buffer.
///
/// @param[in] m_idx Row index (column index in transposed A).
/// @param[in] k_idx Column index (row index in transposed A).
/// @param[in] lda Leading dimension of A.
///
/// @return The offset in bytes to the data element.
size_t tqt_get_a_offset_sgemm_f32_f32t_f32_8x3vl_rvv(size_t m_idx, size_t k_idx, size_t lda);

/// Gets the offset in bytes to the data element in the B matrix buffer.
///
/// @param[in] n_idx Column index.
/// @param[in] k_idx Row index.
/// @param[in] ldb Leading dimension of B.
///
/// @return The offset in bytes to the data element.
size_t tqt_get_b_offset_sgemm_f32_f32t_f32_8x3vl_rvv(size_t n_idx, size_t k_idx, size_t ldb);

/// Gets the offset in bytes to the data element in the C matrix buffer.
///
/// @param[in] m_idx Row index.
/// @param[in] n_idx Column index.
/// @param[in] ldc Leading dimension of C.
///
/// @return The offset in bytes to the data element.
size_t tqt_get_c_offset_sgemm_f32_f32t_f32_8x3vl_rvv(size_t m_idx, size_t n_idx, size_t ldc);

/// Gets the offset in bytes to the data element in the D (output) matrix buffer.
///
/// @param[in] m_idx Row index.
/// @param[in] n_idx Column index.
/// @param[in] ldd Leading dimension of D.
///
/// @return The offset in bytes to the data element.
size_t tqt_get_d_offset_sgemm_f32_f32t_f32_8x3vl_rvv(size_t m_idx, size_t n_idx, size_t ldd);

/// Gets the size in bytes of the D (output) matrix buffer.
///
/// @param[in] m Number of rows.
/// @param[in] n Number of columns.
///
/// @return The size in bytes of the output matrix buffer.
size_t tqt_get_d_size_sgemm_f32_f32t_f32_8x3vl_rvv(size_t m, size_t n);

/// Runs the SGEMM microkernel.
///
/// Computes: D = alpha * A^T * B + beta * C
///
/// @param[in]  m Number of rows in D (columns of transposed A).
/// @param[in]  n Number of columns in B and D.
/// @param[in]  k Common dimension (rows of transposed A, rows of B).
/// @param[in]  A Input matrix A [k, m] in row-major format (transposed).
/// @param[in]  lda Leading dimension of A.
/// @param[in]  k_idx_a K index in A buffer.
/// @param[in]  B Input matrix B [k, n] in row-major format.
/// @param[in]  ldb Leading dimension of B.
/// @param[in]  k_idx_b K index in B buffer.
/// @param[in]  C Input matrix C [m, n] in row-major format (may be nullptr if beta == 0).
/// @param[in]  ldc Leading dimension of C.
/// @param[out] D Output matrix D [m, n] in row-major format.
/// @param[in]  ldd Leading dimension of D.
/// @param[in]  alpha Scalar multiplier for the A^T*B product.
/// @param[in]  beta Scalar multiplier for the C matrix.
///
/// @note All output matrices are in row-major format with float elements.
/// @note A is in transposed format, stored as [k, m].
/// @note B is in standard format, stored as [k, n].
/// @note k_idx_a and k_idx_b are used when A and B are sub-buffers with different sizes.
///
void tqt_run_sgemm_f32_f32t_f32_8x3vl_rvv(size_t m, size_t n, size_t k, const void *A, size_t lda,
                                          size_t k_idx_a, const void *B, size_t ldb, size_t k_idx_b,
                                          const void *C, size_t ldc, void *D, size_t ldd,
                                          float alpha, float beta);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
