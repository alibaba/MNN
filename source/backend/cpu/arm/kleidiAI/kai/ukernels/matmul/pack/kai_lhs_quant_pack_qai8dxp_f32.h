//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Gets the m step value.
/// The micro-kernel can process any M values. However, the starting M index to
/// be processed must be a multiple of m step.
///
/// @param[in] mr The number of M rows to interleave on the same output row.
///
/// @return the m step value
size_t kai_get_m_step_lhs_quant_pack_qai8dxp_f32(size_t mr);

/// Gets the offset in bytes for the LHS matrix (not packed)
///
/// This function should be called before passing the pointer to the LHS matrix to the micro-kernel.
///
/// @param[in] m_idx      Row index in the LHS matrix (not packed).
/// @param[in] lhs_stride The number of bytes in in each row of the LHS matrix (not packed)
///
/// @return the offset in bytes to the LHS matrix
size_t kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f32(size_t m_idx, size_t lhs_stride);

/// Gets the offset in bytes for the packed LHS matrix,
/// which contains the packed 8-bit quantized asymmetric per-row (qa8dx) values.
///
/// This function should be called before passing the pointer to the packed LHS matrix to the micro-kernel.
///
/// @param[in] m_idx Row index in the LHS matrix (not packed).
/// @param[in] k     Total number of columns in the LHS matrix (not packed).
/// @param[in] mr    The number of M rows to interleave on the same output row.
/// @param[in] kr    The number of columns loaded in the single inner most loop of the matmul micro-kernel.
/// @param[in] sr    The number of kr splits. It can be 1 (no splits) up to kr.
///
/// @return the offset in bytes to the packed LHS matrix
size_t kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32(size_t m_idx, size_t k, size_t mr, size_t kr, size_t sr);

/// Gets the size in bytes for the quantized and packed LHS matrix
///
/// @param[in] m  Total number of rows in the LHS matrix (not packed).
/// @param[in] k  Total number of columns in the LHS matrix (not packed).
/// @param[in] mr The number of M rows to interleave on the same output row.
/// @param[in] kr The number of columns loaded in the single inner most loop of the matmul micro-kernel.
/// @param[in] sr The number of kr splits. It can be 1 (no splits) up to kr.
///
/// @return the packed LHS matrix size in bytes
size_t kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(size_t m, size_t k, size_t mr, size_t kr, size_t sr);

/// Run the micro-kernel to quantize and pack the LHS matrix.
///
/// @param[in]  m           The number of output rows written.
/// @param[in]  k           The number of channels. The common dimension of LHS & RHS. It must be multiple of 8.
/// @param[in]  mr          The number of M rows to interleave on the same output row.
/// @param[in]  kr          The number of columns loaded in the single inner most loop of the matmul micro-kernel.
/// @param[in]  sr          The number of kr splits. It can be 1 (no splits) up to kr.
///                         However, kr must be multiple of sr.
/// @param[in]  m_idx_start The starting M index.
/// @param[in]  lhs         LHS of the vector-by-matrix.
/// @param[in]  lhs_stride  Stride in bytes between two rows of LHS.
/// @param[out] lhs_packed  The quantized and packed LHS matrix.
void kai_run_lhs_quant_pack_qai8dxp_f32(
    size_t m, size_t k, size_t mr, size_t kr, size_t sr, size_t m_idx_start, const float* lhs, size_t lhs_stride,
    void* lhs_packed);

#ifdef __cplusplus
}
#endif
