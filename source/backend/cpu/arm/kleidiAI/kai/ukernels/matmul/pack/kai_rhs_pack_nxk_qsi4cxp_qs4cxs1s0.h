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

struct kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0_params {
    int8_t lhs_zero_point;
    uint8_t rhs_zero_point;
};

/// Get the n step value.
/// The micro-kernel can process any N values. However, the starting N index to
/// be processed must be a multiple of n step.
///
/// @param[in] nr The number of columns written by the matmul micro-kernel
///
/// @return the n step value
size_t kai_get_n_step_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(size_t nr);

/// Gets the offset in bytes for the RHS matrix (not packed).
///
/// @note  The int4 values are stored in a N x K matrix. Two int4 values are stored in one byte.
///        The lower order part of the byte (low) holds the first nibble (K-index + 0).
///        The higher order of the byte holds the second nibble (K-index + 1).
///
/// @param[in] n_idx      Row index in the RHS matrix (not packed). It must be a multiple of n_step.
/// @param[in] rhs_stride The number of bytes in in each row of the RHS matrix (not packed)
///
/// @return the offset in bytes to the RHS matrix (not packed)
size_t kai_get_rhs_offset_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(size_t n_idx, size_t rhs_stride);

/// Get the row stride in bytes to the packed RHS matrix
///
/// @param[in] k     In the RHS matrix (not packed), K is the number of columns.
/// @param[in] nr    The number of columns written by the matmul micro-kernel.
/// @param[in] kr    The number of columns loaded in the single inner most loop of the matmul micro-kernel.
/// @param[in] sr    The number of kr splits. It can be 1 (no splits) up to kr.
///
/// @return the stride in bytes to the packed RHS matrix
size_t kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(size_t k, size_t nr, size_t kr, size_t sr);

/// Gets the offset in bytes for the packed RHS matrix, which contains the packed 4-bit quantized symmetric per-channel
/// (qsu4cx) values.
///
/// @param[in] n_idx Row index in the RHS matrix (not packed). It must be a multiple of n_step.
/// @param[in] k     In the RHS matrix (not packed), K is the number of columns.
/// @param[in] nr    The number of columns written by the matmul micro-kernel
/// @param[in] kr    The number of columns loaded in the single inner most loop of the matmul micro-kernel.
/// @param[in] sr    The number of kr splits. It can be 1 (no splits) up to kr.
///
/// @return the offset in bytes to the packed RHS matrix
size_t kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(
    size_t n_idx, size_t k, size_t nr, size_t kr, size_t sr);

/// @brief Gets the size in bytes for the packed RHS matrix
///
/// @param[in] n The number of rows in the RHS matrix (not packed)
/// @param[in] k The number of columns in the RHS matrix (not packed).
/// @param[in] nr The number of columns written by the matmul micro-kernel
/// @param[in] kr The number of columns loaded in the single inner most loop of the matmul micro-kernel.
/// @param[in] sr The number of kr splits. It can be 1 (no splits) up to kr.
///
/// @return the packed RHS matrix size in bytes
size_t kai_get_rhs_packed_size_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(size_t n, size_t k, size_t nr, size_t kr, size_t sr);

/// Run the micro-kernel to pack the RHS matrix.
///
/// @note  The int4 values are stored in a N x K matrix. Two int4 values are stored in one byte.
///        The lower order part of the byte (low) holds the first nibble (K-index + 0).
///        The higher order of the byte holds the second nibble (K-index + 1).
///
/// @param[in]  num_groups  The number of groups. It must be 1.
/// @param[in]  n           The number of rows.
/// @param[in]  k           The common dimension between the LHS and RHS matrix (K). It must be an even value.
/// @param[in]  nr          The number of N rows to interleave on the same output output row.
/// @param[in]  kr          The number of K values loaded in the single inner most loop of the matmul micro-kernel.
/// @param[in]  sr          The number of kr splits. It can be 1 (no splits) up to kr.
///                         However, kr must be multiple of sr.
/// @param[in]  rhs         The RHS matrix containing the 4-bit values.
///                         Size in bytes is expected to be greater than or equal to n * k * (sizeof(uint8_t) / 2).
/// @param[in]  bias        The biases.
/// @param[in]  scale       The scale for each output channel.
/// @param[out] rhs_packed  The packed RHS matrix.
/// @param[in]  extra_bytes Extra bytes to append to the end of each row of the packed RHS matrix.
/// @param[in]  params      Parameters for the micro-kernel.
void kai_run_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(
    size_t num_groups,   //
    size_t n,            //
    size_t k,            //
    size_t nr,           //
    size_t kr,           //
    size_t sr,           //
    const uint8_t* rhs,  //
    const float* bias,   //
    const float* scale,  //
    void* rhs_packed,    //
    size_t extra_bytes,  //
    const struct kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0_params* params);

#ifdef __cplusplus
}
#endif
