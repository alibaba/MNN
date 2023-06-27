//
//  FunctionSummary.hpp
//  MNN
//
//  Created by MNN on 2021/02/23.
//  Copyright © 2018 - 2021 Alibaba Group Holding Limited

#ifndef FUNCTIONSUMMARY_HPP_
#define FUNCTIONSUMMARY_HPP_

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "core/Macro.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __aarch64__
void MNNPackC8(float* dest, const float* source, size_t l, size_t h);
#endif

#if defined(MNN_SUPPORT_BF16)
void NEON_MNNGetMatMulPackMode_BF16(int* eP, int* lP, int* hP);

void NEON_MNNPackC4ForMatMul_A_BF16(float* destOrigin, float const** sourceGroup, const int32_t* info,
                                    const int32_t* el);


void NEON_MNNPackForMatMul_B_BF16(float* dest, const float* source, size_t h, size_t l, bool transpose);

void NEON_MNNPackedMatMul_BF16(float* C, const float* A, const float* B, const size_t* parameter,
                               const float* postParameters, const float* bias, const float* k, const float* b);
void NEON_MNNPackedMatMulRemain_BF16(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter,
                                     const float* postParameters, const float* bias, const float* k, const float* b);

void NEON_MNNConvRunForUnitDepthWise_BF16(float* dst, const float* src, const float* weight, size_t fw, size_t fh,
                                     size_t weight_y_step, size_t dilateX_step, size_t dilateY_step);
void NEON_MNNConvRunForLineDepthwise_BF16(float* dst, const float* src, const float* weight, size_t width,
                                     size_t src_w_setup, size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step,
                                     size_t height, size_t srcHStep, size_t dstHStep);
void NEON_MNNAxByClampBroadcastC4_BF16(float* C, const float* A, const float* B, size_t width, size_t cStride,
                                  size_t aStride, size_t height, const float* parameters);

void MNNPackC4_BF16(float* dest, const float* source, size_t area, size_t depth, int32_t* areaOffset);
#ifdef __aarch64__
void MNNPackC8_BF16(float* dest, const float* source, size_t l, size_t h);
void ARMV86_MNNGetMatMulPackMode_BF16(int* eP, int* lP, int* hP);
void ARMV86_MNNPackForMatMul_B_BF16(float* dest, const float* source, size_t h, size_t l, bool transpose);
void ARMV86_MNNPackC4ForMatMul_A_BF16(float* destOrigin, float const** sourceGroup, const int32_t* info, const int32_t* el);
void ARMV86_MNNPackedMatMul_BF16(float* C, const float* A, const float* B, const size_t* parameter,
                                 const float* postParameters, const float* bias, const float* k, const float* b);
void ARMV86_MNNPackedMatMulRemain_BF16(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter,
                                       const float* postParameters, const float* bias, const float* k, const float* b);
#endif
#endif

#ifdef __cplusplus
}
#endif

#endif
