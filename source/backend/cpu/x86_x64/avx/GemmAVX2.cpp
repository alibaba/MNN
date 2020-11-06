//
//  GemmAVX2.cpp
//  MNN
//
//  Created by MNN on b'2020/09/22'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "FunctionSummary.hpp"
#include "GemmCommon.hpp"
#include "core/Macro.h"
#define MNNAVXFMA(x, y, z) _mm256_add_ps(_mm256_mul_ps(x, y), z)
#define MNNSSEFMA(x, y, z) _mm_add_ps(_mm_mul_ps(x, y), z)

#include "GemmFunction.hpp"

void _AVX_MNNPackedMatMul(float* C, const float* A, const float* B, const size_t* parameter, float* cache,
                          const float* postParameters, const float* bias) {
    auto h       = parameter[2];
    auto hC4     = UP_DIV(h, 4);
    auto cStride = parameter[3] / sizeof(float);
    _AVX_MNNPackedMatMul_24(C, A, B, parameter);
    AVX2GemmPostTreat(C, 24, parameter, postParameters, bias);
}

void _AVX_MNNPackedMatMulRemain(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter,
                                float* cache, const float* postParameters, const float* bias) {
    _AVX_MNNPackednMatMulRemainCommon(C, A, B, eSize, parameter, cache, postParameters, bias);
    AVX2GemmPostTreat(C, eSize, parameter, postParameters, bias);
}
