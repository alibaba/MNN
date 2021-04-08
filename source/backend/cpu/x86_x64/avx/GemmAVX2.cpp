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
#define BROAD_LOAD(x) _mm256_broadcast_ss(x)
#define BROAD_LOAD_4(x) _mm_broadcast_ss(x)
#define LOAD8(x) _mm256_loadu_ps(x)
#define LOAD4(x) _mm_loadu_ps(x)
#define STORE_4(d, x) _mm_store_ps(d, x) // The memory is aligned for 4
#define STORE_8(d, x) _mm256_storeu_ps(d, x)
#include "GemmFunction.hpp"

void _AVX_MNNPackedMatMul(float* C, const float* A, const float* B, const size_t* parameter,
                          const float* postParameters, const float* bias) {
    auto h       = parameter[2];
    auto hC4     = UP_DIV(h, 4);
    auto cStride = parameter[3] / sizeof(float);
    _AVX_MNNPackedMatMul_24(C, A, B, parameter);
    AVX2GemmPostTreat(C, 24, parameter, postParameters, bias);
}

void _AVX_MNNPackedMatMulRemain(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter,
                             const float* postParameters, const float* bias) {
    _AVX_MNNPackednMatMulRemainCommon(C, A, B, eSize, parameter);
    AVX2GemmPostTreat(C, eSize, parameter, postParameters, bias);
}
