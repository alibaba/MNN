//
//  GemmAVX2FMA.cpp
//  MNN
//
//  Created by MNN on b'2020/09/22'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "FunctionSummary.hpp"
#include "GemmCommon.hpp"
#include "core/Macro.h"
#define MNNAVXFMA _mm256_fmadd_ps
#define MNNSSEFMA _mm_fmadd_ps
#include "GemmFunction.hpp"
#ifdef MNN_X86_USE_ASM
extern "C" {
void _AVX_MNNGemmFloatUnitMainFMA(float* C, const float* A, const float* B, const size_t* parameter, size_t hC4);
}
#endif

void _AVX_MNNPackedMatMulFMA(float* C, const float* A, const float* B, const size_t* parameter, float* cache,
                             const float* postParameters, const float* bias) {
    auto h       = parameter[2];
    auto hC4     = UP_DIV(h, 4);
    auto cStride = parameter[3] / sizeof(float);
#ifdef MNN_X86_USE_ASM
    _AVX_MNNGemmFloatUnitMainFMA(C, A, B, parameter, hC4);
#else
    _AVX_MNNPackedMatMul_24(C, A, B, parameter);
#endif
    AVX2GemmPostTreat(C, 24, parameter, postParameters, bias);
}

void _AVX_MNNPackedMatMulRemainFMA(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter,
                                   float* cache, const float* postParameters, const float* bias) {
    _AVX_MNNPackednMatMulRemainCommon(C, A, B, eSize, parameter, cache, postParameters, bias);
    AVX2GemmPostTreat(C, eSize, parameter, postParameters, bias);
}
