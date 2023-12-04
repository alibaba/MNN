//
//  GemmSSE.cpp
//  MNN
//
//  Created by MNN on 2020/09/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "FunctionSummary.hpp"
#include "GemmCommon.hpp"
#include "core/Macro.h"
#define MNNSSEFMA(x, y, z) _mm_add_ps(_mm_mul_ps(x, y), z)
#include "GemmFunction.hpp"

void _SSE_MNNPackedMatMul(float* C, const float* A, const float* B, const size_t* parameter,
                          const float* postParameters, const float* bias, const float* k, const float* b) {
    auto h       = parameter[2];
    auto hC4     = UP_DIV(h, 4);
    auto cStride = parameter[3] / sizeof(float);
    _SSE_MNNPackedMatMul_12(C, A, B, parameter);
    _SSE_GemmPostTreat(C, 12, parameter, postParameters, bias);
}

void _SSE_MNNPackedMatMulRemain(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter,
                                const float* postParameters, const float* bias, const float* k, const float* b) {
    _SSE_MNNPackednMatMulRemainCommon(C, A, B, eSize, parameter, postParameters, bias);
    _SSE_GemmPostTreat(C, eSize, parameter, postParameters, bias);
}

#ifdef MNN_LOW_MEMORY
//----------------------- MatMul(float, int4) Functions ---------------------------//
void _SSE_MNNPackedMatMul_int4(float* C, const float* A, const float* B, const size_t* parameter,
                               const float* postParameters, const float* bias, const float* k, const float* b) {
    auto h       = parameter[2];
    auto hC4     = UP_DIV(h, 4);
    auto cStride = parameter[3] / sizeof(float);
    _SSE_MNNPackedMatMul_12_int4(C, A, B, parameter, k, b);
    _SSE_GemmPostTreat(C, 12, parameter, postParameters, bias);
}

void _SSE_MNNPackedMatMulRemain_int4(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter,
                                     const float* postParameters, const float* bias, const float* k, const float* b) {
    _SSE_MNNPackednMatMulRemainCommon_int4(C, A, B, eSize, parameter, postParameters, bias, k, b);
    _SSE_GemmPostTreat(C, eSize, parameter, postParameters, bias);
}

void _SSE_MNNPackedMatMul_int8(float* C, const float* A, const float* B, const size_t* parameter,
                               const float* postParameters, const float* bias, const float* k, const float* b) {
    auto h       = parameter[2];
    auto hC4     = UP_DIV(h, 4);
    auto cStride = parameter[3] / sizeof(float);
    _SSE_MNNPackedMatMul_12_int8(C, A, B, parameter, k, b);
    _SSE_GemmPostTreat(C, 12, parameter, postParameters, bias);
}

void _SSE_MNNPackedMatMulRemain_int8(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter,
                                     const float* postParameters, const float* bias, const float* k, const float* b) {
    _SSE_MNNPackednMatMulRemainCommon_int8(C, A, B, eSize, parameter, postParameters, bias, k, b);
    _SSE_GemmPostTreat(C, eSize, parameter, postParameters, bias);
}

void _SSE_MNNGemmHybridInt4(float* C, const int8_t* A, const int8_t* B, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, size_t realSize, const float** param) {
    _SSE_MNNGemmHybrid_int4(C, A,  B, src_depth_quad, dst_step, dst_depth_quad, realSize, param);
}
void _SSE_MNNGemmHybridInt8(float* C, const int8_t* A, const int8_t* B, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, size_t realSize, const float** param) {
    _SSE_MNNGemmHybrid_int8(C, A,  B, src_depth_quad, dst_step, dst_depth_quad, realSize, param);
}
#endif
