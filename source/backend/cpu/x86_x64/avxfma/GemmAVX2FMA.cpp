//
//  GemmAVX2FMA.cpp
//  MNN
//
//  Created by MNN on b'2020/09/22'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "FunctionSummary.hpp"
#include "../avx/GemmCommon.hpp"
#include "core/Macro.h"
#define MNNAVXFMA _mm256_fmadd_ps
#define MNNSSEFMA _mm_fmadd_ps
#define BROAD_LOAD(x) _mm256_broadcast_ss(x)
#define BROAD_LOAD_4(x) _mm_broadcast_ss(x)
#define LOAD8(x) _mm256_loadu_ps(x)
#define LOAD4(x) _mm_loadu_ps(x)
#define STORE_4(d, x) _mm_store_ps(d, x)
#define STORE_8(d, x) _mm256_storeu_ps(d, x)

#include "../avx/GemmFunction.hpp"
#ifdef MNN_X86_USE_ASM
extern "C" {
void _AVX_MNNGemmFloatUnitMainFMA(float* C, const float* A, const float* B, const size_t* parameter, size_t hC4);
}
#endif

void _AVX_MNNPackedMatMulFMA(float* C, const float* A, const float* B, const size_t* parameter,
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

void _AVX_MNNPackedMatMulRemainFMA(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias) {
    _AVX_MNNPackednMatMulRemainCommon(C, A, B, eSize, parameter);
    AVX2GemmPostTreat(C, eSize, parameter, postParameters, bias);
}

void _AVX_MNNComputeMatMulForE_1FMA(const float* A, const float* B, float* C, const float* biasPtr, const MatMulParam* param, size_t tId) {
    auto l = param->l;
    auto h = param->h;
    auto numberThread = param->numberThread;
    auto lC4 = l / 8;
    auto lR = lC4 * 8;
    if (param->BTranspose) {
        for (int y=tId; y<h; y+=numberThread) {
            auto sumValue = _mm256_set1_ps(0.0f);
            auto by = B + y * l;
            for (int x=0; x<lC4; ++x) {
                sumValue = _mm256_fmadd_ps(_mm256_loadu_ps(A + x * 8), _mm256_loadu_ps(by + x * 8), sumValue);
            }
            float sumRemain = 0.0f;
            for (int x=lR; x<l; ++x) {
                sumRemain = sumRemain + A[x] * by[x];
            }
            if (nullptr != biasPtr) {
                sumRemain += biasPtr[y];
            }
            sumValue = _mm256_hadd_ps(sumValue, sumValue);
            sumValue = _mm256_hadd_ps(sumValue, sumValue);
            auto s = _mm_cvtss_f32(_mm256_extractf128_ps(sumValue, 0)) + _mm_cvtss_f32(_mm256_extractf128_ps(sumValue, 1));
            C[y] = sumRemain + s;
        }
    } else {
        auto hC4 = h / 8;
        auto hR = hC4 * 8;
        for (int y=tId; y<hC4; y+=numberThread) {
            auto bs = B + 8 * y;
            auto sumValue = _mm256_set1_ps(0.0f);
            if (biasPtr != nullptr) {
                sumValue = _mm256_loadu_ps(biasPtr + 8 * y);
            }
            auto srcY = A + y * l;
            for (int x=0; x<l; ++x) {
                sumValue = _mm256_fmadd_ps(_mm256_broadcast_ss(A + x), _mm256_loadu_ps(bs + h * x), sumValue);
            }
            _mm256_storeu_ps(C + 8 * y, sumValue);
        }
        for (int y= hR + tId; y<h; y+=numberThread) {
            auto bs = B + y;
            float sumValue = 0.0f;
            if (biasPtr != nullptr) {
                sumValue = biasPtr[y];
            }
            auto srcY = A + y * l;
            for (int x=0; x<l; ++x) {
                sumValue = sumValue + A[x] * bs[h * x];
            }
            C[y] = sumValue;
        }
    }
}
