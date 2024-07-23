//
//  GemmAVX2.cpp
//  MNN
//
//  Created by MNN on 2020/09/22.
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
#define STORE_4(d, x) _mm_storeu_ps(d, x) // The memory is aligned for 4
#define STORE_8(d, x) _mm256_storeu_ps(d, x)
#include "GemmFunction.hpp"

void _AVX_MNNPackedMatMul(float* C, const float* A, const float* B, const size_t* parameter,
                          const float* postParameters, const float* bias, const float* k, const float* b) {
    _AVX_MNNPackedMatMul_Main(C, A, B, parameter);
    AVX2GemmPostTreat(C, MNN_UNIT_E, parameter, postParameters, bias);
}

void _AVX_MNNPackedMatMulRemain(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter,
                             const float* postParameters, const float* bias, const float* k, const float* b) {
    _AVX_MNNPackednMatMulRemainCommon(C, A, B, eSize, parameter);
    AVX2GemmPostTreat(C, eSize, parameter, postParameters, bias);
}

#ifdef MNN_LOW_MEMORY
void _AVX_MNNPackedMatMul_int4(float* C, const float* A, const float* B, const size_t* parameter,
                               const float* postParameters, const float* bias, const float* k, const float* b) {
    _AVX_MNNPackedMatMul_Main_int4(C, A, B, parameter, k, b);
    if (nullptr != bias) {
        AVX2GemmPostTreat(C, MNN_UNIT_E, parameter, postParameters, bias);
    }
}
void _AVX_MNNPackedMatMulRemain_int4(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter,
                             const float* postParameters, const float* bias, const float* k, const float* b) {
    _AVX_MNNPackednMatMulRemainCommon_int4(C, A, B, eSize, parameter, k, b);
    if (nullptr != bias) {
        AVX2GemmPostTreat(C, eSize, parameter, postParameters, bias);
    }
}
void _AVX_MNNPackedMatMul_int8(float* C, const float* A, const float* B, const size_t* parameter,
                               const float* postParameters, const float* bias, const float* k, const float* b) {
    _AVX_MNNPackedMatMul_Main_int8(C, A, B, parameter, k, b);
    if (nullptr != bias) {
        AVX2GemmPostTreat(C, MNN_UNIT_E, parameter, postParameters, bias);
    }
}
void _AVX_MNNPackedMatMulRemain_int8(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter,
                             const float* postParameters, const float* bias, const float* k, const float* b) {
    _AVX_MNNPackednMatMulRemainCommon_int8(C, A, B, eSize, parameter, k, b);
    if (nullptr != bias) {
        AVX2GemmPostTreat(C, eSize, parameter, postParameters, bias);
    }
}
static __m128i _load_int4_to_int8(const uint8_t* src) {
    uint8_t c = 0xf;
    uint8_t temp[16];
    for (int i = 0; i < 8; ++i) {
        temp[2 * i] = (src[i] >> 4);
        temp[2 * i +1] = (src[i] & c);
    }
    auto int8_tx16 = _mm_loadu_si128((const __m128i*)temp);
    return int8_tx16;
}

void _AVX_MNNAbsMaxFP32(const float* source, float* absmax, size_t src_depth_quad, size_t realSize, int pack) {
    // source: (ic/8, N, 8)
    auto srcStep = pack * realSize;
    auto constant = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    float temp[8];
    for (int i = 0; i < realSize; ++i) {
        __m256 res = _mm256_setzero_ps();
        for (int c = 0; c < src_depth_quad; ++c) {
            auto src0 = source + c * srcStep + i * pack;
            __m256 vecA = _mm256_loadu_ps(src0);
            __m256 absVecA = _mm256_and_ps(vecA, constant);
            __m256 mask = _mm256_cmp_ps(absVecA, res, 1);
            res = _mm256_blendv_ps(absVecA, res, mask);
        }
        _mm256_storeu_ps(temp, res);
        float absmaxVal = temp[0];
        for (int k = 1; k < pack; ++k) {
            if (absmaxVal < temp[k]) {
                absmaxVal = temp[k];
            }
        }
        absmax[i] = absmaxVal;
    }
}

#endif

void _AVX_MNNComputeMatMulForE_1(const float* A, const float* B, float* C, const float* biasPtr, const MatMulParam* param, size_t tId) {
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
                sumValue = _mm256_add_ps(sumValue, _mm256_mul_ps(_mm256_loadu_ps(A + x * 8), _mm256_loadu_ps(by + x * 8)));
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
                sumValue = _mm256_add_ps(sumValue, _mm256_mul_ps(_mm256_broadcast_ss(A + x), _mm256_loadu_ps(bs + h * x)));
            }
            _mm256_storeu_ps(C + 8 * y, sumValue);
        }
        for (int y = hR + tId; y<h; y+=numberThread) {
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

void _AVX_MNNComputeMatMulForH_1(const float* A, const float* B, float* C, const float* biasPtr, const MatMulParam* param, size_t tId) {
    int e = param->e;
    int l = param->l;
    int numberThread = param->numberThread;
    const int unit = 8;
    float biasVUnit = 0.0f;
    __m256 biasValue = _mm256_setzero_ps();
    if (nullptr != biasPtr) {
        biasValue = _mm256_broadcast_ss(biasPtr);
        biasVUnit = biasPtr[0];
    }
    if (param->ATranspose) {
        auto eC4 = e / unit;
        auto eR = eC4 * unit;
        for (int y=tId; y<eC4; y+=numberThread) {
            auto sumValue = biasValue;
            auto srcY = A + y * unit;
            for (int x=0; x<l; ++x) {
                sumValue = _mm256_add_ps(sumValue, _mm256_mul_ps(_mm256_loadu_ps(srcY + x * e), _mm256_broadcast_ss(B + x)));
            }
            _mm256_storeu_ps(C + unit * y, sumValue);
        }
        if (0 == tId) {
            for (int y=eR; y<e; ++y) {
                float sumValue = biasVUnit;
                auto srcY = A + y;
                for (int x=0; x<l; ++x) {
                    sumValue = sumValue + srcY[x * e] * B[x];
                }
                C[y] = sumValue;
            }
        }
        return;
    }
    auto lC4 = l / unit;
    auto lR = lC4 * unit;
    int eU = e / unit;
    int eR = e % unit;
    for (int y=tId; y<eU; y+=numberThread) {
        auto D0 = _mm256_setzero_ps();
        auto D1 = _mm256_setzero_ps();
        auto D2 = _mm256_setzero_ps();
        auto D3 = _mm256_setzero_ps();
        auto D4 = _mm256_setzero_ps();
        auto D5 = _mm256_setzero_ps();
        auto D6 = _mm256_setzero_ps();
        auto D7 = _mm256_setzero_ps();

        auto s0 = A + l * (y * unit + 0);
        auto s1 = A + l * (y * unit + 1);
        auto s2 = A + l * (y * unit + 2);
        auto s3 = A + l * (y * unit + 3);
        auto s4 = A + l * (y * unit + 4);
        auto s5 = A + l * (y * unit + 5);
        auto s6 = A + l * (y * unit + 6);
        auto s7 = A + l * (y * unit + 7);
        for (int x=0; x<lC4; ++x) {
            auto B0 = _mm256_loadu_ps(B + unit * x);
            auto A0 = _mm256_loadu_ps(s0);
            auto A1 = _mm256_loadu_ps(s1);
            auto A2 = _mm256_loadu_ps(s2);
            auto A3 = _mm256_loadu_ps(s3);
            auto A4 = _mm256_loadu_ps(s4);
            auto A5 = _mm256_loadu_ps(s5);
            auto A6 = _mm256_loadu_ps(s6);
            auto A7 = _mm256_loadu_ps(s7);
#define COMPUTE_TEMP(i) D##i = _mm256_add_ps(D##i, _mm256_mul_ps(A##i, B0))
            COMPUTE_TEMP(0);
            COMPUTE_TEMP(1);
            COMPUTE_TEMP(2);
            COMPUTE_TEMP(3);
            COMPUTE_TEMP(4);
            COMPUTE_TEMP(5);
            COMPUTE_TEMP(6);
            COMPUTE_TEMP(7);
            s0 += unit;
            s1 += unit;
            s2 += unit;
            s3 += unit;
            s4 += unit;
            s5 += unit;
            s6 += unit;
            s7 += unit;
        }
        if (lR < l) {
            int remain = l - lR;
            float tempB[8] = {0.0f};
            float tempA[8] = {0.0f};
            ::memcpy(tempB, B + unit * lC4, remain * sizeof(float));
            auto B0 = _mm256_loadu_ps(tempB);
            ::memcpy(tempA, s0, remain * sizeof(float));
            auto A0 = _mm256_loadu_ps(tempA);
            ::memcpy(tempA, s1, remain * sizeof(float));
            auto A1 = _mm256_loadu_ps(tempA);
            ::memcpy(tempA, s2, remain * sizeof(float));
            auto A2 = _mm256_loadu_ps(tempA);
            ::memcpy(tempA, s3, remain * sizeof(float));
            auto A3 = _mm256_loadu_ps(tempA);
            ::memcpy(tempA, s4, remain * sizeof(float));
            auto A4 = _mm256_loadu_ps(tempA);
            ::memcpy(tempA, s5, remain * sizeof(float));
            auto A5 = _mm256_loadu_ps(tempA);
            ::memcpy(tempA, s6, remain * sizeof(float));
            auto A6 = _mm256_loadu_ps(tempA);
            ::memcpy(tempA, s7, remain * sizeof(float));
            auto A7 = _mm256_loadu_ps(tempA);
            COMPUTE_TEMP(0);
            COMPUTE_TEMP(1);
            COMPUTE_TEMP(2);
            COMPUTE_TEMP(3);
            COMPUTE_TEMP(4);
            COMPUTE_TEMP(5);
            COMPUTE_TEMP(6);
            COMPUTE_TEMP(7);
        }
#undef COMPUTE_TEMP

        D0 = _mm256_hadd_ps(D0, D1);
        D2 = _mm256_hadd_ps(D2, D3);
        D4 = _mm256_hadd_ps(D4, D5);
        D6 = _mm256_hadd_ps(D6, D7);

        D0 = _mm256_hadd_ps(D0, D2);
        D4 = _mm256_hadd_ps(D4, D6);

        auto r0 = _mm_add_ps(_mm256_extractf128_ps(D0, 0), _mm256_extractf128_ps(D0, 1));
        auto r1 = _mm_add_ps(_mm256_extractf128_ps(D4, 0), _mm256_extractf128_ps(D4, 1));
        _mm_storeu_ps(C + y * unit + 0, r0);
        _mm_storeu_ps(C + y * unit + 4, r1);
    }
    for (int y=tId + eU * unit; y<e; y+=numberThread) {
        auto sumValue = _mm256_setzero_ps();
        auto srcY = A + y * l;
        for (int x=0; x<lC4; ++x) {
            sumValue = _mm256_add_ps(sumValue, _mm256_mul_ps(_mm256_loadu_ps(srcY + unit * x), _mm256_loadu_ps(B + unit * x)));
        }
        float temp[8];
        _mm256_storeu_ps(temp, sumValue);
        float sumSingle = biasVUnit;
        for (int i=0; i<8; ++i) {
            sumSingle += temp[i];
        }
        for (int x=lR; x<l; ++x) {
            sumSingle += srcY[x] * B[x];
        }
        C[y] = sumSingle;
    }
}

