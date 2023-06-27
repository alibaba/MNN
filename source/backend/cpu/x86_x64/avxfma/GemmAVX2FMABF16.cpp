//
//  GemmAVX2FMABF16.cpp
//  MNN
//
//  Created by MNN on 2021/01/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_SUPPORT_BF16
#include "FunctionSummary.hpp"
#include "../avx/GemmCommon.hpp"
#include "core/Macro.h"

inline __m128i mnn_mm_loadu_si16(const void* x) {
    union S {
        short v16;
        __m128i v;
    } s;
    s.v16 = *((int16_t*)(x));
    return s.v;
}

#define MNNAVXFMA _mm256_fmadd_ps
#define MNNSSEFMA _mm_fmadd_ps
#ifndef MNN_SSE_USE_FP16_INSTEAD
#define BROAD_LOAD(x) _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepi16_epi32(_mm_broadcastw_epi16(mnn_mm_loadu_si16(x))), 16))
#define BROAD_LOAD_4(x) _mm256_extractf128_ps(_mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepi16_epi32(_mm_broadcastw_epi16(mnn_mm_loadu_si16(x))), 16)), 0)
#define LOAD8(x) _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepi16_epi32(_mm_loadu_si128((__m128i*)(x))), 16))
#define LOAD4(x) _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepi16_epi32(_mm_loadl_epi64((__m128i*)(x))), 16))
#define STORE_4(d, x) _mm_storel_epi64((__m128i*)(d), _mm_packs_epi32(_mm_srai_epi32(_mm_castps_si128(x), 16), _mm_srai_epi32(_mm_castps_si128(x), 16)))
#define STORE_8(d, x) _mm_storeu_ps((float*)(d), _mm_castsi128_ps(_mm_packs_epi32(_mm256_extractf128_si256(_mm256_srai_epi32(_mm256_castps_si256(x), 16), 0), _mm256_extractf128_si256(_mm256_srai_epi32(_mm256_castps_si256(x), 16), 1))))
#else
#define BROAD_LOAD(x) _mm256_cvtph_ps(_mm_broadcastw_epi16(mnn_mm_loadu_si16(x)))
#define BROAD_LOAD_4(x) _mm_cvtph_ps(_mm_broadcastw_epi16(mnn_mm_loadu_si16(x)))
#define LOAD8(x) _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(x)))
#define LOAD4(x) _mm_cvtph_ps(_mm_loadl_epi64((__m128i*)(x)))
#define STORE_4(d, x) _mm_storel_epi64((__m128i*)(d), _mm_cvtps_ph(x, 0x8))
#define STORE_8(d, x) _mm_storeu_si128((__m128i*)(d), _mm256_cvtps_ph(x, 0x8))
#endif
#include "../avx/GemmFunctionPackL.hpp"

void AVX2GemmPostTreatBF16(float* CO, size_t eSize, const size_t* parameter, const float* postParameters,
                       const float* biasO) {
    if (nullptr == postParameters) {
        return;
    }
    auto C = (int16_t*)CO;
    auto bias = (int16_t*)biasO;
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(int16_t);
    auto hC4          = UP_DIV(h, 4);
    auto minValue     = _mm_broadcast_ss(postParameters + 2);
    auto maxValue     = _mm_broadcast_ss(postParameters + 3);
    int eC2           = eSize / 2;
    int eR            = eSize % 2;
    auto minV2        = _mm256_broadcast_ss(postParameters + 2);
    auto maxV2        = _mm256_broadcast_ss(postParameters + 3);
    if (nullptr != bias) {
        if (eR > 0) {
            for (int y = 0; y < hC4; ++y) {
                auto biasValue = LOAD4(bias + 4 * y);
                auto bias2     = _mm256_castsi256_ps(_mm256_broadcastsi128_si256(_mm_castps_si128(biasValue)));
                auto dst       = C + y * cStride;
                for (int x = 0; x < eC2; ++x) {
                    auto sum = _mm256_add_ps(bias2, LOAD8(dst));
                    sum      = _mm256_max_ps(sum, minV2);
                    sum      = _mm256_min_ps(sum, maxV2);
                    STORE_8(dst, sum);
                    dst += 8;
                }
                auto sum = _mm_add_ps(biasValue, LOAD4(dst));
                sum      = _mm_max_ps(sum, minValue);
                sum      = _mm_min_ps(sum, maxValue);
                STORE_4(dst, sum);
            }
        } else {
            for (int y = 0; y < hC4; ++y) {
                auto biasValue = LOAD4(bias + 4 * y);
                auto bias2     = _mm256_castsi256_ps(_mm256_broadcastsi128_si256(_mm_castps_si128(biasValue)));
                auto dst       = C + y * cStride;
                for (int x = 0; x < eC2; ++x) {
                    auto sum = _mm256_add_ps(bias2, LOAD8(dst));
                    sum      = _mm256_max_ps(sum, minV2);
                    sum      = _mm256_min_ps(sum, maxV2);
                    STORE_8(dst, sum);
                    dst += 8;
                }
            }
        }
    } else {
        if (eR > 0) {
            for (int y = 0; y < hC4; ++y) {
                auto dst = C + y * cStride;
                for (int x = 0; x < eC2; ++x) {
                    auto sum = LOAD8(dst);
                    sum      = _mm256_max_ps(sum, minV2);
                    sum      = _mm256_min_ps(sum, maxV2);
                    STORE_8(dst, sum);
                    dst += 8;
                }
                auto sum = LOAD4(dst);
                sum      = _mm_max_ps(sum, minValue);
                sum      = _mm_min_ps(sum, maxValue);
                STORE_4(dst, sum);
            }
        } else {
            for (int y = 0; y < hC4; ++y) {
                auto dst = C + y * cStride;
                for (int x = 0; x < eC2; ++x) {
                    auto sum = LOAD8(dst);
                    sum      = _mm256_max_ps(sum, minV2);
                    sum      = _mm256_min_ps(sum, maxV2);
                    STORE_8(dst, sum);
                    dst += 8;
                }
            }
        }
    }
}

void _AVX_MNNPackedMatMulFMA_BF16(float* C, const float* A, const float* B, const size_t* parameter,
                             const float* postParameters, const float* bias, const float* k, const float* b) {
    _AVX_MNNPackedMatMul_3<int16_t>((int16_t*)C, (const int16_t*)A, (const int16_t*)B, parameter);
    AVX2GemmPostTreatBF16(C, 3, parameter, postParameters, bias);
}
void _AVX_MNNPackedMatMulRemainFMA_BF16(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias, const float* k, const float* b) {
    _AVX_MNNPackednMatMulRemainCommon<int16_t>((int16_t*)C, (const int16_t*)A, (const int16_t*)B, eSize, parameter);
    AVX2GemmPostTreatBF16(C, eSize, parameter, postParameters, bias);
}
#endif
