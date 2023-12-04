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
    AVX2GemmPostTreat(C, MNN_UNIT_E, parameter, postParameters, bias);
}
void _AVX_MNNPackedMatMulRemain_int4(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter,
                             const float* postParameters, const float* bias, const float* k, const float* b) {
    _AVX_MNNPackednMatMulRemainCommon_int4(C, A, B, eSize, parameter, k, b);
    AVX2GemmPostTreat(C, eSize, parameter, postParameters, bias);
}
void _AVX_MNNPackedMatMul_int8(float* C, const float* A, const float* B, const size_t* parameter,
                               const float* postParameters, const float* bias, const float* k, const float* b) {
    _AVX_MNNPackedMatMul_Main_int8(C, A, B, parameter, k, b);
    AVX2GemmPostTreat(C, MNN_UNIT_E, parameter, postParameters, bias);
}
void _AVX_MNNPackedMatMulRemain_int8(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter,
                             const float* postParameters, const float* bias, const float* k, const float* b) {
    _AVX_MNNPackednMatMulRemainCommon_int8(C, A, B, eSize, parameter, k, b);
    AVX2GemmPostTreat(C, eSize, parameter, postParameters, bias);
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

void _AVX_MNNGemmHybridInt4(float* C, const int8_t* A, const int8_t* B, size_t src_depth_quad, size_t dst_step,
                            size_t dst_depth_quad, size_t realSize, const float** param) {
    int pack = 8;
    size_t weight_step = src_depth_quad * pack * pack * 0.5;
    size_t weight_stride = pack * pack * 0.5;
    const float* alpha_ptr = param[0];
    const float* zero_ptr = param[1];
    const float* bias_ptr = param[2];
    const float* sums_ptr = param[3];
    const float* scale_ptr = param[4];
    auto one_int16 = _mm256_set1_epi16(1);
    auto offset_int8 = _mm256_set1_epi8(128);
    auto _int4_signed_8 = _mm256_set1_ps(8);
    for (int ci = 0; ci < dst_depth_quad; ++ci) {
        float* dstZ = C + ci * pack * realSize;
        const int8_t*    weight = B + ci * weight_step;
        auto alpha = alpha_ptr + ci * pack;
        auto zero  = zero_ptr + ci * pack;
        auto bias  = bias_ptr + ci * pack;
        __m256 alphaValue = _mm256_loadu_ps(alpha);
        auto extra_sum = _mm256_mul_ps(_int4_signed_8, alphaValue);
        for (int j = 0; j < realSize; ++j) {
            const float* sums = sums_ptr + j;
            const float* scale = scale_ptr + j;
            float* dstX = dstZ + j * pack;
            __m256  scaleValue = _mm256_set1_ps(scale[0]);
            auto sum_val = _mm256_set1_ps(sums[0]);
            __m256 biasValue  = _mm256_add_ps(_mm256_loadu_ps(bias), _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(zero), extra_sum), sum_val));
            const int8_t* srcBatch = A + j * pack;
            auto oc0123_int16 = _mm256_set1_epi16(0); 
            auto oc4567_int16 = _mm256_set1_epi16(0);
            auto oc0123_int32 = _mm256_set1_epi32(0);
            auto oc4567_int32 = _mm256_set1_epi32(0);
            const __m256i mask = _mm256_set1_epi8(0xf);
            // auto extra = _mm256_set1_epi32(0);
            for (int k = 0; k < src_depth_quad; ++k) {
                auto srcZ = srcBatch + k * pack * realSize;
                const uint8_t* weightZ = (uint8_t*)weight + k * weight_stride;
                auto s0 = _mm256_castpd_si256(_mm256_broadcast_sd((double*)srcZ));
                auto wi4 = _mm256_castps_si256(_mm256_loadu_ps((const float*)weightZ));
                auto w_high = _mm256_and_si256(mask, _mm256_srli_epi16(wi4, 4));
                auto w_low  = _mm256_and_si256(mask, wi4);
                auto w0_ = _mm256_unpacklo_epi8(w_high, w_low);
                auto w1_ = _mm256_unpackhi_epi8(w_high, w_low);
                auto w0 = _mm256_permute2x128_si256(w0_, w1_, 0x20);
                auto w1 = _mm256_permute2x128_si256(w0_, w1_, 0x31);
                oc0123_int16 = _mm256_maddubs_epi16(w0, s0); // int16_t sum
                oc4567_int16 = _mm256_maddubs_epi16(w1, s0); // int16_t sum
                oc0123_int32 = _mm256_add_epi32(_mm256_madd_epi16(oc0123_int16, one_int16), oc0123_int32);
                oc4567_int32 = _mm256_add_epi32(_mm256_madd_epi16(oc4567_int16, one_int16), oc4567_int32);
            }

            auto oc0426_int32 = _mm256_unpacklo_epi32(oc0123_int32, oc4567_int32);
            auto oc1537_int32 = _mm256_unpackhi_epi32(oc0123_int32, oc4567_int32);
            auto tmp0 = _mm256_unpacklo_epi32(oc0426_int32, oc1537_int32); // 01452367
            auto tmp1 = _mm256_unpackhi_epi32(oc0426_int32, oc1537_int32); // 01452367
            auto tmp2 = _mm256_add_epi32(tmp0, tmp1); // 01452367
            auto oc0145 = _mm256_extractf128_si256(tmp2, 0);
            auto oc2367 = _mm256_extractf128_si256(tmp2, 1);
            auto oc0123 = _mm_unpacklo_epi64(oc0145, oc2367);
            auto oc4567 = _mm_unpackhi_epi64(oc0145, oc2367);

            auto sum8 = _mm256_set_m128i(oc4567, oc0123);

            __m256 f0 = _mm256_cvtepi32_ps(sum8);
            __m256 fs = _mm256_mul_ps(_mm256_mul_ps(f0, scaleValue), alphaValue);
            fs = _mm256_add_ps(biasValue, fs);
            _mm256_storeu_ps(dstX, fs);

        }
    }
}
void _AVX_MNNGemmHybridInt8(float* C, const int8_t* A, const int8_t* B, size_t src_depth_quad, size_t dst_step,
                            size_t dst_depth_quad, size_t realSize, const float** param) {
    int pack = 8;
    size_t weight_step = src_depth_quad * pack * pack;
    const float* alpha_ptr = param[0];
    const float* zero_ptr = param[1];
    const float* bias_ptr = param[2];
    const float* sums_ptr = param[3];
    const float* scale_ptr = param[4];
    for (int ci = 0; ci < dst_depth_quad; ++ci) {
        float* dstZ = C + ci * pack * realSize;
        const int8_t*    weight = B + ci * weight_step;
        auto alpha = alpha_ptr + ci * pack;
        auto zero  = zero_ptr + ci * pack;
        auto bias  = bias_ptr + ci * pack;
        __m256 alphaValue = _mm256_load_ps(alpha);
        for (int j = 0; j < realSize; ++j) {
            const float* sums = sums_ptr + j;
            const float* scale = scale_ptr + j;
            float* dstX = dstZ + j * pack;
            __m256  scaleValue = _mm256_set1_ps(scale[0]);
            __m256 biasValue  = _mm256_add_ps(_mm256_load_ps(bias), _mm256_mul_ps(_mm256_load_ps(zero), _mm256_set1_ps(sums[0])));
            const int8_t* srcBatch = A + j * pack;
            auto oc0_and_1 = _mm256_set1_epi32(0);
            auto oc2_and_3 = _mm256_set1_epi32(0);
            auto oc4_and_5 = _mm256_set1_epi32(0);
            auto oc6_and_7 = _mm256_set1_epi32(0);
            for (int k = 0; k < src_depth_quad; ++k) {
                const int8_t* srcZ = srcBatch + k * pack * realSize;
                const int8_t* weightZ = weight + k * pack * pack;
                auto w0 = _mm_loadu_si128((__m128i const*)weightZ); // w0-1
                auto w1 = _mm_loadu_si128((__m128i const*)(weightZ + 16));
                auto w2 = _mm_loadu_si128((__m128i const*)(weightZ + 16 * 2));
                auto w3 = _mm_loadu_si128((__m128i const*)(weightZ + 16 * 3));
                auto w0_16=  _mm256_cvtepi8_epi16(w0); //16xint16_t
                auto w1_16=  _mm256_cvtepi8_epi16(w1);
                auto w2_16=  _mm256_cvtepi8_epi16(w2);
                auto w3_16=  _mm256_cvtepi8_epi16(w3);
                auto s0 = _mm_castps_si128(_mm_broadcast_ss((float*)srcZ + 0));
                auto s1 = _mm_castps_si128(_mm_broadcast_ss((float*)srcZ + 1));
                auto s0_16 = _mm256_cvtepi8_epi16(s0);
                auto s1_16 = _mm256_cvtepi8_epi16(s1);
                auto S_int16 = _mm256_unpacklo_epi64(s0_16, s1_16);
                oc0_and_1 = _mm256_add_epi32(oc0_and_1, _mm256_madd_epi16(S_int16, w0_16));
                oc2_and_3 = _mm256_add_epi32(oc2_and_3, _mm256_madd_epi16(S_int16, w1_16));
                oc4_and_5 = _mm256_add_epi32(oc4_and_5, _mm256_madd_epi16(S_int16, w2_16));
                oc6_and_7 = _mm256_add_epi32(oc6_and_7, _mm256_madd_epi16(S_int16, w3_16));
            }
            auto oc0 = _mm256_extractf128_si256(oc0_and_1, 0);
            auto oc1 = _mm256_extractf128_si256(oc0_and_1, 1);
            auto oc2 = _mm256_extractf128_si256(oc2_and_3, 0);
            auto oc3 = _mm256_extractf128_si256(oc2_and_3, 1);
            auto oc4 = _mm256_extractf128_si256(oc4_and_5, 0);
            auto oc5 = _mm256_extractf128_si256(oc4_and_5, 1);
            auto oc6 = _mm256_extractf128_si256(oc6_and_7, 0);
            auto oc7 = _mm256_extractf128_si256(oc6_and_7, 1);
            auto d0 = _mm_unpacklo_epi32(oc0, oc1);
            auto d1 = _mm_unpackhi_epi32(oc0, oc1);
            auto d2 = _mm_unpacklo_epi32(oc2, oc3);
            auto d3 = _mm_unpackhi_epi32(oc2, oc3);
            
            auto e0 = _mm_unpacklo_epi64(d0, d2);
            auto e1 = _mm_unpackhi_epi64(d0, d2);
            auto e2 = _mm_unpacklo_epi64(d1, d3);
            auto e3 = _mm_unpackhi_epi64(d1, d3);
            
            e0 = _mm_add_epi32(e0, e1);
            e2 = _mm_add_epi32(e2, e3);
            e0 = _mm_add_epi32(e0, e2);
            auto r0 = _mm_unpacklo_epi32(oc4, oc5);
            auto r1 = _mm_unpackhi_epi32(oc4, oc5);
            auto r2 = _mm_unpacklo_epi32(oc6, oc7);
            auto r3 = _mm_unpackhi_epi32(oc6, oc7);
            
            auto u0 = _mm_unpacklo_epi64(r0, r2);
            auto u1 = _mm_unpackhi_epi64(r0, r2);
            auto u2 = _mm_unpacklo_epi64(r1, r3);
            auto u3 = _mm_unpackhi_epi64(r1, r3);
            
            u0 = _mm_add_epi32(u0, u1);
            u2 = _mm_add_epi32(u2, u3);
            u0 = _mm_add_epi32(u0, u2);
            auto sum8 = _mm256_set_m128i(u0, e0);
            __m256 f0 = _mm256_cvtepi32_ps(sum8);
            __m256 fs = _mm256_mul_ps(_mm256_mul_ps(f0, scaleValue), alphaValue);
            fs = _mm256_add_ps(biasValue, fs);
            _mm256_storeu_ps(dstX, fs);
        }
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
