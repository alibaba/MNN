//
//  GemmCommon.cpp
//  MNN
//
//  Created by MNN on b'2020/09/22'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "GemmCommon.hpp"
#include "FunctionSummary.hpp"
#include "core/Macro.h"

namespace {
static inline __m128i mm_loadu_si128(const void* addr) {
    return _mm_loadu_si128((__m128i const*)addr);
}
}  // namespace

void _AVX_MNNPackC4ForMatMul_A(float* dest, const float* source, size_t e, size_t l, size_t eReal) {
#define MAIN_COMPUTE                        \
    auto s00 = _mm_loadu_ps(srcX + 0 * 4);  \
    auto s01 = _mm_loadu_ps(srcX + 1 * 4);  \
    auto s02 = _mm_loadu_ps(srcX + 2 * 4);  \
    auto s03 = _mm_loadu_ps(srcX + 3 * 4);  \
    auto s10 = _mm_loadu_ps(srcX + 4 * 4);  \
    auto s11 = _mm_loadu_ps(srcX + 5 * 4);  \
    auto s12 = _mm_loadu_ps(srcX + 6 * 4);  \
    auto s13 = _mm_loadu_ps(srcX + 7 * 4);  \
    auto s20 = _mm_loadu_ps(srcX + 8 * 4);  \
    auto s21 = _mm_loadu_ps(srcX + 9 * 4);  \
    auto s22 = _mm_loadu_ps(srcX + 10 * 4); \
    auto s23 = _mm_loadu_ps(srcX + 11 * 4); \
    auto s30 = _mm_loadu_ps(srcX + 12 * 4); \
    auto s31 = _mm_loadu_ps(srcX + 13 * 4); \
    auto s32 = _mm_loadu_ps(srcX + 14 * 4); \
    auto s33 = _mm_loadu_ps(srcX + 15 * 4); \
    auto s40 = _mm_loadu_ps(srcX + 16 * 4); \
    auto s41 = _mm_loadu_ps(srcX + 17 * 4); \
    auto s42 = _mm_loadu_ps(srcX + 18 * 4); \
    auto s43 = _mm_loadu_ps(srcX + 19 * 4); \
    auto s50 = _mm_loadu_ps(srcX + 20 * 4); \
    auto s51 = _mm_loadu_ps(srcX + 21 * 4); \
    auto s52 = _mm_loadu_ps(srcX + 22 * 4); \
    auto s53 = _mm_loadu_ps(srcX + 23 * 4); \
    _MM_TRANSPOSE4_PS(s00, s01, s02, s03);  \
    _MM_TRANSPOSE4_PS(s10, s11, s12, s13);  \
    _MM_TRANSPOSE4_PS(s20, s21, s22, s23);  \
    _MM_TRANSPOSE4_PS(s30, s31, s32, s33);  \
    _MM_TRANSPOSE4_PS(s40, s41, s42, s43);  \
    _MM_TRANSPOSE4_PS(s50, s51, s52, s53);

#define STORE_TEMP(i)                               \
    _mm_storeu_ps(dstX + 4 * (6 * i + 0), s##0##i); \
    _mm_storeu_ps(dstX + 4 * (6 * i + 1), s##1##i); \
    _mm_storeu_ps(dstX + 4 * (6 * i + 2), s##2##i); \
    _mm_storeu_ps(dstX + 4 * (6 * i + 3), s##3##i); \
    _mm_storeu_ps(dstX + 4 * (6 * i + 4), s##4##i); \
    _mm_storeu_ps(dstX + 4 * (6 * i + 5), s##5##i);

    const int pack   = 24;
    const int mid    = 1; // Deprecate
    const int packC4 = pack / 4;
    auto ePack       = e / pack;
    auto lC4         = l / 4;
    auto lDiv        = UP_DIV(l, 4);
    auto eRemain     = ePack * pack;
    auto lRemain     = lC4 * 4;
    auto lRes        = l - lRemain;
    for (int y = 0; y < ePack; ++y) {
        auto dstY = dest + y * l * pack;
        auto srcY = source + y * pack * 4;
        for (int x = 0; x < lC4; ++x) {
            auto srcX = srcY + x * 4 * eReal;
            auto dstX = dstY + x * pack * 4;
            MAIN_COMPUTE;

            STORE_TEMP(0);
            STORE_TEMP(1);
            STORE_TEMP(2);
            STORE_TEMP(3);
        }
    }
    auto lastLc4Src = source + lC4 * 4 * eReal;
    auto lastLc4Dst = dest + lC4 * pack * 4;
    if (lRes == 3) {
        for (int y = 0; y < ePack; ++y) {
            auto dstX = lastLc4Dst + y * l * pack;
            auto srcX = lastLc4Src + y * pack * 4;
            MAIN_COMPUTE;
            STORE_TEMP(0);
            STORE_TEMP(1);
            STORE_TEMP(2);
        }
    } else if (lRes == 2) {
        for (int y = 0; y < ePack; ++y) {
            auto dstX = lastLc4Dst + y * l * pack;
            auto srcX = lastLc4Src + y * pack * 4;
            MAIN_COMPUTE;
            STORE_TEMP(0);
            STORE_TEMP(1);
        }
    } else if (lRes == 1) {
        for (int y = 0; y < ePack; ++y) {
            auto dstX = lastLc4Dst + y * l * pack;
            auto srcX = lastLc4Src + y * pack * 4;
            MAIN_COMPUTE;
            STORE_TEMP(0);
        }
    }
    // Down
    {
        auto eLast    = e - eRemain;
        auto lastDest = dest + ePack * pack * l;
        for (int xC = 0; xC < lC4; ++xC) {
            for (int y = eRemain; y < e; ++y) {
                auto yR = y - eRemain;
                for (int xR = 0; xR < 4; ++xR) {
                    lastDest[(xC * 4 + xR) * eLast + yR] = source[xC * eReal * 4 + y * 4 + xR];
                }
            }
        }
        for (int x = lC4 * 4; x < l; ++x) {
            auto xR = x % 4;
            auto xC = lC4;
            for (int y = eRemain; y < e; ++y) {
                auto yR                  = y - eRemain;
                lastDest[x * eLast + yR] = source[xC * eReal * 4 + y * 4 + xR];
            }
        }
    }
}

void AVX2GemmPostTreat(float* C, size_t eSize, const size_t* parameter, const float* postParameters,
                       const float* bias) {
    if (nullptr == postParameters) {
        return;
    }
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride      = bExtraStride + l * 4;
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
                auto biasValue = _mm_loadu_ps(bias + 4 * y);
                auto bias2     = _mm256_broadcast_ps((__m128*)(bias + 4 * y));
                auto dst       = C + y * cStride;
                for (int x = 0; x < eC2; ++x) {
                    auto sum = _mm256_add_ps(bias2, _mm256_loadu_ps(dst));
                    sum      = _mm256_max_ps(sum, minV2);
                    sum      = _mm256_min_ps(sum, maxV2);
                    _mm256_storeu_ps(dst, sum);
                    dst += 8;
                }
                auto sum = _mm_add_ps(biasValue, _mm_loadu_ps(dst));
                sum      = _mm_max_ps(sum, minValue);
                sum      = _mm_min_ps(sum, maxValue);
                _mm_storeu_ps(dst, sum);
            }
        } else {
            for (int y = 0; y < hC4; ++y) {
                auto biasValue = _mm_loadu_ps(bias + 4 * y);
                auto bias2     = _mm256_broadcast_ps((__m128*)(bias + 4 * y));
                auto dst       = C + y * cStride;
                for (int x = 0; x < eC2; ++x) {
                    auto sum = _mm256_add_ps(bias2, _mm256_loadu_ps(dst));
                    sum      = _mm256_max_ps(sum, minV2);
                    sum      = _mm256_min_ps(sum, maxV2);
                    _mm256_storeu_ps(dst, sum);
                    dst += 8;
                }
            }
        }
    } else {
        if (eR > 0) {
            for (int y = 0; y < hC4; ++y) {
                auto dst = C + y * cStride;
                for (int x = 0; x < eC2; ++x) {
                    auto sum = _mm256_loadu_ps(dst);
                    sum      = _mm256_max_ps(sum, minV2);
                    sum      = _mm256_min_ps(sum, maxV2);
                    _mm256_storeu_ps(dst, sum);
                    dst += 8;
                }
                auto sum = _mm_loadu_ps(dst);
                sum      = _mm_max_ps(sum, minValue);
                sum      = _mm_min_ps(sum, maxValue);
                _mm_storeu_ps(dst, sum);
            }
        } else {
            for (int y = 0; y < hC4; ++y) {
                auto dst = C + y * cStride;
                for (int x = 0; x < eC2; ++x) {
                    auto sum = _mm256_loadu_ps(dst);
                    sum      = _mm256_max_ps(sum, minV2);
                    sum      = _mm256_min_ps(sum, maxV2);
                    _mm256_storeu_ps(dst, sum);
                    dst += 8;
                }
            }
        }
    }
}
#ifdef MNN_X86_USE_ASM
extern "C" {
void _AVX_MNNGemmInt8AddBiasScale_16x4_UnitMain(int8_t* dst, const int8_t* src, const int8_t* weight, const size_t* strides, const QuanPostTreatParameters* post);
void _AVX_MNNGemmInt8AddBiasScale_16x4_Unit_1(int8_t* dst, const int8_t* src, const int8_t* weight, const size_t* strides, const QuanPostTreatParameters* post);
}
#endif
void _AVX_MNNGemmInt8AddBiasScale_16x4_Unit(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, const QuanPostTreatParameters* post, size_t realDst) {
#ifdef MNN_X86_USE_ASM
    size_t strides[3];
    strides[0] = src_depth_quad;
    strides[1] = dst_step;
    strides[2] = dst_depth_quad;
    if (realDst == GEMM_INT8_DST_XUNIT) {
        _AVX_MNNGemmInt8AddBiasScale_16x4_UnitMain(dst, src, weight, strides, post);
        return;
    }
    if (realDst == 1) {
        _AVX_MNNGemmInt8AddBiasScale_16x4_Unit_1(dst, src, weight, strides, post);
        return;
    }
#endif
#define EXTRACT_ADD(i)\
auto d##i##0 = _mm_castps_si128(_mm256_extractf128_ps(_mm256_castsi256_ps(D##i##0), 0));\
auto d##i##1 = _mm_castps_si128(_mm256_extractf128_ps(_mm256_castsi256_ps(D##i##0), 1));\
auto d##i = _mm_add_epi32(d##i##0, d##i##1);
#define COMPUTE(u, v)\
D##v##u = _mm256_add_epi32(D##v##u, _mm256_madd_epi16(W##u, S##v));

    const auto dst_step_tmp = dst_step / sizeof(int8_t);
    __m128 zero128 = _mm_set1_ps(0.0f);
    __m128 minValue = _mm_set1_ps(post->minValue);
    __m128 maxValue = _mm_set1_ps(post->maxValue);
    __m128 plus = _mm_set1_ps(0.5f);
    __m128 minus = _mm_set1_ps(-0.5f);
    if (4 == realDst) {
        for (int dz = 0; dz < dst_depth_quad; ++dz) {
            const auto weight_dz = weight + dz * src_depth_quad * (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT);
            const auto bias_dz   = post->bias + dz * GEMM_INT8_UNIT;
            const auto scale_dz  = post->scale + dz * GEMM_INT8_UNIT;
            auto dst_z           = dst + dz * dst_step_tmp;
            const auto src_x   = src;
            auto dst_x         = dst_z;
            __m256i D00 = _mm256_set1_epi32(0);
            __m256i D01 = _mm256_set1_epi32(0);
            __m256i D02 = _mm256_set1_epi32(0);
            __m256i D03 = _mm256_set1_epi32(0);

            __m256i D10 = _mm256_set1_epi32(0);
            __m256i D11 = _mm256_set1_epi32(0);
            __m256i D12 = _mm256_set1_epi32(0);
            __m256i D13 = _mm256_set1_epi32(0);

            __m256i D20 = _mm256_set1_epi32(0);
            __m256i D21 = _mm256_set1_epi32(0);
            __m256i D22 = _mm256_set1_epi32(0);
            __m256i D23 = _mm256_set1_epi32(0);

            __m256i D30 = _mm256_set1_epi32(0);
            __m256i D31 = _mm256_set1_epi32(0);
            __m256i D32 = _mm256_set1_epi32(0);
            __m256i D33 = _mm256_set1_epi32(0);

            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT) * sz;
                const auto src_z     = src_x + sz * GEMM_INT8_DST_XUNIT * GEMM_INT8_SRC_UNIT;
                auto w0 = mm_loadu_si128(weight_sz + GEMM_INT8_SRC_UNIT * 0);
                auto w1 = mm_loadu_si128(weight_sz + GEMM_INT8_SRC_UNIT * 1);
                auto w2 = mm_loadu_si128(weight_sz + GEMM_INT8_SRC_UNIT * 2);
                auto w3 = mm_loadu_si128(weight_sz + GEMM_INT8_SRC_UNIT * 3);
                auto W0 = _mm256_cvtepi8_epi16(w0);
                auto W1 = _mm256_cvtepi8_epi16(w1);
                auto W2 = _mm256_cvtepi8_epi16(w2);
                auto W3 = _mm256_cvtepi8_epi16(w3);

                auto s0 = mm_loadu_si128(src_z + GEMM_INT8_SRC_UNIT * 0);
                auto s1 = mm_loadu_si128(src_z + GEMM_INT8_SRC_UNIT * 1);
                auto s2 = mm_loadu_si128(src_z + GEMM_INT8_SRC_UNIT * 2);
                auto s3 = mm_loadu_si128(src_z + GEMM_INT8_SRC_UNIT * 3);
                auto S0 = _mm256_cvtepu8_epi16(s0);
                auto S1 = _mm256_cvtepu8_epi16(s1);
                auto S2 = _mm256_cvtepu8_epi16(s2);
                auto S3 = _mm256_cvtepu8_epi16(s3);

                COMPUTE(0, 0);
                COMPUTE(0, 1);
                COMPUTE(0, 2);
                COMPUTE(0, 3);

                COMPUTE(1, 0);
                COMPUTE(1, 1);
                COMPUTE(1, 2);
                COMPUTE(1, 3);

                COMPUTE(2, 0);
                COMPUTE(2, 1);
                COMPUTE(2, 2);
                COMPUTE(2, 3);

                COMPUTE(3, 0);
                COMPUTE(3, 1);
                COMPUTE(3, 2);
                COMPUTE(3, 3);
            }
            D00 = _mm256_hadd_epi32(D00, D01);
            D02 = _mm256_hadd_epi32(D02, D03);

            D10 = _mm256_hadd_epi32(D10, D11);
            D12 = _mm256_hadd_epi32(D12, D13);

            D20 = _mm256_hadd_epi32(D20, D21);
            D22 = _mm256_hadd_epi32(D22, D23);

            D30 = _mm256_hadd_epi32(D30, D31);
            D32 = _mm256_hadd_epi32(D32, D33);
            
            D00 = _mm256_hadd_epi32(D00, D02);
            D10 = _mm256_hadd_epi32(D10, D12);
            D20 = _mm256_hadd_epi32(D20, D22);
            D30 = _mm256_hadd_epi32(D30, D32);

            EXTRACT_ADD(0);
            EXTRACT_ADD(1);
            EXTRACT_ADD(2);
            EXTRACT_ADD(3);

            auto biasValue = _mm_loadu_si128((__m128i*)(bias_dz));
            d0 = _mm_add_epi32(d0, biasValue);
            d1 = _mm_add_epi32(d1, biasValue);
            d2 = _mm_add_epi32(d2, biasValue);
            d3 = _mm_add_epi32(d3, biasValue);

            auto scaleValue = _mm_loadu_ps(scale_dz);
            __m128 f0 = _mm_cvtepi32_ps(d0);
            __m128 f1 = _mm_cvtepi32_ps(d1);
            __m128 f2 = _mm_cvtepi32_ps(d2);
            __m128 f3 = _mm_cvtepi32_ps(d3);
            f0 = _mm_mul_ps(f0, scaleValue);
            f1 = _mm_mul_ps(f1, scaleValue);
            f2 = _mm_mul_ps(f2, scaleValue);
            f3 = _mm_mul_ps(f3, scaleValue);
            f0 = _mm_min_ps(f0, maxValue);
            f1 = _mm_min_ps(f1, maxValue);
            f2 = _mm_min_ps(f2, maxValue);
            f3 = _mm_min_ps(f3, maxValue);
            f0 = _mm_max_ps(f0, minValue);
            f1 = _mm_max_ps(f1, minValue);
            f2 = _mm_max_ps(f2, minValue);
            f3 = _mm_max_ps(f3, minValue);
            auto m0 = _mm_cmplt_ps(f0, zero128);
            auto m1 = _mm_cmplt_ps(f1, zero128);
            auto m2 = _mm_cmplt_ps(f2, zero128);
            auto m3 = _mm_cmplt_ps(f3, zero128);
            m0 = _mm_blendv_ps(plus, minus, m0);
            m1 = _mm_blendv_ps(plus, minus, m1);
            m2 = _mm_blendv_ps(plus, minus, m2);
            m3 = _mm_blendv_ps(plus, minus, m3);
            f0 = _mm_add_ps(f0, m0);
            f1 = _mm_add_ps(f1, m1);
            f2 = _mm_add_ps(f2, m2);
            f3 = _mm_add_ps(f3, m3);
            // 3: _MM_FROUND_TO_ZERO
            d0 = _mm_cvtps_epi32(_mm_round_ps(f0, 3));
            d1 = _mm_cvtps_epi32(_mm_round_ps(f1, 3));
            d2 = _mm_cvtps_epi32(_mm_round_ps(f2, 3));
            d3 = _mm_cvtps_epi32(_mm_round_ps(f3, 3));
            
            // Int32 -> Int8
            d0 = _mm_packs_epi32(d0, d1);
            d2 = _mm_packs_epi32(d2, d3);
            d0 = _mm_packs_epi16(d0, d2);
            _mm_storeu_ps((float*)dst_x, _mm_castsi128_ps(d0));
        }
    }

    if (3 == realDst) {
        for (int dz = 0; dz < dst_depth_quad; ++dz) {
            const auto weight_dz = weight + dz * src_depth_quad * (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT);
            const auto bias_dz   = post->bias + dz * GEMM_INT8_UNIT;
            const auto scale_dz  = post->scale + dz * GEMM_INT8_UNIT;
            auto dst_z           = dst + dz * dst_step_tmp;
            const auto src_x   = src;
            auto dst_x         = dst_z;
            __m256i D00 = _mm256_set1_epi32(0);
            __m256i D01 = _mm256_set1_epi32(0);
            __m256i D02 = _mm256_set1_epi32(0);
            __m256i D03 = _mm256_set1_epi32(0);

            __m256i D10 = _mm256_set1_epi32(0);
            __m256i D11 = _mm256_set1_epi32(0);
            __m256i D12 = _mm256_set1_epi32(0);
            __m256i D13 = _mm256_set1_epi32(0);

            __m256i D20 = _mm256_set1_epi32(0);
            __m256i D21 = _mm256_set1_epi32(0);
            __m256i D22 = _mm256_set1_epi32(0);
            __m256i D23 = _mm256_set1_epi32(0);

            __m256i D30 = _mm256_set1_epi32(0);
            __m256i D31 = _mm256_set1_epi32(0);
            __m256i D32 = _mm256_set1_epi32(0);
            __m256i D33 = _mm256_set1_epi32(0);

            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT) * sz;
                const auto src_z     = src_x + sz * GEMM_INT8_DST_XUNIT * GEMM_INT8_SRC_UNIT;
                auto w0 = mm_loadu_si128(weight_sz + GEMM_INT8_SRC_UNIT * 0);
                auto w1 = mm_loadu_si128(weight_sz + GEMM_INT8_SRC_UNIT * 1);
                auto w2 = mm_loadu_si128(weight_sz + GEMM_INT8_SRC_UNIT * 2);
                auto w3 = mm_loadu_si128(weight_sz + GEMM_INT8_SRC_UNIT * 3);
                auto W0 = _mm256_cvtepi8_epi16(w0);
                auto W1 = _mm256_cvtepi8_epi16(w1);
                auto W2 = _mm256_cvtepi8_epi16(w2);
                auto W3 = _mm256_cvtepi8_epi16(w3);

                auto s0 = mm_loadu_si128(src_z + GEMM_INT8_SRC_UNIT * 0);
                auto s1 = mm_loadu_si128(src_z + GEMM_INT8_SRC_UNIT * 1);
                auto s2 = mm_loadu_si128(src_z + GEMM_INT8_SRC_UNIT * 2);
                auto S0 = _mm256_cvtepu8_epi16(s0);
                auto S1 = _mm256_cvtepu8_epi16(s1);
                auto S2 = _mm256_cvtepu8_epi16(s2);

                COMPUTE(0, 0);
                COMPUTE(0, 1);
                COMPUTE(0, 2);

                COMPUTE(1, 0);
                COMPUTE(1, 1);
                COMPUTE(1, 2);

                COMPUTE(2, 0);
                COMPUTE(2, 1);
                COMPUTE(2, 2);

                COMPUTE(3, 0);
                COMPUTE(3, 1);
                COMPUTE(3, 2);
            }
            D00 = _mm256_hadd_epi32(D00, D01);
            D02 = _mm256_hadd_epi32(D02, D03);

            D10 = _mm256_hadd_epi32(D10, D11);
            D12 = _mm256_hadd_epi32(D12, D13);

            D20 = _mm256_hadd_epi32(D20, D21);
            D22 = _mm256_hadd_epi32(D22, D23);

            D30 = _mm256_hadd_epi32(D30, D31);
            D32 = _mm256_hadd_epi32(D32, D33);
            
            D00 = _mm256_hadd_epi32(D00, D02);
            D10 = _mm256_hadd_epi32(D10, D12);
            D20 = _mm256_hadd_epi32(D20, D22);
            D30 = _mm256_hadd_epi32(D30, D32);

            EXTRACT_ADD(0);
            EXTRACT_ADD(1);
            EXTRACT_ADD(2);
            EXTRACT_ADD(3);

            auto biasValue = _mm_loadu_si128((__m128i*)(bias_dz));
            d0 = _mm_add_epi32(d0, biasValue);
            d1 = _mm_add_epi32(d1, biasValue);
            d2 = _mm_add_epi32(d2, biasValue);
            d3 = _mm_add_epi32(d3, biasValue);

            auto scaleValue = _mm_loadu_ps(scale_dz);
            __m128 f0 = _mm_cvtepi32_ps(d0);
            __m128 f1 = _mm_cvtepi32_ps(d1);
            __m128 f2 = _mm_cvtepi32_ps(d2);
            __m128 f3 = _mm_cvtepi32_ps(d3);
            f0 = _mm_mul_ps(f0, scaleValue);
            f1 = _mm_mul_ps(f1, scaleValue);
            f2 = _mm_mul_ps(f2, scaleValue);
            f3 = _mm_mul_ps(f3, scaleValue);
            f0 = _mm_min_ps(f0, maxValue);
            f1 = _mm_min_ps(f1, maxValue);
            f2 = _mm_min_ps(f2, maxValue);
            f3 = _mm_min_ps(f3, maxValue);
            f0 = _mm_max_ps(f0, minValue);
            f1 = _mm_max_ps(f1, minValue);
            f2 = _mm_max_ps(f2, minValue);
            f3 = _mm_max_ps(f3, minValue);
            auto m0 = _mm_cmplt_ps(f0, zero128);
            auto m1 = _mm_cmplt_ps(f1, zero128);
            auto m2 = _mm_cmplt_ps(f2, zero128);
            auto m3 = _mm_cmplt_ps(f3, zero128);
            m0 = _mm_blendv_ps(plus, minus, m0);
            m1 = _mm_blendv_ps(plus, minus, m1);
            m2 = _mm_blendv_ps(plus, minus, m2);
            m3 = _mm_blendv_ps(plus, minus, m3);
            f0 = _mm_add_ps(f0, m0);
            f1 = _mm_add_ps(f1, m1);
            f2 = _mm_add_ps(f2, m2);
            f3 = _mm_add_ps(f3, m3);
            // 3: _MM_FROUND_TO_ZERO
            d0 = _mm_cvtps_epi32(_mm_round_ps(f0, 3));
            d1 = _mm_cvtps_epi32(_mm_round_ps(f1, 3));
            d2 = _mm_cvtps_epi32(_mm_round_ps(f2, 3));
            d3 = _mm_cvtps_epi32(_mm_round_ps(f3, 3));
            
            // Int32 -> Int8
            d0 = _mm_packs_epi32(d0, d1);
            d2 = _mm_packs_epi32(d2, d3);
            d0 = _mm_packs_epi16(d0, d2);
            int32_t tempV[4];
            _mm_storeu_si128((__m128i*)tempV, d0);
            for (int j=0; j<realDst; ++j) {
                ((int32_t*)dst_x)[j] = tempV[j];
            }
        }
    }
    if (2 == realDst) {
        for (int dz = 0; dz < dst_depth_quad; ++dz) {
            const auto weight_dz = weight + dz * src_depth_quad * (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT);
            const auto bias_dz   = post->bias + dz * GEMM_INT8_UNIT;
            const auto scale_dz  = post->scale + dz * GEMM_INT8_UNIT;
            auto dst_z           = dst + dz * dst_step_tmp;
            const auto src_x   = src;
            auto dst_x         = dst_z;
            __m256i D00 = _mm256_set1_epi32(0);
            __m256i D01 = _mm256_set1_epi32(0);
            __m256i D02 = _mm256_set1_epi32(0);
            __m256i D03 = _mm256_set1_epi32(0);

            __m256i D10 = _mm256_set1_epi32(0);
            __m256i D11 = _mm256_set1_epi32(0);
            __m256i D12 = _mm256_set1_epi32(0);
            __m256i D13 = _mm256_set1_epi32(0);

            __m256i D20 = _mm256_set1_epi32(0);
            __m256i D21 = _mm256_set1_epi32(0);
            __m256i D22 = _mm256_set1_epi32(0);
            __m256i D23 = _mm256_set1_epi32(0);

            __m256i D30 = _mm256_set1_epi32(0);
            __m256i D31 = _mm256_set1_epi32(0);
            __m256i D32 = _mm256_set1_epi32(0);
            __m256i D33 = _mm256_set1_epi32(0);

            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT) * sz;
                const auto src_z     = src_x + sz * GEMM_INT8_DST_XUNIT * GEMM_INT8_SRC_UNIT;
                auto w0 = mm_loadu_si128(weight_sz + GEMM_INT8_SRC_UNIT * 0);
                auto w1 = mm_loadu_si128(weight_sz + GEMM_INT8_SRC_UNIT * 1);
                auto w2 = mm_loadu_si128(weight_sz + GEMM_INT8_SRC_UNIT * 2);
                auto w3 = mm_loadu_si128(weight_sz + GEMM_INT8_SRC_UNIT * 3);
                auto W0 = _mm256_cvtepi8_epi16(w0);
                auto W1 = _mm256_cvtepi8_epi16(w1);
                auto W2 = _mm256_cvtepi8_epi16(w2);
                auto W3 = _mm256_cvtepi8_epi16(w3);

                auto s0 = mm_loadu_si128(src_z + GEMM_INT8_SRC_UNIT * 0);
                auto s1 = mm_loadu_si128(src_z + GEMM_INT8_SRC_UNIT * 1);
                auto S0 = _mm256_cvtepu8_epi16(s0);
                auto S1 = _mm256_cvtepu8_epi16(s1);
                
                COMPUTE(0, 0);
                COMPUTE(0, 1);

                COMPUTE(1, 0);
                COMPUTE(1, 1);

                COMPUTE(2, 0);
                COMPUTE(2, 1);

                COMPUTE(3, 0);
                COMPUTE(3, 1);
            }
            D00 = _mm256_hadd_epi32(D00, D01);
            D02 = _mm256_hadd_epi32(D02, D03);

            D10 = _mm256_hadd_epi32(D10, D11);
            D12 = _mm256_hadd_epi32(D12, D13);

            D20 = _mm256_hadd_epi32(D20, D21);
            D22 = _mm256_hadd_epi32(D22, D23);

            D30 = _mm256_hadd_epi32(D30, D31);
            D32 = _mm256_hadd_epi32(D32, D33);
            
            D00 = _mm256_hadd_epi32(D00, D02);
            D10 = _mm256_hadd_epi32(D10, D12);
            D20 = _mm256_hadd_epi32(D20, D22);
            D30 = _mm256_hadd_epi32(D30, D32);

            EXTRACT_ADD(0);
            EXTRACT_ADD(1);
            EXTRACT_ADD(2);
            EXTRACT_ADD(3);

            auto biasValue = _mm_loadu_si128((__m128i*)(bias_dz));
            d0 = _mm_add_epi32(d0, biasValue);
            d1 = _mm_add_epi32(d1, biasValue);
            d2 = _mm_add_epi32(d2, biasValue);
            d3 = _mm_add_epi32(d3, biasValue);

            auto scaleValue = _mm_loadu_ps(scale_dz);
            __m128 f0 = _mm_cvtepi32_ps(d0);
            __m128 f1 = _mm_cvtepi32_ps(d1);
            __m128 f2 = _mm_cvtepi32_ps(d2);
            __m128 f3 = _mm_cvtepi32_ps(d3);
            f0 = _mm_mul_ps(f0, scaleValue);
            f1 = _mm_mul_ps(f1, scaleValue);
            f2 = _mm_mul_ps(f2, scaleValue);
            f3 = _mm_mul_ps(f3, scaleValue);
            f0 = _mm_min_ps(f0, maxValue);
            f1 = _mm_min_ps(f1, maxValue);
            f2 = _mm_min_ps(f2, maxValue);
            f3 = _mm_min_ps(f3, maxValue);
            f0 = _mm_max_ps(f0, minValue);
            f1 = _mm_max_ps(f1, minValue);
            f2 = _mm_max_ps(f2, minValue);
            f3 = _mm_max_ps(f3, minValue);
            auto m0 = _mm_cmplt_ps(f0, zero128);
            auto m1 = _mm_cmplt_ps(f1, zero128);
            auto m2 = _mm_cmplt_ps(f2, zero128);
            auto m3 = _mm_cmplt_ps(f3, zero128);
            m0 = _mm_blendv_ps(plus, minus, m0);
            m1 = _mm_blendv_ps(plus, minus, m1);
            m2 = _mm_blendv_ps(plus, minus, m2);
            m3 = _mm_blendv_ps(plus, minus, m3);
            f0 = _mm_add_ps(f0, m0);
            f1 = _mm_add_ps(f1, m1);
            f2 = _mm_add_ps(f2, m2);
            f3 = _mm_add_ps(f3, m3);
            // 3: _MM_FROUND_TO_ZERO
            d0 = _mm_cvtps_epi32(_mm_round_ps(f0, 3));
            d1 = _mm_cvtps_epi32(_mm_round_ps(f1, 3));
            d2 = _mm_cvtps_epi32(_mm_round_ps(f2, 3));
            d3 = _mm_cvtps_epi32(_mm_round_ps(f3, 3));
            
            // Int32 -> Int8
            d0 = _mm_packs_epi32(d0, d1);
            d2 = _mm_packs_epi32(d2, d3);
            d0 = _mm_packs_epi16(d0, d2);
            int32_t tempV[4];
            _mm_storeu_si128((__m128i*)tempV, d0);
            for (int j=0; j<realDst; ++j) {
                ((int32_t*)dst_x)[j] = tempV[j];
            }
        }
    }
    if (1 == realDst) {
        for (int dz = 0; dz < dst_depth_quad; ++dz) {
            const auto weight_dz = weight + dz * src_depth_quad * (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT);
            const auto bias_dz   = post->bias + dz * GEMM_INT8_UNIT;
            const auto scale_dz  = post->scale + dz * GEMM_INT8_UNIT;
            auto dst_z           = dst + dz * dst_step_tmp;
            const auto src_x   = src;
            auto dst_x         = dst_z;
            __m256i D00 = _mm256_set1_epi32(0);
            __m256i D01 = _mm256_set1_epi32(0);
            __m256i D02 = _mm256_set1_epi32(0);
            __m256i D03 = _mm256_set1_epi32(0);

            __m256i D10 = _mm256_set1_epi32(0);
            __m256i D11 = _mm256_set1_epi32(0);
            __m256i D12 = _mm256_set1_epi32(0);
            __m256i D13 = _mm256_set1_epi32(0);

            __m256i D20 = _mm256_set1_epi32(0);
            __m256i D21 = _mm256_set1_epi32(0);
            __m256i D22 = _mm256_set1_epi32(0);
            __m256i D23 = _mm256_set1_epi32(0);

            __m256i D30 = _mm256_set1_epi32(0);
            __m256i D31 = _mm256_set1_epi32(0);
            __m256i D32 = _mm256_set1_epi32(0);
            __m256i D33 = _mm256_set1_epi32(0);

            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT) * sz;
                const auto src_z     = src_x + sz * GEMM_INT8_DST_XUNIT * GEMM_INT8_SRC_UNIT;
                auto w0 = mm_loadu_si128(weight_sz + GEMM_INT8_SRC_UNIT * 0);
                auto w1 = mm_loadu_si128(weight_sz + GEMM_INT8_SRC_UNIT * 1);
                auto w2 = mm_loadu_si128(weight_sz + GEMM_INT8_SRC_UNIT * 2);
                auto w3 = mm_loadu_si128(weight_sz + GEMM_INT8_SRC_UNIT * 3);
                auto W0 = _mm256_cvtepi8_epi16(w0);
                auto W1 = _mm256_cvtepi8_epi16(w1);
                auto W2 = _mm256_cvtepi8_epi16(w2);
                auto W3 = _mm256_cvtepi8_epi16(w3);

                auto s0 = mm_loadu_si128(src_z + GEMM_INT8_SRC_UNIT * 0);
                auto S0 = _mm256_cvtepu8_epi16(s0);

                COMPUTE(0, 0);

                COMPUTE(1, 0);

                COMPUTE(2, 0);

                COMPUTE(3, 0);
            }
            D00 = _mm256_hadd_epi32(D00, D01);
            D02 = _mm256_hadd_epi32(D02, D03);

            D10 = _mm256_hadd_epi32(D10, D11);
            D12 = _mm256_hadd_epi32(D12, D13);

            D20 = _mm256_hadd_epi32(D20, D21);
            D22 = _mm256_hadd_epi32(D22, D23);

            D30 = _mm256_hadd_epi32(D30, D31);
            D32 = _mm256_hadd_epi32(D32, D33);
            
            D00 = _mm256_hadd_epi32(D00, D02);
            D10 = _mm256_hadd_epi32(D10, D12);
            D20 = _mm256_hadd_epi32(D20, D22);
            D30 = _mm256_hadd_epi32(D30, D32);

            EXTRACT_ADD(0);
            EXTRACT_ADD(1);
            EXTRACT_ADD(2);
            EXTRACT_ADD(3);

            auto biasValue = _mm_loadu_si128((__m128i*)(bias_dz));
            d0 = _mm_add_epi32(d0, biasValue);
            d1 = _mm_add_epi32(d1, biasValue);
            d2 = _mm_add_epi32(d2, biasValue);
            d3 = _mm_add_epi32(d3, biasValue);

            auto scaleValue = _mm_loadu_ps(scale_dz);
            __m128 f0 = _mm_cvtepi32_ps(d0);
            __m128 f1 = _mm_cvtepi32_ps(d1);
            __m128 f2 = _mm_cvtepi32_ps(d2);
            __m128 f3 = _mm_cvtepi32_ps(d3);
            f0 = _mm_mul_ps(f0, scaleValue);
            f1 = _mm_mul_ps(f1, scaleValue);
            f2 = _mm_mul_ps(f2, scaleValue);
            f3 = _mm_mul_ps(f3, scaleValue);
            f0 = _mm_min_ps(f0, maxValue);
            f1 = _mm_min_ps(f1, maxValue);
            f2 = _mm_min_ps(f2, maxValue);
            f3 = _mm_min_ps(f3, maxValue);
            f0 = _mm_max_ps(f0, minValue);
            f1 = _mm_max_ps(f1, minValue);
            f2 = _mm_max_ps(f2, minValue);
            f3 = _mm_max_ps(f3, minValue);
            auto m0 = _mm_cmplt_ps(f0, zero128);
            auto m1 = _mm_cmplt_ps(f1, zero128);
            auto m2 = _mm_cmplt_ps(f2, zero128);
            auto m3 = _mm_cmplt_ps(f3, zero128);
            m0 = _mm_blendv_ps(plus, minus, m0);
            m1 = _mm_blendv_ps(plus, minus, m1);
            m2 = _mm_blendv_ps(plus, minus, m2);
            m3 = _mm_blendv_ps(plus, minus, m3);
            f0 = _mm_add_ps(f0, m0);
            f1 = _mm_add_ps(f1, m1);
            f2 = _mm_add_ps(f2, m2);
            f3 = _mm_add_ps(f3, m3);
            // 3: _MM_FROUND_TO_ZERO
            d0 = _mm_cvtps_epi32(_mm_round_ps(f0, 3));
            d1 = _mm_cvtps_epi32(_mm_round_ps(f1, 3));
            d2 = _mm_cvtps_epi32(_mm_round_ps(f2, 3));
            d3 = _mm_cvtps_epi32(_mm_round_ps(f3, 3));
            
            // Int32 -> Int8
            d0 = _mm_packs_epi32(d0, d1);
            d2 = _mm_packs_epi32(d2, d3);
            d0 = _mm_packs_epi16(d0, d2);
            int32_t tempV[4];
            _mm_storeu_si128((__m128i*)tempV, d0);
            for (int j=0; j<realDst; ++j) {
                ((int32_t*)dst_x)[j] = tempV[j];
            }
        }
    }
}
void _AVX_MNNGemmInt8AddBiasScale_16x4_Unit_Fast(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, const QuanPostTreatParameters* post, size_t realDst) {
    const auto dst_step_tmp = dst_step / sizeof(int8_t);
    auto zero128 = _mm256_set1_ps(0.0f);
    auto minValue = _mm256_set1_ps(post->minValue);
    auto maxValue = _mm256_set1_ps(post->maxValue);
    auto plus = _mm256_set1_ps(0.5f);
    auto minus = _mm256_set1_ps(-0.5f);
    auto oneValue = _mm256_set1_epi16(1);
    if (4 == realDst) {
        for (int dz = 0; dz < dst_depth_quad; ++dz) {
            const auto weight_dz = weight + dz * src_depth_quad * (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT);
            const auto bias_dz   = post->bias + dz * GEMM_INT8_UNIT;
            const auto scale_dz  = post->scale + dz * GEMM_INT8_UNIT;
            auto dst_z           = dst + dz * dst_step_tmp;
            const auto src_x   = src;
            auto dst_x         = dst_z;
            __m256i D00 = _mm256_set1_epi32(0);
            __m256i D01 = _mm256_set1_epi32(0);

            __m256i D10 = _mm256_set1_epi32(0);
            __m256i D11 = _mm256_set1_epi32(0);

            __m256i D20 = _mm256_set1_epi32(0);
            __m256i D21 = _mm256_set1_epi32(0);

            __m256i D30 = _mm256_set1_epi32(0);
            __m256i D31 = _mm256_set1_epi32(0);

            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT) * sz;
                const auto src_z     = src_x + sz * GEMM_INT8_DST_XUNIT * GEMM_INT8_SRC_UNIT;
                auto w0 = _mm256_loadu_si256((__m256i*)(weight_sz + GEMM_INT8_SRC_UNIT * 0));
                auto w1 = _mm256_loadu_si256((__m256i*)(weight_sz + GEMM_INT8_SRC_UNIT * 2));

                auto s0 = _mm256_broadcastsi128_si256(mm_loadu_si128(src_z + GEMM_INT8_SRC_UNIT * 0));
                auto s1 = _mm256_broadcastsi128_si256(mm_loadu_si128(src_z + GEMM_INT8_SRC_UNIT * 1));
                auto s2 = _mm256_broadcastsi128_si256(mm_loadu_si128(src_z + GEMM_INT8_SRC_UNIT * 2));
                auto s3 = _mm256_broadcastsi128_si256(mm_loadu_si128(src_z + GEMM_INT8_SRC_UNIT * 3));

                D00 = _mm256_add_epi32(D00, _mm256_madd_epi16(_mm256_maddubs_epi16(s0, w0), oneValue));
                D01 = _mm256_add_epi32(D01, _mm256_madd_epi16(_mm256_maddubs_epi16(s0, w1), oneValue));
                D10 = _mm256_add_epi32(D10, _mm256_madd_epi16(_mm256_maddubs_epi16(s1, w0), oneValue));
                D11 = _mm256_add_epi32(D11, _mm256_madd_epi16(_mm256_maddubs_epi16(s1, w1), oneValue));
                D20 = _mm256_add_epi32(D20, _mm256_madd_epi16(_mm256_maddubs_epi16(s2, w0), oneValue));
                D21 = _mm256_add_epi32(D21, _mm256_madd_epi16(_mm256_maddubs_epi16(s2, w1), oneValue));
                D30 = _mm256_add_epi32(D30, _mm256_madd_epi16(_mm256_maddubs_epi16(s3, w0), oneValue));
                D31 = _mm256_add_epi32(D31, _mm256_madd_epi16(_mm256_maddubs_epi16(s3, w1), oneValue));
            }
            
            auto D0 = _mm256_hadd_epi32(_mm256_permute2f128_si256(D00, D01, 32), _mm256_permute2f128_si256(D00, D01, 49));
            auto D1 = _mm256_hadd_epi32(_mm256_permute2f128_si256(D10, D11, 32), _mm256_permute2f128_si256(D10, D11, 49));
            auto D2 = _mm256_hadd_epi32(_mm256_permute2f128_si256(D20, D21, 32), _mm256_permute2f128_si256(D20, D21, 49));
            auto D3 = _mm256_hadd_epi32(_mm256_permute2f128_si256(D30, D31, 32), _mm256_permute2f128_si256(D30, D31, 49));

            D0 = _mm256_hadd_epi32(_mm256_permute2f128_si256(D0, D1, 32), _mm256_permute2f128_si256(D0, D1, 49));
            D2 = _mm256_hadd_epi32(_mm256_permute2f128_si256(D2, D3, 32), _mm256_permute2f128_si256(D2, D3, 49));

            auto biasValue = _mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(bias_dz)));
            D0 = _mm256_add_epi32(D0, biasValue);
            D2 = _mm256_add_epi32(D2, biasValue);

            auto scaleValue = _mm256_castsi256_ps(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)scale_dz)));
            auto f0 = _mm256_cvtepi32_ps(D0);
            auto f1 = _mm256_cvtepi32_ps(D2);
            f0 = _mm256_mul_ps(f0, scaleValue);
            f1 = _mm256_mul_ps(f1, scaleValue);
            f0 = _mm256_min_ps(f0, maxValue);
            f1 = _mm256_min_ps(f1, maxValue);
            f0 = _mm256_max_ps(f0, minValue);
            f1 = _mm256_max_ps(f1, minValue);
            auto m0 = _mm256_cmp_ps(f0, zero128, 1);
            auto m1 = _mm256_cmp_ps(f1, zero128, 1);
            m0 = _mm256_blendv_ps(plus, minus, m0);
            m1 = _mm256_blendv_ps(plus, minus, m1);
            f0 = _mm256_add_ps(f0, m0);
            f1 = _mm256_add_ps(f1, m1);
            // 3: _MM_FROUND_TO_ZERO
            D0 = _mm256_cvtps_epi32(_mm256_round_ps(f0, 3));
            D2 = _mm256_cvtps_epi32(_mm256_round_ps(f1, 3));

            auto d0 = _mm256_extracti128_si256(D0, 0);
            auto d1 = _mm256_extracti128_si256(D0, 1);
            auto d2 = _mm256_extracti128_si256(D2, 0);
            auto d3 = _mm256_extracti128_si256(D2, 1);

            // Int32 -> Int8
            d0 = _mm_packs_epi32(d0, d1);
            d2 = _mm_packs_epi32(d2, d3);
            d0 = _mm_packs_epi16(d0, d2);
            _mm_storeu_ps((float*)dst_x, _mm_castsi128_ps(d0));
        }
        return;
    }
    if (3 == realDst) {
        for (int dz = 0; dz < dst_depth_quad; ++dz) {
            const auto weight_dz = weight + dz * src_depth_quad * (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT);
            const auto bias_dz   = post->bias + dz * GEMM_INT8_UNIT;
            const auto scale_dz  = post->scale + dz * GEMM_INT8_UNIT;
            auto dst_z           = dst + dz * dst_step_tmp;
            const auto src_x   = src;
            auto dst_x         = dst_z;
            __m256i D00 = _mm256_set1_epi32(0);
            __m256i D01 = _mm256_set1_epi32(0);

            __m256i D10 = _mm256_set1_epi32(0);
            __m256i D11 = _mm256_set1_epi32(0);

            __m256i D20 = _mm256_set1_epi32(0);
            __m256i D21 = _mm256_set1_epi32(0);

            __m256i D30 = _mm256_set1_epi32(0);
            __m256i D31 = _mm256_set1_epi32(0);

            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT) * sz;
                const auto src_z     = src_x + sz * GEMM_INT8_DST_XUNIT * GEMM_INT8_SRC_UNIT;
                auto w0 = _mm256_loadu_si256((__m256i*)(weight_sz + GEMM_INT8_SRC_UNIT * 0));
                auto w1 = _mm256_loadu_si256((__m256i*)(weight_sz + GEMM_INT8_SRC_UNIT * 2));

                auto s0 = _mm256_broadcastsi128_si256(mm_loadu_si128(src_z + GEMM_INT8_SRC_UNIT * 0));
                auto s1 = _mm256_broadcastsi128_si256(mm_loadu_si128(src_z + GEMM_INT8_SRC_UNIT * 1));
                auto s2 = _mm256_broadcastsi128_si256(mm_loadu_si128(src_z + GEMM_INT8_SRC_UNIT * 2));

                D00 = _mm256_add_epi32(D00, _mm256_madd_epi16(_mm256_maddubs_epi16(s0, w0), oneValue));
                D01 = _mm256_add_epi32(D01, _mm256_madd_epi16(_mm256_maddubs_epi16(s0, w1), oneValue));
                D10 = _mm256_add_epi32(D10, _mm256_madd_epi16(_mm256_maddubs_epi16(s1, w0), oneValue));
                D11 = _mm256_add_epi32(D11, _mm256_madd_epi16(_mm256_maddubs_epi16(s1, w1), oneValue));
                D20 = _mm256_add_epi32(D20, _mm256_madd_epi16(_mm256_maddubs_epi16(s2, w0), oneValue));
                D21 = _mm256_add_epi32(D21, _mm256_madd_epi16(_mm256_maddubs_epi16(s2, w1), oneValue));
            }
            
            auto D0 = _mm256_hadd_epi32(_mm256_permute2f128_si256(D00, D01, 32), _mm256_permute2f128_si256(D00, D01, 49));
            auto D1 = _mm256_hadd_epi32(_mm256_permute2f128_si256(D10, D11, 32), _mm256_permute2f128_si256(D10, D11, 49));
            auto D2 = _mm256_hadd_epi32(_mm256_permute2f128_si256(D20, D21, 32), _mm256_permute2f128_si256(D20, D21, 49));
            auto D3 = _mm256_hadd_epi32(_mm256_permute2f128_si256(D30, D31, 32), _mm256_permute2f128_si256(D30, D31, 49));

            D0 = _mm256_hadd_epi32(_mm256_permute2f128_si256(D0, D1, 32), _mm256_permute2f128_si256(D0, D1, 49));
            D2 = _mm256_hadd_epi32(_mm256_permute2f128_si256(D2, D3, 32), _mm256_permute2f128_si256(D2, D3, 49));

            auto biasValue = _mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(bias_dz)));
            D0 = _mm256_add_epi32(D0, biasValue);
            D2 = _mm256_add_epi32(D2, biasValue);

            auto scaleValue = _mm256_castsi256_ps(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)scale_dz)));
            auto f0 = _mm256_cvtepi32_ps(D0);
            auto f1 = _mm256_cvtepi32_ps(D2);
            f0 = _mm256_mul_ps(f0, scaleValue);
            f1 = _mm256_mul_ps(f1, scaleValue);
            f0 = _mm256_min_ps(f0, maxValue);
            f1 = _mm256_min_ps(f1, maxValue);
            f0 = _mm256_max_ps(f0, minValue);
            f1 = _mm256_max_ps(f1, minValue);
            auto m0 = _mm256_cmp_ps(f0, zero128, 1);
            auto m1 = _mm256_cmp_ps(f1, zero128, 1);
            m0 = _mm256_blendv_ps(plus, minus, m0);
            m1 = _mm256_blendv_ps(plus, minus, m1);
            f0 = _mm256_add_ps(f0, m0);
            f1 = _mm256_add_ps(f1, m1);
            // 3: _MM_FROUND_TO_ZERO
            D0 = _mm256_cvtps_epi32(_mm256_round_ps(f0, 3));
            D2 = _mm256_cvtps_epi32(_mm256_round_ps(f1, 3));

            auto d0 = _mm256_extracti128_si256(D0, 0);
            auto d1 = _mm256_extracti128_si256(D0, 1);
            auto d2 = _mm256_extracti128_si256(D2, 0);
            auto d3 = _mm256_extracti128_si256(D2, 1);

            // Int32 -> Int8
            d0 = _mm_packs_epi32(d0, d1);
            d2 = _mm_packs_epi32(d2, d3);
            d0 = _mm_packs_epi16(d0, d2);
            int32_t tempV[4];
            _mm_storeu_si128((__m128i*)tempV, d0);
            for (int j=0; j<realDst; ++j) {
                ((int32_t*)dst_x)[j] = tempV[j];
            }
        }
        return;
    }
    if (2 == realDst) {
        for (int dz = 0; dz < dst_depth_quad; ++dz) {
            const auto weight_dz = weight + dz * src_depth_quad * (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT);
            const auto bias_dz   = post->bias + dz * GEMM_INT8_UNIT;
            const auto scale_dz  = post->scale + dz * GEMM_INT8_UNIT;
            auto dst_z           = dst + dz * dst_step_tmp;
            const auto src_x   = src;
            auto dst_x         = dst_z;
            __m256i D00 = _mm256_set1_epi32(0);
            __m256i D01 = _mm256_set1_epi32(0);

            __m256i D10 = _mm256_set1_epi32(0);
            __m256i D11 = _mm256_set1_epi32(0);

            __m256i D20 = _mm256_set1_epi32(0);
            __m256i D21 = _mm256_set1_epi32(0);

            __m256i D30 = _mm256_set1_epi32(0);
            __m256i D31 = _mm256_set1_epi32(0);

            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT) * sz;
                const auto src_z     = src_x + sz * GEMM_INT8_DST_XUNIT * GEMM_INT8_SRC_UNIT;
                auto w0 = _mm256_loadu_si256((__m256i*)(weight_sz + GEMM_INT8_SRC_UNIT * 0));
                auto w1 = _mm256_loadu_si256((__m256i*)(weight_sz + GEMM_INT8_SRC_UNIT * 2));

                auto s0 = _mm256_broadcastsi128_si256(mm_loadu_si128(src_z + GEMM_INT8_SRC_UNIT * 0));
                auto s1 = _mm256_broadcastsi128_si256(mm_loadu_si128(src_z + GEMM_INT8_SRC_UNIT * 1));

                D00 = _mm256_add_epi32(D00, _mm256_madd_epi16(_mm256_maddubs_epi16(s0, w0), oneValue));
                D01 = _mm256_add_epi32(D01, _mm256_madd_epi16(_mm256_maddubs_epi16(s0, w1), oneValue));
                D10 = _mm256_add_epi32(D10, _mm256_madd_epi16(_mm256_maddubs_epi16(s1, w0), oneValue));
                D11 = _mm256_add_epi32(D11, _mm256_madd_epi16(_mm256_maddubs_epi16(s1, w1), oneValue));
            }
            
            auto D0 = _mm256_hadd_epi32(_mm256_permute2f128_si256(D00, D01, 32), _mm256_permute2f128_si256(D00, D01, 49));
            auto D1 = _mm256_hadd_epi32(_mm256_permute2f128_si256(D10, D11, 32), _mm256_permute2f128_si256(D10, D11, 49));
            auto D2 = _mm256_hadd_epi32(_mm256_permute2f128_si256(D20, D21, 32), _mm256_permute2f128_si256(D20, D21, 49));
            auto D3 = _mm256_hadd_epi32(_mm256_permute2f128_si256(D30, D31, 32), _mm256_permute2f128_si256(D30, D31, 49));

            D0 = _mm256_hadd_epi32(_mm256_permute2f128_si256(D0, D1, 32), _mm256_permute2f128_si256(D0, D1, 49));
            D2 = _mm256_hadd_epi32(_mm256_permute2f128_si256(D2, D3, 32), _mm256_permute2f128_si256(D2, D3, 49));

            auto biasValue = _mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(bias_dz)));
            D0 = _mm256_add_epi32(D0, biasValue);
            D2 = _mm256_add_epi32(D2, biasValue);

            auto scaleValue = _mm256_castsi256_ps(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)scale_dz)));
            auto f0 = _mm256_cvtepi32_ps(D0);
            auto f1 = _mm256_cvtepi32_ps(D2);
            f0 = _mm256_mul_ps(f0, scaleValue);
            f1 = _mm256_mul_ps(f1, scaleValue);
            f0 = _mm256_min_ps(f0, maxValue);
            f1 = _mm256_min_ps(f1, maxValue);
            f0 = _mm256_max_ps(f0, minValue);
            f1 = _mm256_max_ps(f1, minValue);
            auto m0 = _mm256_cmp_ps(f0, zero128, 1);
            auto m1 = _mm256_cmp_ps(f1, zero128, 1);
            m0 = _mm256_blendv_ps(plus, minus, m0);
            m1 = _mm256_blendv_ps(plus, minus, m1);
            f0 = _mm256_add_ps(f0, m0);
            f1 = _mm256_add_ps(f1, m1);
            // 3: _MM_FROUND_TO_ZERO
            D0 = _mm256_cvtps_epi32(_mm256_round_ps(f0, 3));
            D2 = _mm256_cvtps_epi32(_mm256_round_ps(f1, 3));

            auto d0 = _mm256_extracti128_si256(D0, 0);
            auto d1 = _mm256_extracti128_si256(D0, 1);
            auto d2 = _mm256_extracti128_si256(D2, 0);
            auto d3 = _mm256_extracti128_si256(D2, 1);

            // Int32 -> Int8
            d0 = _mm_packs_epi32(d0, d1);
            d2 = _mm_packs_epi32(d2, d3);
            d0 = _mm_packs_epi16(d0, d2);
            int32_t tempV[4];
            _mm_storeu_si128((__m128i*)tempV, d0);
            for (int j=0; j<realDst; ++j) {
                ((int32_t*)dst_x)[j] = tempV[j];
            }
        }
        return;
    }
    if (1 == realDst) {
        for (int dz = 0; dz < dst_depth_quad; ++dz) {
            const auto weight_dz = weight + dz * src_depth_quad * (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT);
            const auto bias_dz   = post->bias + dz * GEMM_INT8_UNIT;
            const auto scale_dz  = post->scale + dz * GEMM_INT8_UNIT;
            auto dst_z           = dst + dz * dst_step_tmp;
            const auto src_x   = src;
            auto dst_x         = dst_z;
            __m256i D00 = _mm256_set1_epi32(0);
            __m256i D01 = _mm256_set1_epi32(0);

            __m256i D10 = _mm256_set1_epi32(0);
            __m256i D11 = _mm256_set1_epi32(0);

            __m256i D20 = _mm256_set1_epi32(0);
            __m256i D21 = _mm256_set1_epi32(0);

            __m256i D30 = _mm256_set1_epi32(0);
            __m256i D31 = _mm256_set1_epi32(0);

            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT) * sz;
                const auto src_z     = src_x + sz * GEMM_INT8_DST_XUNIT * GEMM_INT8_SRC_UNIT;
                auto w0 = _mm256_loadu_si256((__m256i*)(weight_sz + GEMM_INT8_SRC_UNIT * 0));
                auto w1 = _mm256_loadu_si256((__m256i*)(weight_sz + GEMM_INT8_SRC_UNIT * 2));

                auto s0 = _mm256_broadcastsi128_si256(mm_loadu_si128(src_z + GEMM_INT8_SRC_UNIT * 0));

                D00 = _mm256_add_epi32(D00, _mm256_madd_epi16(_mm256_maddubs_epi16(s0, w0), oneValue));
                D01 = _mm256_add_epi32(D01, _mm256_madd_epi16(_mm256_maddubs_epi16(s0, w1), oneValue));
            }
            
            auto D0 = _mm256_hadd_epi32(_mm256_permute2f128_si256(D00, D01, 32), _mm256_permute2f128_si256(D00, D01, 49));
            auto D1 = _mm256_hadd_epi32(_mm256_permute2f128_si256(D10, D11, 32), _mm256_permute2f128_si256(D10, D11, 49));
            auto D2 = _mm256_hadd_epi32(_mm256_permute2f128_si256(D20, D21, 32), _mm256_permute2f128_si256(D20, D21, 49));
            auto D3 = _mm256_hadd_epi32(_mm256_permute2f128_si256(D30, D31, 32), _mm256_permute2f128_si256(D30, D31, 49));

            D0 = _mm256_hadd_epi32(_mm256_permute2f128_si256(D0, D1, 32), _mm256_permute2f128_si256(D0, D1, 49));
            D2 = _mm256_hadd_epi32(_mm256_permute2f128_si256(D2, D3, 32), _mm256_permute2f128_si256(D2, D3, 49));

            auto biasValue = _mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(bias_dz)));
            D0 = _mm256_add_epi32(D0, biasValue);
            D2 = _mm256_add_epi32(D2, biasValue);

            auto scaleValue = _mm256_castsi256_ps(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)scale_dz)));
            auto f0 = _mm256_cvtepi32_ps(D0);
            auto f1 = _mm256_cvtepi32_ps(D2);
            f0 = _mm256_mul_ps(f0, scaleValue);
            f1 = _mm256_mul_ps(f1, scaleValue);
            f0 = _mm256_min_ps(f0, maxValue);
            f1 = _mm256_min_ps(f1, maxValue);
            f0 = _mm256_max_ps(f0, minValue);
            f1 = _mm256_max_ps(f1, minValue);
            auto m0 = _mm256_cmp_ps(f0, zero128, 1);
            auto m1 = _mm256_cmp_ps(f1, zero128, 1);
            m0 = _mm256_blendv_ps(plus, minus, m0);
            m1 = _mm256_blendv_ps(plus, minus, m1);
            f0 = _mm256_add_ps(f0, m0);
            f1 = _mm256_add_ps(f1, m1);
            // 3: _MM_FROUND_TO_ZERO
            D0 = _mm256_cvtps_epi32(_mm256_round_ps(f0, 3));
            D2 = _mm256_cvtps_epi32(_mm256_round_ps(f1, 3));

            auto d0 = _mm256_extracti128_si256(D0, 0);
            auto d1 = _mm256_extracti128_si256(D0, 1);
            auto d2 = _mm256_extracti128_si256(D2, 0);
            auto d3 = _mm256_extracti128_si256(D2, 1);

            // Int32 -> Int8
            d0 = _mm_packs_epi32(d0, d1);
            d2 = _mm_packs_epi32(d2, d3);
            d0 = _mm_packs_epi16(d0, d2);
            int32_t tempV[4];
            _mm_storeu_si128((__m128i*)tempV, d0);
            for (int j=0; j<realDst; ++j) {
                ((int32_t*)dst_x)[j] = tempV[j];
            }
        }
        return;
    }
}
