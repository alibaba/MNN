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
void _SSE_MNNPackC4ForMatMul_A(float* dest, const float* source, size_t e, size_t l, size_t eReal) {
    const int pack   = 12;
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
            auto s00  = _mm_loadu_ps(srcX + 0 * 4);
            auto s01  = _mm_loadu_ps(srcX + 1 * 4);
            auto s02  = _mm_loadu_ps(srcX + 2 * 4);
            auto s03  = _mm_loadu_ps(srcX + 3 * 4);
            auto s10  = _mm_loadu_ps(srcX + 4 * 4);
            auto s11  = _mm_loadu_ps(srcX + 5 * 4);
            auto s12  = _mm_loadu_ps(srcX + 6 * 4);
            auto s13  = _mm_loadu_ps(srcX + 7 * 4);
            auto s20  = _mm_loadu_ps(srcX + 8 * 4);
            auto s21  = _mm_loadu_ps(srcX + 9 * 4);
            auto s22  = _mm_loadu_ps(srcX + 10 * 4);
            auto s23  = _mm_loadu_ps(srcX + 11 * 4);

            _MM_TRANSPOSE4_PS(s00, s01, s02, s03);
            _MM_TRANSPOSE4_PS(s10, s11, s12, s13);
            _MM_TRANSPOSE4_PS(s20, s21, s22, s23);

#define STORE_TEMP(i)                               \
    _mm_storeu_ps(dstX + 4 * (3 * i + 0), s##0##i); \
    _mm_storeu_ps(dstX + 4 * (3 * i + 1), s##1##i); \
    _mm_storeu_ps(dstX + 4 * (3 * i + 2), s##2##i);

            STORE_TEMP(0);
            STORE_TEMP(1);
            STORE_TEMP(2);
            STORE_TEMP(3);
        }
        if (lRes == 0) {
            continue;
        }
        auto srcX = srcY + lC4 * 4 * eReal;
        auto dstX = dstY + lC4 * pack * 4;
        auto s00  = _mm_loadu_ps(srcX + 0 * 4);
        auto s01  = _mm_loadu_ps(srcX + 1 * 4);
        auto s02  = _mm_loadu_ps(srcX + 2 * 4);
        auto s03  = _mm_loadu_ps(srcX + 3 * 4);
        auto s10  = _mm_loadu_ps(srcX + 4 * 4);
        auto s11  = _mm_loadu_ps(srcX + 5 * 4);
        auto s12  = _mm_loadu_ps(srcX + 6 * 4);
        auto s13  = _mm_loadu_ps(srcX + 7 * 4);
        auto s20  = _mm_loadu_ps(srcX + 8 * 4);
        auto s21  = _mm_loadu_ps(srcX + 9 * 4);
        auto s22  = _mm_loadu_ps(srcX + 10 * 4);
        auto s23  = _mm_loadu_ps(srcX + 11 * 4);

        _MM_TRANSPOSE4_PS(s00, s01, s02, s03);
        _MM_TRANSPOSE4_PS(s10, s11, s12, s13);
        _MM_TRANSPOSE4_PS(s20, s21, s22, s23);
        if (lRes == 3) {
            STORE_TEMP(0);
            STORE_TEMP(1);
            STORE_TEMP(2);
        } else if (lRes == 2) {
            STORE_TEMP(0);
            STORE_TEMP(1);
        } else {
            STORE_TEMP(0);
        }
    }
    // Down
    {
        auto eLast    = e - eRemain;
        auto lastDest = dest + ePack * pack * l;
        for (int y = eRemain; y < e; ++y) {
            auto yR = y - eRemain;
            for (int x = 0; x < l; ++x) {
                auto xR                  = x % 4;
                auto xC                  = x / 4;
                lastDest[x * eLast + yR] = source[xC * eReal * 4 + y * 4 + xR];
            }
        }
    }
}

void _SSE_GemmPostTreat(float* C, size_t eSize, const size_t* parameter, const float* postParameters,
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
    auto minValue     = _mm_set1_ps(postParameters[2]);
    auto maxValue     = _mm_set1_ps(postParameters[3]);
    if (nullptr != bias) {
        for (int y = 0; y < hC4; ++y) {
            auto biasValue = _mm_loadu_ps(bias + 4 * y);
            auto dst       = C + y * cStride;
            for (int x = 0; x < eSize; ++x) {
                auto sum = _mm_add_ps(biasValue, _mm_loadu_ps(dst + 4 * x));
                sum      = _mm_max_ps(sum, minValue);
                sum      = _mm_min_ps(sum, maxValue);
                _mm_storeu_ps(dst + 4 * x, sum);
            }
        }
    } else {
        for (int y = 0; y < hC4; ++y) {
            auto dst = C + y * cStride;
            for (int x = 0; x < eSize; ++x) {
                auto sum = _mm_loadu_ps(dst + 4 * x);
                sum      = _mm_max_ps(sum, minValue);
                sum      = _mm_min_ps(sum, maxValue);
                _mm_storeu_ps(dst + 4 * x, sum);
            }
        }
    }
}

void _SSE_MNNGemmInt8AddBiasScale_16x4_Unit(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step,
                                            size_t dst_depth_quad, const QuanPostTreatParameters* post) {
    const auto dst_step_tmp = dst_step / sizeof(int8_t);
    __m128i zero = _mm_set1_epi32(0);
    __m128 minValue = _mm_set1_ps(post->minValue);
    __m128 maxValue = _mm_set1_ps(post->maxValue);
    __m128 plus = _mm_set1_ps(0.5f);
    __m128 minus = _mm_set1_ps(-0.5f);
    for (int dz = 0; dz < dst_depth_quad; ++dz) {
        const auto weight_dz = weight + dz * src_depth_quad * (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT);
        const auto bias_dz   = post->bias + dz * GEMM_INT8_UNIT;
        const auto scale_dz  = post->scale + dz * GEMM_INT8_UNIT;
        auto dst_z           = dst + dz * dst_step_tmp;
        const auto src_x   = src;
        auto dst_x         = dst_z;
        __m128i d0 = _mm_set1_epi32(0);
        __m128i d1 = _mm_set1_epi32(0);
        __m128i d2 = _mm_set1_epi32(0);
        __m128i d3 = _mm_set1_epi32(0);

        __m128i e0 = _mm_set1_epi32(0);
        __m128i e1 = _mm_set1_epi32(0);
        __m128i e2 = _mm_set1_epi32(0);
        __m128i e3 = _mm_set1_epi32(0);

        __m128i D0 = _mm_set1_epi32(0);
        __m128i D1 = _mm_set1_epi32(0);
        __m128i D2 = _mm_set1_epi32(0);
        __m128i D3 = _mm_set1_epi32(0);

        __m128i E0 = _mm_set1_epi32(0);
        __m128i E1 = _mm_set1_epi32(0);
        __m128i E2 = _mm_set1_epi32(0);
        __m128i E3 = _mm_set1_epi32(0);

        for (int sz = 0; sz < src_depth_quad; ++sz) {
            const auto weight_sz = weight_dz + (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT) * sz;
            const auto src_z     = src_x + sz * GEMM_INT8_DST_XUNIT * GEMM_INT8_SRC_UNIT;
            auto w0 = *(__m128i*)(weight_sz + GEMM_INT8_SRC_UNIT * 0);
            auto w1 = *(__m128i*)(weight_sz + GEMM_INT8_SRC_UNIT * 1);
            auto w2 = *(__m128i*)(weight_sz + GEMM_INT8_SRC_UNIT * 2);
            auto w3 = *(__m128i*)(weight_sz + GEMM_INT8_SRC_UNIT * 3);
#define WINT8ToINT16(i)\
auto w##i##0 = _mm_srai_epi16(_mm_unpacklo_epi8(zero, w##i), 8);\
auto w##i##1 = _mm_srai_epi16(_mm_unpackhi_epi8(zero, w##i), 8);\

            WINT8ToINT16(0);
            WINT8ToINT16(1);
            WINT8ToINT16(2);
            WINT8ToINT16(3);

#define SINT8ToINT16(i)\
auto s##i##0 = _mm_srai_epi16(_mm_unpacklo_epi8(zero, s##i), 8);\
auto s##i##1 = _mm_srai_epi16(_mm_unpackhi_epi8(zero, s##i), 8);\

            auto s0 = *(__m128i*)(src_z + GEMM_INT8_SRC_UNIT * 0);
            auto s1 = *(__m128i*)(src_z + GEMM_INT8_SRC_UNIT * 1);
            auto s2 = *(__m128i*)(src_z + GEMM_INT8_SRC_UNIT * 2);
            auto s3 = *(__m128i*)(src_z + GEMM_INT8_SRC_UNIT * 3);

            SINT8ToINT16(0);
            SINT8ToINT16(1);
            SINT8ToINT16(2);
            SINT8ToINT16(3);

#define COMPUTE(u, v)\
d##u = _mm_add_epi32(d##u, _mm_madd_epi16(w##u##v, s0##v));\
e##u = _mm_add_epi32(e##u, _mm_madd_epi16(w##u##v, s1##v));\
D##u = _mm_add_epi32(D##u, _mm_madd_epi16(w##u##v, s2##v));\
E##u = _mm_add_epi32(E##u, _mm_madd_epi16(w##u##v, s3##v));\

            COMPUTE(0, 0);
            COMPUTE(0, 1);
            COMPUTE(1, 0);
            COMPUTE(1, 1);
            COMPUTE(2, 0);
            COMPUTE(2, 1);
            COMPUTE(3, 0);
            COMPUTE(3, 1);
        }
        d0 = _mm_hadd_epi32(d0, d1);
        d1 = _mm_hadd_epi32(d2, d3);
        d0 = _mm_hadd_epi32(d0, d1);

        e0 = _mm_hadd_epi32(e0, e1);
        e1 = _mm_hadd_epi32(e2, e3);
        d1 = _mm_hadd_epi32(e0, e1);
        
        D0 = _mm_hadd_epi32(D0, D1);
        D1 = _mm_hadd_epi32(D2, D3);
        d2 = _mm_hadd_epi32(D0, D1);

        E0 = _mm_hadd_epi32(E0, E1);
        E1 = _mm_hadd_epi32(E2, E3);
        d3 = _mm_hadd_epi32(E0, E1);

        auto biasValue = *(__m128i*)(bias_dz);
        auto scaleValue = _mm_loadu_ps(scale_dz);
        d0 = _mm_add_epi32(d0, biasValue);
        d1 = _mm_add_epi32(d1, biasValue);
        d2 = _mm_add_epi32(d2, biasValue);
        d3 = _mm_add_epi32(d3, biasValue);
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
        auto m0 = _mm_cmplt_ps(f0, _mm_castsi128_ps(zero));
        auto m1 = _mm_cmplt_ps(f1, _mm_castsi128_ps(zero));
        auto m2 = _mm_cmplt_ps(f2, _mm_castsi128_ps(zero));
        auto m3 = _mm_cmplt_ps(f3, _mm_castsi128_ps(zero));
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
