//
//  GemmInt8.cpp
//  MNN
//
//  Created by MNN on 2020/09/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_AVX512_VNNI
#include "FunctionSummary.hpp"
#include "core/Macro.h"
#define PACK_UNIT 16
namespace {
static inline __m128i mm_loadu_si128(const void* addr) {
    return _mm_loadu_si128((__m128i const*)addr);
}
}  // namespace

void _AVX512_MNNGemmInt8AddBiasScale_16x4_Unit(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, const QuanPostTreatParameters* post, size_t realDst) {
    const auto dst_step_tmp = dst_step / sizeof(int8_t);
    __m128 zero128 = _mm_set1_ps(0.0f);
    __m128 minValue = _mm_set1_ps(post->minValue);
    __m128 maxValue = _mm_set1_ps(post->maxValue);
    __m128 plus = _mm_set1_ps(0.5f);
    __m128 minus = _mm_set1_ps(-0.5f);
    if (realDst == 2) {
        for (int dz = 0; dz < dst_depth_quad; ++dz) {
            const auto weight_dz = weight + dz * src_depth_quad * (16 * 16);
            const auto bias_dz = post->bias + dz * 16;
            const float* scale_dz = nullptr;
            if (post->scale != nullptr) {
                scale_dz  = post->scale + dz * 16;
            }
            auto dst_z           = dst + dz * dst_step_tmp;
            const auto src_x   = src;
            auto dst_x         = dst_z;
            __m512i D0 = _mm512_set1_epi32(0);
            __m512i D1 = _mm512_set1_epi32(0);
            __m512i D2 = _mm512_set1_epi32(0);
            __m512i D3 = _mm512_set1_epi32(0);
            __m512i D4 = _mm512_set1_epi32(0);
            __m512i D5 = _mm512_set1_epi32(0);
            __m512i D6 = _mm512_set1_epi32(0);
            __m512i D7 = _mm512_set1_epi32(0);

            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + (16 * 16) * sz;
                const auto src_z     = src_x + sz * 2 * 16;
                auto w0 = _mm512_loadu_si512(weight_sz + 64 * 0);
                auto w1 = _mm512_loadu_si512(weight_sz + 64 * 1);
                auto w2 = _mm512_loadu_si512(weight_sz + 64 * 2);
                auto w3 = _mm512_loadu_si512(weight_sz + 64 * 3);

                auto s0 = _mm512_broadcast_i32x4(mm_loadu_si128(src_z + 16 * 0));
                auto s1 = _mm512_broadcast_i32x4(mm_loadu_si128(src_z + 16 * 1));
                
                D0 = _mm512_dpbusds_epi32(D0, s0, w0);
                D1 = _mm512_dpbusds_epi32(D1, s0, w1);
                D2 = _mm512_dpbusds_epi32(D2, s0, w2);
                D3 = _mm512_dpbusds_epi32(D3, s0, w3);

                D4 = _mm512_dpbusds_epi32(D4, s1, w0);
                D5 = _mm512_dpbusds_epi32(D5, s1, w1);
                D6 = _mm512_dpbusds_epi32(D6, s1, w2);
                D7 = _mm512_dpbusds_epi32(D7, s1, w3);
            }
            auto d00 = _mm512_extracti32x4_epi32(D0, 0);
            auto d01 = _mm512_extracti32x4_epi32(D0, 1);
            auto d02 = _mm512_extracti32x4_epi32(D0, 2);
            auto d03 = _mm512_extracti32x4_epi32(D0, 3);

            auto d10 = _mm512_extracti32x4_epi32(D1, 0);
            auto d11 = _mm512_extracti32x4_epi32(D1, 1);
            auto d12 = _mm512_extracti32x4_epi32(D1, 2);
            auto d13 = _mm512_extracti32x4_epi32(D1, 3);

            auto d20 = _mm512_extracti32x4_epi32(D2, 0);
            auto d21 = _mm512_extracti32x4_epi32(D2, 1);
            auto d22 = _mm512_extracti32x4_epi32(D2, 2);
            auto d23 = _mm512_extracti32x4_epi32(D2, 3);

            auto d30 = _mm512_extracti32x4_epi32(D3, 0);
            auto d31 = _mm512_extracti32x4_epi32(D3, 1);
            auto d32 = _mm512_extracti32x4_epi32(D3, 2);
            auto d33 = _mm512_extracti32x4_epi32(D3, 3);

            auto d40 = _mm512_extracti32x4_epi32(D4, 0);
            auto d41 = _mm512_extracti32x4_epi32(D4, 1);
            auto d42 = _mm512_extracti32x4_epi32(D4, 2);
            auto d43 = _mm512_extracti32x4_epi32(D4, 3);

            auto d50 = _mm512_extracti32x4_epi32(D5, 0);
            auto d51 = _mm512_extracti32x4_epi32(D5, 1);
            auto d52 = _mm512_extracti32x4_epi32(D5, 2);
            auto d53 = _mm512_extracti32x4_epi32(D5, 3);

            auto d60 = _mm512_extracti32x4_epi32(D6, 0);
            auto d61 = _mm512_extracti32x4_epi32(D6, 1);
            auto d62 = _mm512_extracti32x4_epi32(D6, 2);
            auto d63 = _mm512_extracti32x4_epi32(D6, 3);

            auto d70 = _mm512_extracti32x4_epi32(D7, 0);
            auto d71 = _mm512_extracti32x4_epi32(D7, 1);
            auto d72 = _mm512_extracti32x4_epi32(D7, 2);
            auto d73 = _mm512_extracti32x4_epi32(D7, 3);
            
            d00 = _mm_hadd_epi32(d00, d01);
            d02 = _mm_hadd_epi32(d02, d03);
            d10 = _mm_hadd_epi32(d10, d11);
            d12 = _mm_hadd_epi32(d12, d13);
            auto d0 = _mm_hadd_epi32(d00, d02);
            auto d1 = _mm_hadd_epi32(d10, d12);

            d20 = _mm_hadd_epi32(d20, d21);
            d22 = _mm_hadd_epi32(d22, d23);
            d30 = _mm_hadd_epi32(d30, d31);
            d32 = _mm_hadd_epi32(d32, d33);
            auto d2 = _mm_hadd_epi32(d20, d22);
            auto d3 = _mm_hadd_epi32(d30, d32);

            d40 = _mm_hadd_epi32(d40, d41);
            d42 = _mm_hadd_epi32(d42, d43);
            d50 = _mm_hadd_epi32(d50, d51);
            d52 = _mm_hadd_epi32(d52, d53);
            auto d4 = _mm_hadd_epi32(d40, d42);
            auto d5 = _mm_hadd_epi32(d50, d52);

            d60 = _mm_hadd_epi32(d60, d61);
            d62 = _mm_hadd_epi32(d62, d63);
            d70 = _mm_hadd_epi32(d70, d71);
            d72 = _mm_hadd_epi32(d72, d73);
            auto d6 = _mm_hadd_epi32(d60, d62);
            auto d7 = _mm_hadd_epi32(d70, d72);

            if (post->scale != nullptr) {
                auto biasValue0 = _mm_loadu_si128((__m128i*)(bias_dz));
                auto biasValue1 = _mm_loadu_si128((__m128i*)(bias_dz + 4));
                auto biasValue2 = _mm_loadu_si128((__m128i*)(bias_dz + 8));
                auto biasValue3 = _mm_loadu_si128((__m128i*)(bias_dz + 12));
                d0 = _mm_add_epi32(d0, biasValue0);
                d1 = _mm_add_epi32(d1, biasValue1);
                d2 = _mm_add_epi32(d2, biasValue2);
                d3 = _mm_add_epi32(d3, biasValue3);
                d4 = _mm_add_epi32(d4, biasValue0);
                d5 = _mm_add_epi32(d5, biasValue1);
                d6 = _mm_add_epi32(d6, biasValue2);
                d7 = _mm_add_epi32(d7, biasValue3);
                auto scaleValue0 = _mm_loadu_ps(scale_dz);
                auto scaleValue1 = _mm_loadu_ps(scale_dz + 4);
                auto scaleValue2 = _mm_loadu_ps(scale_dz + 8);
                auto scaleValue3 = _mm_loadu_ps(scale_dz + 12);
                __m128 f0 = _mm_cvtepi32_ps(d0);
                __m128 f1 = _mm_cvtepi32_ps(d1);
                __m128 f2 = _mm_cvtepi32_ps(d2);
                __m128 f3 = _mm_cvtepi32_ps(d3);
                __m128 f4 = _mm_cvtepi32_ps(d4);
                __m128 f5 = _mm_cvtepi32_ps(d5);
                __m128 f6 = _mm_cvtepi32_ps(d6);
                __m128 f7 = _mm_cvtepi32_ps(d7);
                f0 = _mm_mul_ps(f0, scaleValue0);
                f1 = _mm_mul_ps(f1, scaleValue1);
                f2 = _mm_mul_ps(f2, scaleValue2);
                f3 = _mm_mul_ps(f3, scaleValue3);
                f4 = _mm_mul_ps(f4, scaleValue0);
                f5 = _mm_mul_ps(f5, scaleValue1);
                f6 = _mm_mul_ps(f6, scaleValue2);
                f7 = _mm_mul_ps(f7, scaleValue3);
                f0 = _mm_min_ps(f0, maxValue);
                f1 = _mm_min_ps(f1, maxValue);
                f2 = _mm_min_ps(f2, maxValue);
                f3 = _mm_min_ps(f3, maxValue);
                f4 = _mm_min_ps(f4, maxValue);
                f5 = _mm_min_ps(f5, maxValue);
                f6 = _mm_min_ps(f6, maxValue);
                f7 = _mm_min_ps(f7, maxValue);
                f0 = _mm_max_ps(f0, minValue);
                f1 = _mm_max_ps(f1, minValue);
                f2 = _mm_max_ps(f2, minValue);
                f3 = _mm_max_ps(f3, minValue);
                f4 = _mm_max_ps(f4, minValue);
                f5 = _mm_max_ps(f5, minValue);
                f6 = _mm_max_ps(f6, minValue);
                f7 = _mm_max_ps(f7, minValue);
                auto m0 = _mm_cmplt_ps(f0, zero128);
                auto m1 = _mm_cmplt_ps(f1, zero128);
                auto m2 = _mm_cmplt_ps(f2, zero128);
                auto m3 = _mm_cmplt_ps(f3, zero128);
                auto m4 = _mm_cmplt_ps(f4, zero128);
                auto m5 = _mm_cmplt_ps(f5, zero128);
                auto m6 = _mm_cmplt_ps(f6, zero128);
                auto m7 = _mm_cmplt_ps(f7, zero128);
                m0 = _mm_blendv_ps(plus, minus, m0);
                m1 = _mm_blendv_ps(plus, minus, m1);
                m2 = _mm_blendv_ps(plus, minus, m2);
                m3 = _mm_blendv_ps(plus, minus, m3);
                m4 = _mm_blendv_ps(plus, minus, m4);
                m5 = _mm_blendv_ps(plus, minus, m5);
                m6 = _mm_blendv_ps(plus, minus, m6);
                m7 = _mm_blendv_ps(plus, minus, m7);
                f0 = _mm_add_ps(f0, m0);
                f1 = _mm_add_ps(f1, m1);
                f2 = _mm_add_ps(f2, m2);
                f3 = _mm_add_ps(f3, m3);
                f4 = _mm_add_ps(f4, m4);
                f5 = _mm_add_ps(f5, m5);
                f6 = _mm_add_ps(f6, m6);
                f7 = _mm_add_ps(f7, m7);
                // 3: _MM_FROUND_TO_ZERO
                d0 = _mm_cvtps_epi32(_mm_round_ps(f0, 3));
                d1 = _mm_cvtps_epi32(_mm_round_ps(f1, 3));
                d2 = _mm_cvtps_epi32(_mm_round_ps(f2, 3));
                d3 = _mm_cvtps_epi32(_mm_round_ps(f3, 3));
                d4 = _mm_cvtps_epi32(_mm_round_ps(f4, 3));
                d5 = _mm_cvtps_epi32(_mm_round_ps(f5, 3));
                d6 = _mm_cvtps_epi32(_mm_round_ps(f6, 3));
                d7 = _mm_cvtps_epi32(_mm_round_ps(f7, 3));

                // Int32 -> Int8
                auto offset = _mm_set1_epi16(128);
                d0 = _mm_packs_epi32(d0, d1);
                d2 = _mm_packs_epi32(d2, d3);
                d0 = _mm_add_epi16(d0, offset);
                d2 = _mm_add_epi16(d2, offset);
                d0 = _mm_packus_epi16(d0, d2);

                d4 = _mm_packs_epi32(d4, d5);
                d6 = _mm_packs_epi32(d6, d7);
                d4 = _mm_add_epi16(d4, offset);
                d6 = _mm_add_epi16(d6, offset);
                d4 = _mm_packus_epi16(d4, d6);

                _mm_storeu_si128((__m128i*)dst_x, d0);
                _mm_storeu_si128((__m128i*)dst_x + 1, d4);
            } else {
                auto biasValue0 = _mm_loadu_si128((__m128i*)(bias_dz));
                auto biasValue1 = _mm_loadu_si128((__m128i*)(bias_dz + 4));
                auto biasValue2 = _mm_loadu_si128((__m128i*)(bias_dz + 8));
                auto biasValue3 = _mm_loadu_si128((__m128i*)(bias_dz + 12));
                d0 = _mm_add_epi32(d0, biasValue0);
                d1 = _mm_add_epi32(d1, biasValue1);
                d2 = _mm_add_epi32(d2, biasValue2);
                d3 = _mm_add_epi32(d3, biasValue3);
                d4 = _mm_add_epi32(d4, biasValue0);
                d5 = _mm_add_epi32(d5, biasValue1);
                d6 = _mm_add_epi32(d6, biasValue2);
                d7 = _mm_add_epi32(d7, biasValue3);
                __m128 f0 = _mm_cvtepi32_ps(d0);
                __m128 f1 = _mm_cvtepi32_ps(d1);
                __m128 f2 = _mm_cvtepi32_ps(d2);
                __m128 f3 = _mm_cvtepi32_ps(d3);
                __m128 f4 = _mm_cvtepi32_ps(d4);
                __m128 f5 = _mm_cvtepi32_ps(d5);
                __m128 f6 = _mm_cvtepi32_ps(d6);
                __m128 f7 = _mm_cvtepi32_ps(d7);
                _mm_storeu_ps(((float*)dst_x), f0);
                _mm_storeu_ps(((float*)dst_x) + 4, f1);
                _mm_storeu_ps(((float*)dst_x) + 8, f2);
                _mm_storeu_ps(((float*)dst_x) + 12, f3);
                _mm_storeu_ps(((float*)dst_x) + 16, f4);
                _mm_storeu_ps(((float*)dst_x) + 20, f5);
                _mm_storeu_ps(((float*)dst_x) + 24, f6);
                _mm_storeu_ps(((float*)dst_x) + 28, f7);
            }
        }
        return;
    }
    for (int dz = 0; dz < dst_depth_quad; ++dz) {
        const auto weight_dz = weight + dz * src_depth_quad * (16 * 16);
        const auto bias_dz = post->bias + dz * 16;
        const float* scale_dz = nullptr;
        if (post->scale != nullptr) {
            scale_dz  = post->scale + dz * 16;
        }
        auto dst_z           = dst + dz * dst_step_tmp;
        const auto src_x   = src;
        auto dst_x         = dst_z;
        __m512i D0 = _mm512_set1_epi32(0);
        __m512i D1 = _mm512_set1_epi32(0);
        __m512i D2 = _mm512_set1_epi32(0);
        __m512i D3 = _mm512_set1_epi32(0);

        for (int sz = 0; sz < src_depth_quad; ++sz) {
            const auto weight_sz = weight_dz + (16 * 16) * sz;
            const auto src_z     = src_x + sz * 2 * 16;
            auto w0 = _mm512_loadu_si512(weight_sz + 64 * 0);
            auto w1 = _mm512_loadu_si512(weight_sz + 64 * 1);
            auto w2 = _mm512_loadu_si512(weight_sz + 64 * 2);
            auto w3 = _mm512_loadu_si512(weight_sz + 64 * 3);

            auto s0 = _mm512_broadcast_i32x4(mm_loadu_si128(src_z + 16 * 0));
            
            D0 = _mm512_dpbusds_epi32(D0, s0, w0);
            D1 = _mm512_dpbusds_epi32(D1, s0, w1);
            D2 = _mm512_dpbusds_epi32(D2, s0, w2);
            D3 = _mm512_dpbusds_epi32(D3, s0, w3);

        }
        auto d00 = _mm512_extracti32x4_epi32(D0, 0);
        auto d01 = _mm512_extracti32x4_epi32(D0, 1);
        auto d02 = _mm512_extracti32x4_epi32(D0, 2);
        auto d03 = _mm512_extracti32x4_epi32(D0, 3);

        auto d10 = _mm512_extracti32x4_epi32(D1, 0);
        auto d11 = _mm512_extracti32x4_epi32(D1, 1);
        auto d12 = _mm512_extracti32x4_epi32(D1, 2);
        auto d13 = _mm512_extracti32x4_epi32(D1, 3);

        auto d20 = _mm512_extracti32x4_epi32(D2, 0);
        auto d21 = _mm512_extracti32x4_epi32(D2, 1);
        auto d22 = _mm512_extracti32x4_epi32(D2, 2);
        auto d23 = _mm512_extracti32x4_epi32(D2, 3);

        auto d30 = _mm512_extracti32x4_epi32(D3, 0);
        auto d31 = _mm512_extracti32x4_epi32(D3, 1);
        auto d32 = _mm512_extracti32x4_epi32(D3, 2);
        auto d33 = _mm512_extracti32x4_epi32(D3, 3);

        d00 = _mm_hadd_epi32(d00, d01);
        d02 = _mm_hadd_epi32(d02, d03);
        d10 = _mm_hadd_epi32(d10, d11);
        d12 = _mm_hadd_epi32(d12, d13);
        auto d0 = _mm_hadd_epi32(d00, d02);
        auto d1 = _mm_hadd_epi32(d10, d12);

        d20 = _mm_hadd_epi32(d20, d21);
        d22 = _mm_hadd_epi32(d22, d23);
        d30 = _mm_hadd_epi32(d30, d31);
        d32 = _mm_hadd_epi32(d32, d33);
        auto d2 = _mm_hadd_epi32(d20, d22);
        auto d3 = _mm_hadd_epi32(d30, d32);

        if (post->scale != nullptr) {
            auto biasValue0 = _mm_loadu_si128((__m128i*)(bias_dz));
            auto biasValue1 = _mm_loadu_si128((__m128i*)(bias_dz + 4));
            auto biasValue2 = _mm_loadu_si128((__m128i*)(bias_dz + 8));
            auto biasValue3 = _mm_loadu_si128((__m128i*)(bias_dz + 12));
            d0 = _mm_add_epi32(d0, biasValue0);
            d1 = _mm_add_epi32(d1, biasValue1);
            d2 = _mm_add_epi32(d2, biasValue2);
            d3 = _mm_add_epi32(d3, biasValue3);
            auto scaleValue0 = _mm_loadu_ps(scale_dz);
            auto scaleValue1 = _mm_loadu_ps(scale_dz + 4);
            auto scaleValue2 = _mm_loadu_ps(scale_dz + 8);
            auto scaleValue3 = _mm_loadu_ps(scale_dz + 12);
            __m128 f0 = _mm_cvtepi32_ps(d0);
            __m128 f1 = _mm_cvtepi32_ps(d1);
            __m128 f2 = _mm_cvtepi32_ps(d2);
            __m128 f3 = _mm_cvtepi32_ps(d3);
            f0 = _mm_mul_ps(f0, scaleValue0);
            f1 = _mm_mul_ps(f1, scaleValue1);
            f2 = _mm_mul_ps(f2, scaleValue2);
            f3 = _mm_mul_ps(f3, scaleValue3);
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
            auto offset = _mm_set1_epi16(128);
            d0 = _mm_packs_epi32(d0, d1);
            d2 = _mm_packs_epi32(d2, d3);
            d0 = _mm_add_epi16(d0, offset);
            d2 = _mm_add_epi16(d2, offset);
            d0 = _mm_packus_epi16(d0, d2);

            _mm_storeu_si128((__m128i*)dst_x, d0);
        } else {
            auto biasValue0 = _mm_loadu_si128((__m128i*)(bias_dz));
            auto biasValue1 = _mm_loadu_si128((__m128i*)(bias_dz + 4));
            auto biasValue2 = _mm_loadu_si128((__m128i*)(bias_dz + 8));
            auto biasValue3 = _mm_loadu_si128((__m128i*)(bias_dz + 12));
            d0 = _mm_add_epi32(d0, biasValue0);
            d1 = _mm_add_epi32(d1, biasValue1);
            d2 = _mm_add_epi32(d2, biasValue2);
            d3 = _mm_add_epi32(d3, biasValue3);
            __m128 f0 = _mm_cvtepi32_ps(d0);
            __m128 f1 = _mm_cvtepi32_ps(d1);
            __m128 f2 = _mm_cvtepi32_ps(d2);
            __m128 f3 = _mm_cvtepi32_ps(d3);
            _mm_storeu_ps(((float*)dst_x), f0);
            _mm_storeu_ps(((float*)dst_x) + 4, f1);
            _mm_storeu_ps(((float*)dst_x) + 8, f2);
            _mm_storeu_ps(((float*)dst_x) + 12, f3);
        }
    }
}

void _AVX512_MNNLineDepthWiseInt8AddBiasScaleUnit(int8_t* dstO, const int8_t* srcO, const int8_t* weightO, const QuanPostTreatParameters* parameters, size_t width, size_t src_w_step, size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step) {
    auto dst = dstO;
    auto src = (const int16_t*)srcO;
    int widthC4 = width / 2;
    int widthRemain = width % 2;
    auto weight = (const int16_t*)weightO;
    auto biasValue0 = _mm256_castps_si256(_mm256_loadu_ps((const float*)parameters->bias));
    auto scaleValue0 = _mm256_loadu_ps((const float*)parameters->scale);
    auto biasValue1 = _mm256_castps_si256(_mm256_loadu_ps((const float*)parameters->bias + 8));
    auto scaleValue1 = _mm256_loadu_ps((const float*)parameters->scale + 8);
    __m256i d0, d1, d2, d3;
    int dx, fx, fy;
    __m256i zero = _mm256_setzero_si256();
    __m256 zero128 = _mm256_set1_ps(0.0f);
    __m128i minValue = _mm_set1_epi16(parameters->minValue + 128);
    __m128i maxValue = _mm_set1_epi16(parameters->maxValue + 128);
    __m256 plus = _mm256_set1_ps(0.5f);
    __m256 minus = _mm256_set1_ps(-0.5f);
    for (dx = 0; dx < widthC4; ++dx) {
        d0 = biasValue0;
        d1 = biasValue1;
        d2 = biasValue0;
        d3 = biasValue1;

        auto dst_x          = dst;
        const auto src_z    = src;
        for (fy = 0; fy < fh; ++fy) {
            const auto src_y    = src_z + fy * dilateY_step;
            const auto weight_y = weight + fy * fw * PACK_UNIT;
            for (fx = 0; fx < fw; ++fx) {
                const auto src_x    = src_y + fx * dilateX_step;
                auto S0 = _mm256_castsi128_si256(_mm_castps_si128(_mm_loadu_ps((const float*)(src_x + 0 * src_w_step))));
                auto S1 = _mm256_castsi128_si256(_mm_castps_si128(_mm_loadu_ps((const float*)(src_x + 0 * src_w_step + 8))));
                auto S2 = _mm256_castsi128_si256(_mm_castps_si128(_mm_loadu_ps((const float*)(src_x + 1 * src_w_step))));
                auto S3 = _mm256_castsi128_si256(_mm_castps_si128(_mm_loadu_ps((const float*)(src_x + 1 * src_w_step + 8))));
                const auto weight_x = weight_y + PACK_UNIT * fx;
                auto W0 = _mm256_castsi128_si256(_mm_castps_si128(_mm_loadu_ps((const float*)weight_x)));
                auto W1 = _mm256_castsi128_si256(_mm_castps_si128(_mm_loadu_ps((const float*)(weight_x + 8))));
                auto s00 = _mm256_unpacklo_epi16(S0, zero);
                auto s10 = _mm256_unpacklo_epi16(S1, zero);
                auto s20 = _mm256_unpacklo_epi16(S2, zero);
                auto s30 = _mm256_unpacklo_epi16(S3, zero);
                auto s01 = _mm256_unpackhi_epi16(S0, zero);
                auto s11 = _mm256_unpackhi_epi16(S1, zero);
                auto s21 = _mm256_unpackhi_epi16(S2, zero);
                auto s31 = _mm256_unpackhi_epi16(S3, zero);
                auto w00 = _mm256_unpacklo_epi16(W0, zero);
                auto w01 = _mm256_unpackhi_epi16(W0, zero);
                auto w10 = _mm256_unpacklo_epi16(W1, zero);
                auto w11 = _mm256_unpackhi_epi16(W1, zero);

                S0 = _mm256_permute2f128_si256(s00, s01, 32);
                S1 = _mm256_permute2f128_si256(s10, s11, 32);
                S2 = _mm256_permute2f128_si256(s20, s21, 32);
                S3 = _mm256_permute2f128_si256(s30, s31, 32);
                W0 = _mm256_permute2f128_si256(w00, w01, 32);
                W1 = _mm256_permute2f128_si256(w10, w11, 32);
                d0 = _mm256_add_epi32(d0, _mm256_madd_epi16(W0, S0));
                d1 = _mm256_add_epi32(d1, _mm256_madd_epi16(W1, S1));
                d2 = _mm256_add_epi32(d2, _mm256_madd_epi16(W0, S2));
                d3 = _mm256_add_epi32(d3, _mm256_madd_epi16(W1, S3));
            }
        }
        __m256 f0 = _mm256_cvtepi32_ps(d0);
        __m256 f1 = _mm256_cvtepi32_ps(d1);
        __m256 f2 = _mm256_cvtepi32_ps(d2);
        __m256 f3 = _mm256_cvtepi32_ps(d3);
        f0 = _mm256_mul_ps(f0, scaleValue0);
        f1 = _mm256_mul_ps(f1, scaleValue1);
        f2 = _mm256_mul_ps(f2, scaleValue0);
        f3 = _mm256_mul_ps(f3, scaleValue1);
        auto m0 = _mm256_cmp_ps(f0, zero128, 1);
        auto m1 = _mm256_cmp_ps(f1, zero128, 1);
        auto m2 = _mm256_cmp_ps(f2, zero128, 1);
        auto m3 = _mm256_cmp_ps(f3, zero128, 1);
        m0 = _mm256_blendv_ps(plus, minus, m0);
        m1 = _mm256_blendv_ps(plus, minus, m1);
        m2 = _mm256_blendv_ps(plus, minus, m2);
        m3 = _mm256_blendv_ps(plus, minus, m3);
        f0 = _mm256_add_ps(f0, m0);
        f1 = _mm256_add_ps(f1, m1);
        f2 = _mm256_add_ps(f2, m2);
        f3 = _mm256_add_ps(f3, m3);
        // 3: _MM_FROUND_TO_ZERO
        d0 = _mm256_cvtps_epi32(_mm256_round_ps(f0, 3));
        d1 = _mm256_cvtps_epi32(_mm256_round_ps(f1, 3));
        d2 = _mm256_cvtps_epi32(_mm256_round_ps(f2, 3));
        d3 = _mm256_cvtps_epi32(_mm256_round_ps(f3, 3));
        auto offset = _mm256_set1_epi32(128);
        d0 = _mm256_add_epi32(d0, offset);
        d1 = _mm256_add_epi32(d1, offset);
        d2 = _mm256_add_epi32(d2, offset);
        d3 = _mm256_add_epi32(d3, offset);

        auto e0 = _mm256_permute2f128_si256(d0, d1, 32);
        auto e1 = _mm256_permute2f128_si256(d0, d1, 49);
        auto e2 = _mm256_permute2f128_si256(d2, d3, 32);
        auto e3 = _mm256_permute2f128_si256(d2, d3, 49);
        // Int32 -> Int8
        d0 = _mm256_packs_epi32(e0, e1);
        d2 = _mm256_packs_epi32(e2, e3);

        auto D0 = _mm256_extracti128_si256(d0, 0);
        auto D1 = _mm256_extracti128_si256(d0, 1);
        auto D2 = _mm256_extracti128_si256(d2, 0);
        auto D3 = _mm256_extracti128_si256(d2, 1);

        D0 = _mm_min_epi16(D0, maxValue);
        D1 = _mm_min_epi16(D1, maxValue);
        D0 = _mm_max_epi16(D0, minValue);
        D1 = _mm_max_epi16(D1, minValue);

        D2 = _mm_min_epi16(D2, maxValue);
        D3 = _mm_min_epi16(D3, maxValue);
        D2 = _mm_max_epi16(D2, minValue);
        D3 = _mm_max_epi16(D3, minValue);
        _mm_storeu_ps((float*)(dst), _mm_castsi128_ps(_mm_packus_epi16(D0, D1)));
        _mm_storeu_ps((float*)(dst + 16), _mm_castsi128_ps(_mm_packus_epi16(D2, D3)));
        dst += 32;
        src += src_w_step * 2;
    }
    if (widthRemain > 0) {
        d0 = biasValue0;
        d1 = biasValue1;

        auto dst_x          = dst;
        const auto src_z    = src;
        for (fy = 0; fy < fh; ++fy) {
            const auto src_y    = src_z + fy * dilateY_step;
            const auto weight_y = weight + fy * fw * PACK_UNIT;
            for (fx = 0; fx < fw; ++fx) {
                const auto src_x    = src_y + fx * dilateX_step;
                auto S0 = _mm256_castsi128_si256(_mm_castps_si128(_mm_loadu_ps((const float*)(src_x + 0 * src_w_step))));
                auto S1 = _mm256_castsi128_si256(_mm_castps_si128(_mm_loadu_ps((const float*)(src_x + 0 * src_w_step + 8))));
                const auto weight_x = weight_y + PACK_UNIT * fx;
                auto W0 = _mm256_castsi128_si256(_mm_castps_si128(_mm_loadu_ps((const float*)weight_x)));
                auto W1 = _mm256_castsi128_si256(_mm_castps_si128(_mm_loadu_ps((const float*)(weight_x + 8))));
                auto s00 = _mm256_unpacklo_epi16(S0, zero);
                auto s10 = _mm256_unpacklo_epi16(S1, zero);
                auto s01 = _mm256_unpackhi_epi16(S0, zero);
                auto s11 = _mm256_unpackhi_epi16(S1, zero);
                auto w00 = _mm256_unpacklo_epi16(W0, zero);
                auto w01 = _mm256_unpackhi_epi16(W0, zero);
                auto w10 = _mm256_unpacklo_epi16(W1, zero);
                auto w11 = _mm256_unpackhi_epi16(W1, zero);
                S0 = _mm256_permute2f128_si256(s00, s01, 32);
                S1 = _mm256_permute2f128_si256(s10, s11, 32);
                W0 = _mm256_permute2f128_si256(w00, w01, 32);
                W1 = _mm256_permute2f128_si256(w10, w11, 32);
                d0 = _mm256_add_epi32(d0, _mm256_madd_epi16(W0, S0));
                d1 = _mm256_add_epi32(d1, _mm256_madd_epi16(W1, S1));
            }
        }
        __m256 f0 = _mm256_cvtepi32_ps(d0);
        __m256 f1 = _mm256_cvtepi32_ps(d1);
        f0 = _mm256_mul_ps(f0, scaleValue0);
        f1 = _mm256_mul_ps(f1, scaleValue1);
        auto m0 = _mm256_cmp_ps(f0, zero128, 1);
        auto m1 = _mm256_cmp_ps(f1, zero128, 1);
        m0 = _mm256_blendv_ps(plus, minus, m0);
        m1 = _mm256_blendv_ps(plus, minus, m1);
        f0 = _mm256_add_ps(f0, m0);
        f1 = _mm256_add_ps(f1, m1);
        // 3: _MM_FROUND_TO_ZERO
        d0 = _mm256_cvtps_epi32(_mm256_round_ps(f0, 3));
        d1 = _mm256_cvtps_epi32(_mm256_round_ps(f1, 3));

        auto offset = _mm256_set1_epi32(128);
        d0 = _mm256_add_epi32(d0, offset);
        d1 = _mm256_add_epi32(d1, offset);

        auto e0 = _mm256_permute2f128_si256(d0, d1, 32);
        auto e1 = _mm256_permute2f128_si256(d0, d1, 49);
        // Int32 -> Int8
        d0 = _mm256_packs_epi32(e0, e1);
        auto D0 = _mm256_extracti128_si256(d0, 0);
        auto D1 = _mm256_extracti128_si256(d0, 1);

        D0 = _mm_min_epi16(D0, maxValue);
        D1 = _mm_min_epi16(D1, maxValue);
        D0 = _mm_max_epi16(D0, minValue);
        D1 = _mm_max_epi16(D1, minValue);

        _mm_storeu_ps((float*)(dst), _mm_castsi128_ps(_mm_packus_epi16(D0, D1)));
    }
}

void _AVX512_MNNFloat2Int8(const float* src, int8_t* dst, size_t sizeQuad, const float* scalep, ssize_t minV, ssize_t maxV, ssize_t zeroPoint) {
    auto zero = _mm256_set1_epi32(0);
    auto minValue = _mm256_set1_ps(minV);
    auto maxValue = _mm256_set1_ps(maxV);
    auto zeroPointValue = _mm256_set1_ps(zeroPoint);
    auto offset = _mm256_set1_epi32(128);
    auto plus = _mm256_set1_ps(0.5f);
    auto minus = _mm256_set1_ps(-0.5f);
    auto scaleValue0 = _mm256_loadu_ps(scalep);
    auto scaleValue1 = _mm256_loadu_ps(scalep + 8);

    for (int i = 0; i < sizeQuad; ++i) {
        auto f0 = _mm256_loadu_ps(src + PACK_UNIT * i);
        auto f1 = _mm256_loadu_ps(src + PACK_UNIT * i + 8);
        f0 = _mm256_mul_ps(f0, scaleValue0);
        f1 = _mm256_mul_ps(f1, scaleValue1);
        f0 = _mm256_add_ps(f0, zeroPointValue);
        f1 = _mm256_add_ps(f1, zeroPointValue);
        f0 = _mm256_min_ps(f0, maxValue);
        f1 = _mm256_min_ps(f1, maxValue);
        f0 = _mm256_max_ps(f0, minValue);
        f1 = _mm256_max_ps(f1, minValue);
        auto m0 = _mm256_cmp_ps(f0, _mm256_castsi256_ps(zero), 1);
        auto m1 = _mm256_cmp_ps(f1, _mm256_castsi256_ps(zero), 1);
        m0 = _mm256_blendv_ps(plus, minus, m0);
        m1 = _mm256_blendv_ps(plus, minus, m1);
        f0 = _mm256_add_ps(f0, m0);
        f1 = _mm256_add_ps(f1, m1);
        // 3: _MM_FROUND_TO_ZERO
        auto d0 = _mm256_cvtps_epi32(_mm256_round_ps(f0, 3));
        auto d1 = _mm256_cvtps_epi32(_mm256_round_ps(f1, 3));
        d0 = _mm256_add_epi32(d0, offset);
        d1 = _mm256_add_epi32(d1, offset);
        d0 = _mm256_packs_epi32(d0, _mm256_setzero_si256());
        d1 = _mm256_packs_epi32(d1, _mm256_setzero_si256());
        d0 = _mm256_permute4x64_epi64(d0, 0xD8);
        d1 = _mm256_permute4x64_epi64(d1, 0xD8);
#if defined(_MSC_VER)
        __m256i x = static_cast<__m256i>(_mm256_packus_epi16(d0, _mm256_setzero_si256()));
        __m256i y = static_cast<__m256i>(_mm256_packus_epi16(d1, _mm256_setzero_si256()));
        *((int64_t*)dst + 2 * i + 0) = x.m256i_i64[0];
        *((int64_t*)dst + 2 * i + 1) = y.m256i_i64[0];
#else
        __v4di x = static_cast<__v4di>(_mm256_packus_epi16(d0, _mm256_setzero_si256()));
        __v4di y = static_cast<__v4di>(_mm256_packus_epi16(d1, _mm256_setzero_si256()));
        *((int64_t*)dst + 2 * i + 0) = x[0];
        *((int64_t*)dst + 2 * i + 1) = y[0];
#endif
    }
}

void _AVX512_MNNInt8ScaleToFloat(float* dst, const int8_t* src, const float* scale, size_t sizeQuad, ssize_t zeroPoint) {
    auto sizeC4 = sizeQuad / 2;
    auto sizeRemain = sizeQuad % 2;
    auto zero = _mm256_set1_epi32(0);
    auto scaleValue0 = _mm256_loadu_ps(scale);
    auto scaleValue1 = _mm256_loadu_ps(scale + 8);
    auto zeroPointValue = _mm256_set1_epi32(zeroPoint + 128);
    for (int i = 0; i < sizeC4; ++i) {
        auto s = _mm256_castps_si256(_mm256_loadu_ps((const float*)(src)));
        auto s0_16 = _mm256_permute4x64_epi64(_mm256_unpacklo_epi8(s, zero), 0XD8);
        auto s1_16 = _mm256_permute4x64_epi64(_mm256_unpackhi_epi8(s, zero), 0xD8);
        auto s0_32 = _mm256_unpacklo_epi16(s0_16, zero);
        auto s1_32 = _mm256_unpacklo_epi16(s1_16, zero);
        auto s2_32 = _mm256_unpackhi_epi16(s0_16, zero);
        auto s3_32 = _mm256_unpackhi_epi16(s1_16, zero);
        s0_32 = _mm256_sub_epi32(s0_32, zeroPointValue);
        s1_32 = _mm256_sub_epi32(s1_32, zeroPointValue);
        s2_32 = _mm256_sub_epi32(s2_32, zeroPointValue);
        s3_32 = _mm256_sub_epi32(s3_32, zeroPointValue);
        auto s0_f = _mm256_cvtepi32_ps(s0_32);
        auto s1_f = _mm256_cvtepi32_ps(s1_32);
        auto s2_f = _mm256_cvtepi32_ps(s2_32);
        auto s3_f = _mm256_cvtepi32_ps(s3_32);
        _mm256_storeu_ps(dst + 8 * 0, _mm256_mul_ps(s0_f, scaleValue0));
        _mm256_storeu_ps(dst + 8 * 1, _mm256_mul_ps(s1_f, scaleValue1));
        _mm256_storeu_ps(dst + 8 * 2, _mm256_mul_ps(s2_f, scaleValue0));
        _mm256_storeu_ps(dst + 8 * 3, _mm256_mul_ps(s3_f, scaleValue1));
        src += 32;
        dst += 32;
    }
    if (sizeRemain > 0) {
        auto s = _mm256_castsi128_si256(_mm_castps_si128(_mm_loadu_ps((const float*)src)));
        auto s0_16 = _mm256_permute4x64_epi64(_mm256_unpacklo_epi8(s, zero), 0XD8);
        auto s1_16 = _mm256_permute4x64_epi64(_mm256_unpackhi_epi8(s, zero), 0xD8);
        auto s0_32 = _mm256_unpacklo_epi16(s0_16, zero);
        auto s1_32 = _mm256_unpacklo_epi16(s1_16, zero);
        auto s2_32 = _mm256_unpackhi_epi16(s0_16, zero);
        auto s3_32 = _mm256_unpackhi_epi16(s1_16, zero);
        s0_32 = _mm256_sub_epi32(s0_32, zeroPointValue);
        s1_32 = _mm256_sub_epi32(s1_32, zeroPointValue);
        auto s0_f = _mm256_cvtepi32_ps(s0_32);
        auto s1_f = _mm256_cvtepi32_ps(s1_32);
        _mm256_storeu_ps(dst + 8 * 0, _mm256_mul_ps(s0_f, scaleValue0));
        _mm256_storeu_ps(dst + 8 * 1, _mm256_mul_ps(s1_f, scaleValue1));
    }
}

// Assume GEMM_INT8_UNIT == 4 && GEMM_INT8_SRC_UNIT == 16
static void _fastIm2Col(int8_t* colAddr, const int8_t* inputOrigin, int32_t inputZeroPoint,
                        const MNN::ConvolutionCommon::Im2ColParameter* im2colParameter, size_t xIndexStart,
                        size_t realDstCount) {
    const int col_buffer_size = im2colParameter->kernelCountUnit * 16 * 2 * sizeof(int8_t);
    ::memset(colAddr, inputZeroPoint, col_buffer_size); // the padding process, since per-channel is removed, this is all right

    const int icDiv8   = im2colParameter->icDiv4;
    const int srcZStep = im2colParameter->srcZStep;
    inputOrigin += xIndexStart * PACK_UNIT;
    for (int i = 0; i < realDstCount; ++i) {
        auto colAddrI = colAddr + PACK_UNIT * i;
        auto inputK   = inputOrigin + PACK_UNIT * i;
        for (int sz = 0; sz < icDiv8; ++sz) {
            auto inputZ0           = inputK + srcZStep * sz;
            _mm_storeu_ps((float*)(colAddrI + 2 * PACK_UNIT * sz), _mm_loadu_ps((const float*)inputZ0));
        }
    }
}

static void _im2colCommonZ1(int8_t* colAddr, const int8_t* inputOrigin, int32_t inputZeroPoint,
                            const MNN::ConvolutionCommon::Im2ColParameter* im2colParameter, size_t xIndexStart,
                            size_t realDstCount) {
    int col_buffer_size = im2colParameter->kernelCountUnit * 2 * 16 * sizeof(int8_t);
    ::memset(colAddr, inputZeroPoint, col_buffer_size); // the padding process, since per-channel is removed, this is all right

    auto ih                     = im2colParameter->ih;
    auto iw                     = im2colParameter->iw;
    auto kh                     = im2colParameter->kernelY;
    auto kw                     = im2colParameter->kernelX;
    auto dilateX                = im2colParameter->dilateX;
    auto dilateY                = im2colParameter->dilateY;
    auto srcYStep               = im2colParameter->srcYStep;
    for (int i = 0; i < realDstCount; ++i) {
        int xIndex = (int)xIndexStart + i;
        int ox     = xIndex % im2colParameter->ow;
        int oy     = xIndex / im2colParameter->ow;

        int sx = ox * im2colParameter->strideX - im2colParameter->padX;
        int sy = oy * im2colParameter->strideY - im2colParameter->padY;

        int sfy = ALIMAX(0, (UP_DIV(-sy, im2colParameter->dilateY)));
        int efy = ALIMIN(kh, UP_DIV(ih - sy, im2colParameter->dilateY));
        int sfx = ALIMAX(0, (UP_DIV(-sx, im2colParameter->dilateX)));
        int efx = ALIMIN(kw, UP_DIV(iw - sx, im2colParameter->dilateX));
        int fyC = efy - sfy;
        int fxC = efx - sfx;

        auto colAddrI    = colAddr + 16 * i;
        
        auto inputOffset = inputOrigin + (sy + sfy * dilateY) * srcYStep + (sx + sfx * dilateX) * PACK_UNIT;
        auto indexOffset = sfy * kw + sfx;
        for (int fy = 0; fy < fyC; ++fy) {
            for (int fx = 0; fx < fxC; ++fx) {
                auto inputK       = inputOffset + fy * dilateY * srcYStep + fx * dilateX * PACK_UNIT;
                auto indexStart   = indexOffset + fy * kw + fx;
                _mm_storeu_ps((float*)(colAddrI + indexStart * 2 * 16), _mm_loadu_ps((const float*)(inputK)));
            }
        }
    }
}

static void _im2colCommon(int8_t* colAddr, const int8_t* inputOrigin, int32_t inputZeroPoint,
                          const MNN::ConvolutionCommon::Im2ColParameter* im2colParameter, size_t xIndexStart,
                          size_t realDstCount) {
    const int col_buffer_size = im2colParameter->kernelCountUnit * 2 * 16 * sizeof(int8_t);
    ::memset(colAddr, inputZeroPoint, col_buffer_size); // the padding process, since per-channel is removed, this is all right

    auto ih                     = im2colParameter->ih;
    auto iw                     = im2colParameter->iw;
    auto kh                     = im2colParameter->kernelY;
    auto kw                     = im2colParameter->kernelX;
    auto dilateX                = im2colParameter->dilateX;
    auto dilateY                = im2colParameter->dilateY;
    auto icDiv4                 = im2colParameter->icDiv4;
    auto srcZStep               = im2colParameter->srcZStep;
    auto srcYStep               = im2colParameter->srcYStep;
    for (int i = 0; i < realDstCount; ++i) {
        int xIndex = (int)xIndexStart + i;
        int ox     = xIndex % im2colParameter->ow;
        int oy     = xIndex / im2colParameter->ow;

        int sx = ox * im2colParameter->strideX - im2colParameter->padX;
        int sy = oy * im2colParameter->strideY - im2colParameter->padY;

        int sfy = ALIMAX(0, (UP_DIV(-sy, im2colParameter->dilateY)));
        int efy = ALIMIN(kh, UP_DIV(ih - sy, im2colParameter->dilateY));
        int sfx = ALIMAX(0, (UP_DIV(-sx, im2colParameter->dilateX)));
        int efx = ALIMIN(kw, UP_DIV(iw - sx, im2colParameter->dilateX));
        int fyC = efy - sfy;
        int fxC = efx - sfx;

        auto colAddrI    = colAddr + 16 * i;
        
        auto inputOffset = inputOrigin + (sy + sfy * dilateY) * srcYStep + (sx + sfx * dilateX) * PACK_UNIT;
        auto indexOffset = (sfy * kw + sfx) * icDiv4;
        for (int fy = 0; fy < fyC; ++fy) {
            for (int fx = 0; fx < fxC; ++fx) {
                auto inputK     = inputOffset + fy * dilateY * srcYStep + fx * dilateX * PACK_UNIT;
                auto indexStart = indexOffset + (fy * kw + fx) * icDiv4;
                for (int sz = 0; sz < icDiv4; ++sz) {
                    const int yIndex      = indexStart + sz;
                    _mm_storeu_ps((float*)(colAddrI + yIndex * 2 * 16), _mm_loadu_ps((const float*)(inputK)));
                    inputK += srcZStep;
                }
            }
        }
    }
}

static MNN::CoreInt8Functions::Im2ColFunc chooseIm2Col(const MNN::ConvolutionCommon::Im2ColParameter* im2colParam, size_t inputChannel) {
    bool fastIm2Col = im2colParam->kernelX == 1 && im2colParam->kernelY == 1 && im2colParam->icDiv4 % 2 == 0 &&
                      im2colParam->strideX == 1 && im2colParam->strideY == 1 && im2colParam->padX == 0 &&
                      im2colParam->padY == 0;
    int ih = im2colParam->ih, iw = im2colParam->iw;
    fastIm2Col &= (im2colParam->srcYStep == iw * PACK_UNIT && im2colParam->srcZStep == ih * iw * PACK_UNIT);
    if (fastIm2Col) {
        return _fastIm2Col;
    } else if (inputChannel <= PACK_UNIT) {
        return _im2colCommonZ1;
    } else {
        return _im2colCommon;
    }
}

static void _AVX512_MNNGetGemmUnit(int* UNIT, int* SRC_UNIT, int* DST_XUNIT) {
    *UNIT = 16;
    *SRC_UNIT = 16;
    *DST_XUNIT = 2;
}


void _AVX512_MNNInt8FunctionInit(void* functions) {
    auto gAVX2CoreInt8Functions = (MNN::CoreInt8Functions*)functions;
    gAVX2CoreInt8Functions->Int8GemmKernel = _AVX512_MNNGemmInt8AddBiasScale_16x4_Unit;
    gAVX2CoreInt8Functions->Int8GemmKernelFast = _AVX512_MNNGemmInt8AddBiasScale_16x4_Unit;
    // MatMul
    gAVX2CoreInt8Functions->MNNGetGemmUnit = _AVX512_MNNGetGemmUnit;
    // Im2Col
    gAVX2CoreInt8Functions->chooseIm2Col = chooseIm2Col;
    // Int8 <-> Float
    gAVX2CoreInt8Functions->MNNFloat2Int8 = _AVX512_MNNFloat2Int8;
    gAVX2CoreInt8Functions->MNNInt8ScaleToFloat = _AVX512_MNNInt8ScaleToFloat;

    // conv depthwise
    gAVX2CoreInt8Functions->ConvDepthwiseLineInt8 = _AVX512_MNNLineDepthWiseInt8AddBiasScaleUnit;

}

#endif
