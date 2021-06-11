//
//  GemmCommon.cpp
//  MNN
//
//  Created by MNN on b'2020/09/22'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_AVX512_VNNI
#include "FunctionSummary.hpp"
#include "core/Macro.h"

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
    if (realDst == 4) {
        for (int dz = 0; dz < dst_depth_quad; ++dz) {
            const auto weight_dz = weight + dz * src_depth_quad * (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT);
            const auto bias_dz = post->bias + dz * GEMM_INT8_UNIT;
            const float* scale_dz = nullptr;
            if (post->scale != nullptr) {
                scale_dz  = post->scale + dz * GEMM_INT8_UNIT;
            }
            auto dst_z           = dst + dz * dst_step_tmp;
            const auto src_x   = src;
            auto dst_x         = dst_z;
            __m512i D0 = _mm512_set1_epi32(0);
            __m512i D1 = _mm512_set1_epi32(0);
            __m512i D2 = _mm512_set1_epi32(0);
            __m512i D3 = _mm512_set1_epi32(0);

            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT) * sz;
                const auto src_z     = src_x + sz * GEMM_INT8_DST_XUNIT * GEMM_INT8_SRC_UNIT;
                auto w0 = _mm512_loadu_si512(weight_sz + GEMM_INT8_SRC_UNIT * 0);

                auto s0 = _mm512_broadcast_i32x4(mm_loadu_si128(src_z + GEMM_INT8_SRC_UNIT * 0));
                auto s1 = _mm512_broadcast_i32x4(mm_loadu_si128(src_z + GEMM_INT8_SRC_UNIT * 1));
                auto s2 = _mm512_broadcast_i32x4(mm_loadu_si128(src_z + GEMM_INT8_SRC_UNIT * 2));
                auto s3 = _mm512_broadcast_i32x4(mm_loadu_si128(src_z + GEMM_INT8_SRC_UNIT * 3));
                
                D0 = _mm512_dpbusds_epi32(D0, s0, w0);
                D1 = _mm512_dpbusds_epi32(D1, s1, w0);
                D2 = _mm512_dpbusds_epi32(D2, s2, w0);
                D3 = _mm512_dpbusds_epi32(D3, s3, w0);
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
            d20 = _mm_hadd_epi32(d20, d21);
            d22 = _mm_hadd_epi32(d22, d23);
            d30 = _mm_hadd_epi32(d30, d31);
            d32 = _mm_hadd_epi32(d32, d33);

            auto d0 = _mm_hadd_epi32(d00, d02);
            auto d1 = _mm_hadd_epi32(d10, d12);
            auto d2 = _mm_hadd_epi32(d20, d22);
            auto d3 = _mm_hadd_epi32(d30, d32);

            if (post->scale != nullptr) {
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
            } else {
                auto biasValue = _mm_loadu_si128((__m128i*)(bias_dz));
                __m128 f0 = _mm_cvtepi32_ps(_mm_add_epi32(d0, biasValue));
                __m128 f1 = _mm_cvtepi32_ps(_mm_add_epi32(d1, biasValue));
                __m128 f2 = _mm_cvtepi32_ps(_mm_add_epi32(d2, biasValue));
                __m128 f3 = _mm_cvtepi32_ps(_mm_add_epi32(d3, biasValue));
                _mm_storeu_ps(((float*)dst_x), f0);
                _mm_storeu_ps(((float*)dst_x) + 4, f1);
                _mm_storeu_ps(((float*)dst_x) + 8, f2);
                _mm_storeu_ps(((float*)dst_x) + 12, f3);
            }
        }
    }
    if (realDst == 3) {
        for (int dz = 0; dz < dst_depth_quad; ++dz) {
            const auto weight_dz = weight + dz * src_depth_quad * (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT);
            const auto bias_dz = post->bias + dz * GEMM_INT8_UNIT;
            const float* scale_dz = nullptr;
            if (post->scale != nullptr) {
                scale_dz  = post->scale + dz * GEMM_INT8_UNIT;
            }
            auto dst_z           = dst + dz * dst_step_tmp;
            const auto src_x   = src;
            auto dst_x         = dst_z;
            __m512i D0 = _mm512_set1_epi32(0);
            __m512i D1 = _mm512_set1_epi32(0);
            __m512i D2 = _mm512_set1_epi32(0);

            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT) * sz;
                const auto src_z     = src_x + sz * GEMM_INT8_DST_XUNIT * GEMM_INT8_SRC_UNIT;
                auto w0 = _mm512_loadu_si512(weight_sz + GEMM_INT8_SRC_UNIT * 0);

                auto s0 = _mm512_broadcast_i32x4(mm_loadu_si128(src_z + GEMM_INT8_SRC_UNIT * 0));
                auto s1 = _mm512_broadcast_i32x4(mm_loadu_si128(src_z + GEMM_INT8_SRC_UNIT * 1));
                auto s2 = _mm512_broadcast_i32x4(mm_loadu_si128(src_z + GEMM_INT8_SRC_UNIT * 2));
                
                D0 = _mm512_dpbusds_epi32(D0, s0, w0);
                D1 = _mm512_dpbusds_epi32(D1, s1, w0);
                D2 = _mm512_dpbusds_epi32(D2, s2, w0);
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

            d00 = _mm_hadd_epi32(d00, d01);
            d02 = _mm_hadd_epi32(d02, d03);
            d10 = _mm_hadd_epi32(d10, d11);
            d12 = _mm_hadd_epi32(d12, d13);
            d20 = _mm_hadd_epi32(d20, d21);
            d22 = _mm_hadd_epi32(d22, d23);

            auto d0 = _mm_hadd_epi32(d00, d02);
            auto d1 = _mm_hadd_epi32(d10, d12);
            auto d2 = _mm_hadd_epi32(d20, d22);
            
            if (post->scale != nullptr) {
                auto biasValue = _mm_loadu_si128((__m128i*)(bias_dz));
                d0 = _mm_add_epi32(d0, biasValue);
                d1 = _mm_add_epi32(d1, biasValue);
                d2 = _mm_add_epi32(d2, biasValue);
                auto scaleValue = _mm_loadu_ps(scale_dz);
                __m128 f0 = _mm_cvtepi32_ps(d0);
                __m128 f1 = _mm_cvtepi32_ps(d1);
                __m128 f2 = _mm_cvtepi32_ps(d2);
                f0 = _mm_mul_ps(f0, scaleValue);
                f1 = _mm_mul_ps(f1, scaleValue);
                f2 = _mm_mul_ps(f2, scaleValue);
                f0 = _mm_min_ps(f0, maxValue);
                f1 = _mm_min_ps(f1, maxValue);
                f2 = _mm_min_ps(f2, maxValue);
                f0 = _mm_max_ps(f0, minValue);
                f1 = _mm_max_ps(f1, minValue);
                f2 = _mm_max_ps(f2, minValue);
                auto m0 = _mm_cmplt_ps(f0, zero128);
                auto m1 = _mm_cmplt_ps(f1, zero128);
                auto m2 = _mm_cmplt_ps(f2, zero128);
                m0 = _mm_blendv_ps(plus, minus, m0);
                m1 = _mm_blendv_ps(plus, minus, m1);
                m2 = _mm_blendv_ps(plus, minus, m2);
                f0 = _mm_add_ps(f0, m0);
                f1 = _mm_add_ps(f1, m1);
                f2 = _mm_add_ps(f2, m2);
                // 3: _MM_FROUND_TO_ZERO
                d0 = _mm_cvtps_epi32(_mm_round_ps(f0, 3));
                d1 = _mm_cvtps_epi32(_mm_round_ps(f1, 3));
                d2 = _mm_cvtps_epi32(_mm_round_ps(f2, 3));
                
                // Int32 -> Int8
                d0 = _mm_packs_epi32(d0, d1);
                d2 = _mm_packs_epi32(d2, d2);
                d0 = _mm_packs_epi16(d0, d2);
                int32_t tempV[4];
                _mm_storeu_si128((__m128i*)tempV, d0);
                for (int j=0; j<realDst; ++j) {
                    ((int32_t*)dst_x)[j] = tempV[j];
                }
            } else {
                auto biasValue = _mm_loadu_si128((__m128i*)(bias_dz));
                __m128 f0 = _mm_cvtepi32_ps(_mm_add_epi32(d0, biasValue));
                __m128 f1 = _mm_cvtepi32_ps(_mm_add_epi32(d1, biasValue));
                __m128 f2 = _mm_cvtepi32_ps(_mm_add_epi32(d2, biasValue));
                _mm_storeu_ps(((float*)dst_x), f0);
                _mm_storeu_ps(((float*)dst_x) + 4, f1);
                _mm_storeu_ps(((float*)dst_x) + 8, f2);
            }
            
        }
    }
    if (realDst == 2) {
        for (int dz = 0; dz < dst_depth_quad; ++dz) {
            const auto weight_dz = weight + dz * src_depth_quad * (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT);
            const auto bias_dz = post->bias + dz * GEMM_INT8_UNIT;
            const float* scale_dz = nullptr;
            if (post->scale != nullptr) {
                scale_dz  = post->scale + dz * GEMM_INT8_UNIT;
            }
            auto dst_z           = dst + dz * dst_step_tmp;
            const auto src_x   = src;
            auto dst_x         = dst_z;
            __m512i D0 = _mm512_set1_epi32(0);
            __m512i D1 = _mm512_set1_epi32(0);

            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT) * sz;
                const auto src_z     = src_x + sz * GEMM_INT8_DST_XUNIT * GEMM_INT8_SRC_UNIT;
                auto w0 = _mm512_loadu_si512(weight_sz + GEMM_INT8_SRC_UNIT * 0);

                auto s0 = _mm512_broadcast_i32x4(mm_loadu_si128(src_z + GEMM_INT8_SRC_UNIT * 0));
                auto s1 = _mm512_broadcast_i32x4(mm_loadu_si128(src_z + GEMM_INT8_SRC_UNIT * 1));
                
                D0 = _mm512_dpbusds_epi32(D0, s0, w0);
                D1 = _mm512_dpbusds_epi32(D1, s1, w0);
            }
            auto d00 = _mm512_extracti32x4_epi32(D0, 0);
            auto d01 = _mm512_extracti32x4_epi32(D0, 1);
            auto d02 = _mm512_extracti32x4_epi32(D0, 2);
            auto d03 = _mm512_extracti32x4_epi32(D0, 3);

            auto d10 = _mm512_extracti32x4_epi32(D1, 0);
            auto d11 = _mm512_extracti32x4_epi32(D1, 1);
            auto d12 = _mm512_extracti32x4_epi32(D1, 2);
            auto d13 = _mm512_extracti32x4_epi32(D1, 3);

            d00 = _mm_hadd_epi32(d00, d01);
            d02 = _mm_hadd_epi32(d02, d03);
            d10 = _mm_hadd_epi32(d10, d11);
            d12 = _mm_hadd_epi32(d12, d13);
            auto d0 = _mm_hadd_epi32(d00, d02);
            auto d1 = _mm_hadd_epi32(d10, d12);

            if (post->scale != nullptr) {
                auto biasValue = _mm_loadu_si128((__m128i*)(bias_dz));
                d0 = _mm_add_epi32(d0, biasValue);
                d1 = _mm_add_epi32(d1, biasValue);
                auto scaleValue = _mm_loadu_ps(scale_dz);
                __m128 f0 = _mm_cvtepi32_ps(d0);
                __m128 f1 = _mm_cvtepi32_ps(d1);
                f0 = _mm_mul_ps(f0, scaleValue);
                f1 = _mm_mul_ps(f1, scaleValue);
                f0 = _mm_min_ps(f0, maxValue);
                f1 = _mm_min_ps(f1, maxValue);
                f0 = _mm_max_ps(f0, minValue);
                f1 = _mm_max_ps(f1, minValue);
                auto m0 = _mm_cmplt_ps(f0, zero128);
                auto m1 = _mm_cmplt_ps(f1, zero128);
                m0 = _mm_blendv_ps(plus, minus, m0);
                m1 = _mm_blendv_ps(plus, minus, m1);
                f0 = _mm_add_ps(f0, m0);
                f1 = _mm_add_ps(f1, m1);
                // 3: _MM_FROUND_TO_ZERO
                d0 = _mm_cvtps_epi32(_mm_round_ps(f0, 3));
                d1 = _mm_cvtps_epi32(_mm_round_ps(f1, 3));
                
                // Int32 -> Int8
                d0 = _mm_packs_epi32(d0, d1);
                d0 = _mm_packs_epi16(d0, d0);
                int32_t tempV[4];
                _mm_storeu_si128((__m128i*)tempV, d0);
                for (int j=0; j<realDst; ++j) {
                    ((int32_t*)dst_x)[j] = tempV[j];
                }
            } else {
                auto biasValue = _mm_loadu_si128((__m128i*)(bias_dz));
                __m128 f0 = _mm_cvtepi32_ps(_mm_add_epi32(d0, biasValue));
                __m128 f1 = _mm_cvtepi32_ps(_mm_add_epi32(d1, biasValue));
                _mm_storeu_ps(((float*)dst_x), f0);
                _mm_storeu_ps(((float*)dst_x) + 4, f1);
            }
        }
    }
    if (realDst == 1) {
        for (int dz = 0; dz < dst_depth_quad; ++dz) {
            const auto weight_dz = weight + dz * src_depth_quad * (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT);
            const auto bias_dz = post->bias + dz * GEMM_INT8_UNIT;
            const float* scale_dz = nullptr;
            if (post->scale != nullptr) {
                scale_dz  = post->scale + dz * GEMM_INT8_UNIT;
            }
            auto dst_z           = dst + dz * dst_step_tmp;
            const auto src_x   = src;
            auto dst_x         = dst_z;
            __m512i D0 = _mm512_set1_epi32(0);

            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT) * sz;
                const auto src_z     = src_x + sz * GEMM_INT8_DST_XUNIT * GEMM_INT8_SRC_UNIT;
                auto w0 = _mm512_loadu_si512(weight_sz + GEMM_INT8_SRC_UNIT * 0);

                auto s0 = _mm512_broadcast_i32x4(mm_loadu_si128(src_z + GEMM_INT8_SRC_UNIT * 0));
                
                D0 = _mm512_dpbusds_epi32(D0, s0, w0);
            }
            auto d00 = _mm512_extracti32x4_epi32(D0, 0);
            auto d01 = _mm512_extracti32x4_epi32(D0, 1);
            auto d02 = _mm512_extracti32x4_epi32(D0, 2);
            auto d03 = _mm512_extracti32x4_epi32(D0, 3);

            d00 = _mm_hadd_epi32(d00, d01);
            d02 = _mm_hadd_epi32(d02, d03);

            auto d0 = _mm_hadd_epi32(d00, d02);

            if (post->scale != nullptr) {
                auto biasValue = _mm_loadu_si128((__m128i*)(bias_dz));
                d0 = _mm_add_epi32(d0, biasValue);
                auto scaleValue = _mm_loadu_ps(scale_dz);
                __m128 f0 = _mm_cvtepi32_ps(d0);
                f0 = _mm_mul_ps(f0, scaleValue);
                f0 = _mm_min_ps(f0, maxValue);
                f0 = _mm_max_ps(f0, minValue);
                auto m0 = _mm_cmplt_ps(f0, zero128);
                m0 = _mm_blendv_ps(plus, minus, m0);
                f0 = _mm_add_ps(f0, m0);
                // 3: _MM_FROUND_TO_ZERO
                d0 = _mm_cvtps_epi32(_mm_round_ps(f0, 3));
                
                // Int32 -> Int8
                d0 = _mm_packs_epi32(d0, d0);
                d0 = _mm_packs_epi16(d0, d0);
                int32_t tempV[4];
                _mm_storeu_si128((__m128i*)tempV, d0);
                for (int j=0; j<realDst; ++j) {
                    ((int32_t*)dst_x)[j] = tempV[j];
                }
            } else {
                auto biasValue = _mm_loadu_si128((__m128i*)(bias_dz));
                __m128 f0 = _mm_cvtepi32_ps(_mm_add_epi32(d0, biasValue));
                _mm_storeu_ps(((float*)dst_x), f0);
            }
        }
    }
}
#endif
