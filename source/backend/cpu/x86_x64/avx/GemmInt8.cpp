//
//  GemmInt8.cpp
//  MNN
//
//  Created by MNN on 2020/09/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "GemmCommon.hpp"
#include "FunctionSummary.hpp"
#include "core/Macro.h"
#include <math.h>
namespace {
static inline __m128i mm_loadu_si128(const void* addr) {
    return _mm_loadu_si128((__m128i const*)addr);
}
static inline void MNN__mm_storeu_si64(void* add, __m128i value) {
    float temp[4];
    _mm_storeu_ps(temp, _mm_castsi128_ps(value));
    ::memcpy(add, temp, sizeof(int64_t));
}
}  // namespace


#ifdef MNN_X86_USE_ASM
extern "C" {
void _AVX_MNNGemmInt8AddBiasScale_16x4_UnitMain(int8_t* dst, const int8_t* src, const int8_t* weight, const size_t* strides, const QuanPostTreatParameters* post);
void _AVX_MNNGemmInt8AddBiasScale_16x4_Unit_1(int8_t* dst, const int8_t* src, const int8_t* weight, const size_t* strides, const QuanPostTreatParameters* post);
}
#endif
void _AVX_MNNGemmInt8AddBiasScale_16x4_Unit(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, const QuanPostTreatParameters* post, size_t realDst) {
#define EXTRACT_ADD(i)\
auto d##i##0 = _mm_castps_si128(_mm256_extractf128_ps(_mm256_castsi256_ps(D##i), 0));\
auto d##i##1 = _mm_castps_si128(_mm256_extractf128_ps(_mm256_castsi256_ps(D##i), 1));\
auto d##i = _mm_add_epi32(d##i##0, d##i##1);
#define COMPUTE(u, v)\
D##v##u = _mm256_add_epi32(D##v##u, _mm256_madd_epi16(W##u, S##v));

    const auto dst_step_tmp = dst_step / sizeof(int8_t);
    auto zero128 = _mm256_set1_ps(0.0f);
    auto minValue = _mm256_set1_ps(post->minValue);
    auto maxValue = _mm256_set1_ps(post->maxValue);
    auto plus = _mm256_set1_ps(0.5f);
    auto minus = _mm256_set1_ps(-0.5f);
    if (2 == realDst) {
        for (int dz = 0; dz < dst_depth_quad; ++dz) {
            const auto weight_dz = weight + dz * src_depth_quad * (16 * 8);
            const auto bias_dz = post->bias + dz * 8;
            const float* scale_dz = nullptr;
            if (post->scale != nullptr) {
                scale_dz  = post->scale + dz * 8;
            }
            auto dst_z           = dst + dz * dst_step_tmp;
            const auto src_x   = src;
            auto dst_x         = dst_z;
            __m256i D00 = _mm256_set1_epi32(0);
            __m256i D01 = _mm256_set1_epi32(0);
            __m256i D02 = _mm256_set1_epi32(0);
            __m256i D03 = _mm256_set1_epi32(0);
            __m256i D04 = _mm256_set1_epi32(0);
            __m256i D05 = _mm256_set1_epi32(0);
            __m256i D06 = _mm256_set1_epi32(0);
            __m256i D07 = _mm256_set1_epi32(0);
            __m256i D10 = _mm256_set1_epi32(0);
            __m256i D11 = _mm256_set1_epi32(0);
            __m256i D12 = _mm256_set1_epi32(0);
            __m256i D13 = _mm256_set1_epi32(0);
            __m256i D14 = _mm256_set1_epi32(0);
            __m256i D15 = _mm256_set1_epi32(0);
            __m256i D16 = _mm256_set1_epi32(0);
            __m256i D17 = _mm256_set1_epi32(0);

            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + (16 * 8) * sz;
                const auto src_z     = src_x + sz * 32;
                auto w0 = mm_loadu_si128(weight_sz + 16 * 0);
                auto w1 = mm_loadu_si128(weight_sz + 16 * 1);
                auto w2 = mm_loadu_si128(weight_sz + 16 * 2);
                auto w3 = mm_loadu_si128(weight_sz + 16 * 3);
                auto w4 = mm_loadu_si128(weight_sz + 16 * 4);
                auto w5 = mm_loadu_si128(weight_sz + 16 * 5);
                auto w6 = mm_loadu_si128(weight_sz + 16 * 6);
                auto w7 = mm_loadu_si128(weight_sz + 16 * 7);
                auto W0 = _mm256_cvtepi8_epi16(w0);
                auto W1 = _mm256_cvtepi8_epi16(w1);
                auto W2 = _mm256_cvtepi8_epi16(w2);
                auto W3 = _mm256_cvtepi8_epi16(w3);
                auto W4 = _mm256_cvtepi8_epi16(w4);
                auto W5 = _mm256_cvtepi8_epi16(w5);
                auto W6 = _mm256_cvtepi8_epi16(w6);
                auto W7 = _mm256_cvtepi8_epi16(w7);

                auto s0 = mm_loadu_si128(src_z + 16 * 0);
                auto S0 = _mm256_cvtepu8_epi16(s0);
                auto s1 = mm_loadu_si128(src_z + 16 * 1);
                auto S1 = _mm256_cvtepu8_epi16(s1);

                COMPUTE(0, 0);
                COMPUTE(1, 0);
                COMPUTE(2, 0);
                COMPUTE(3, 0);
                COMPUTE(4, 0);
                COMPUTE(5, 0);
                COMPUTE(6, 0);
                COMPUTE(7, 0);
                COMPUTE(0, 1);
                COMPUTE(1, 1);
                COMPUTE(2, 1);
                COMPUTE(3, 1);
                COMPUTE(4, 1);
                COMPUTE(5, 1);
                COMPUTE(6, 1);
                COMPUTE(7, 1);
            }
            D00 = _mm256_hadd_epi32(D00, D01);
            D02 = _mm256_hadd_epi32(D02, D03);
            D04 = _mm256_hadd_epi32(D04, D05);
            D06 = _mm256_hadd_epi32(D06, D07);

            D10 = _mm256_hadd_epi32(D10, D11);
            D12 = _mm256_hadd_epi32(D12, D13);
            D14 = _mm256_hadd_epi32(D14, D15);
            D16 = _mm256_hadd_epi32(D16, D17);

            D00 = _mm256_hadd_epi32(D00, D02);
            D04 = _mm256_hadd_epi32(D04, D06);

            D10 = _mm256_hadd_epi32(D10, D12);
            D14 = _mm256_hadd_epi32(D14, D16);

            auto c0 = _mm256_castps_si256(_mm256_permute2f128_ps(_mm256_castsi256_ps(D00), _mm256_castsi256_ps(D04), 32));
            auto c1 = _mm256_castps_si256(_mm256_permute2f128_ps(_mm256_castsi256_ps(D00), _mm256_castsi256_ps(D04), 49));
            auto e0 = _mm256_castps_si256(_mm256_permute2f128_ps(_mm256_castsi256_ps(D10), _mm256_castsi256_ps(D14), 32));
            auto e1 = _mm256_castps_si256(_mm256_permute2f128_ps(_mm256_castsi256_ps(D10), _mm256_castsi256_ps(D14), 49));
            auto D0 = _mm256_add_epi32(c0, c1);
            auto D1 = _mm256_add_epi32(e0, e1);

            if (post->scale != nullptr) {
                auto biasValue0 = _mm256_loadu_si256((__m256i*)(bias_dz));
                D0 = _mm256_add_epi32(D0, biasValue0);
                D1 = _mm256_add_epi32(D1, biasValue0);

                auto scaleValue = _mm256_loadu_ps(scale_dz);
                auto f0 = _mm256_cvtepi32_ps(D0);
                auto f1 = _mm256_cvtepi32_ps(D1);
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
                D1 = _mm256_cvtps_epi32(_mm256_round_ps(f1, 3));
                auto offset = _mm256_set1_epi32(128);
                D0 = _mm256_add_epi32(D0, offset);
                D1 = _mm256_add_epi32(D1, offset);

                // Int32 -> Int8
                D0 = _mm256_packs_epi32(D0, _mm256_castps_si256(_mm256_permute2f128_ps(_mm256_castsi256_ps(D0), _mm256_castsi256_ps(D0), 1)));
                D1 = _mm256_packs_epi32(D1, _mm256_castps_si256(_mm256_permute2f128_ps(_mm256_castsi256_ps(D1), _mm256_castsi256_ps(D1), 1)));
                auto d0 = _mm_packus_epi16(_mm256_castsi256_si128(D0), _mm256_castsi256_si128(_mm256_castps_si256(zero128)));
                auto d1 = _mm_packus_epi16(_mm256_castsi256_si128(D1), _mm256_castsi256_si128(_mm256_castps_si256(zero128)));
                MNN__mm_storeu_si64(dst_x, d0);
                MNN__mm_storeu_si64(dst_x + 8, d1);
            } else {
                auto biasValue0 = _mm256_loadu_si256((__m256i*)(bias_dz));
                auto f0 = _mm256_cvtepi32_ps(_mm256_add_epi32(D0, biasValue0));
                auto f1 = _mm256_cvtepi32_ps(_mm256_add_epi32(D1, biasValue0));
                _mm256_storeu_ps(((float*)dst_x), f0);
                _mm256_storeu_ps(((float*)dst_x) + 8, f1);
            }
        }
        return;
    }
    // e = 1
    for (int dz = 0; dz < dst_depth_quad; ++dz) {
        const auto weight_dz = weight + dz * src_depth_quad * (16 * 8);
        const auto bias_dz = post->bias + dz * 8;
        const float* scale_dz = nullptr;
        if (post->scale != nullptr) {
            scale_dz  = post->scale + dz * 8;
        }
        auto dst_z           = dst + dz * dst_step_tmp;
        const auto src_x   = src;
        auto dst_x         = dst_z;
        __m256i D00 = _mm256_set1_epi32(0);
        __m256i D01 = _mm256_set1_epi32(0);
        __m256i D02 = _mm256_set1_epi32(0);
        __m256i D03 = _mm256_set1_epi32(0);
        __m256i D04 = _mm256_set1_epi32(0);
        __m256i D05 = _mm256_set1_epi32(0);
        __m256i D06 = _mm256_set1_epi32(0);
        __m256i D07 = _mm256_set1_epi32(0);

        for (int sz = 0; sz < src_depth_quad; ++sz) {
            const auto weight_sz = weight_dz + (16 * 8) * sz;
            const auto src_z     = src_x + sz * 32;
            auto w0 = mm_loadu_si128(weight_sz + 16 * 0);
            auto w1 = mm_loadu_si128(weight_sz + 16 * 1);
            auto w2 = mm_loadu_si128(weight_sz + 16 * 2);
            auto w3 = mm_loadu_si128(weight_sz + 16 * 3);
            auto w4 = mm_loadu_si128(weight_sz + 16 * 4);
            auto w5 = mm_loadu_si128(weight_sz + 16 * 5);
            auto w6 = mm_loadu_si128(weight_sz + 16 * 6);
            auto w7 = mm_loadu_si128(weight_sz + 16 * 7);
            auto W0 = _mm256_cvtepi8_epi16(w0);
            auto W1 = _mm256_cvtepi8_epi16(w1);
            auto W2 = _mm256_cvtepi8_epi16(w2);
            auto W3 = _mm256_cvtepi8_epi16(w3);
            auto W4 = _mm256_cvtepi8_epi16(w4);
            auto W5 = _mm256_cvtepi8_epi16(w5);
            auto W6 = _mm256_cvtepi8_epi16(w6);
            auto W7 = _mm256_cvtepi8_epi16(w7);

            auto s0 = mm_loadu_si128(src_z + 16 * 0);
            auto S0 = _mm256_cvtepu8_epi16(s0);

            COMPUTE(0, 0);
            COMPUTE(1, 0);
            COMPUTE(2, 0);
            COMPUTE(3, 0);
            COMPUTE(4, 0);
            COMPUTE(5, 0);
            COMPUTE(6, 0);
            COMPUTE(7, 0);
        }
        D00 = _mm256_hadd_epi32(D00, D01);
        D02 = _mm256_hadd_epi32(D02, D03);
        D04 = _mm256_hadd_epi32(D04, D05);
        D06 = _mm256_hadd_epi32(D06, D07);

        D00 = _mm256_hadd_epi32(D00, D02);
        D04 = _mm256_hadd_epi32(D04, D06);

        auto c0 = _mm256_castps_si256(_mm256_permute2f128_ps(_mm256_castsi256_ps(D00), _mm256_castsi256_ps(D04), 32));
        auto c1 = _mm256_castps_si256(_mm256_permute2f128_ps(_mm256_castsi256_ps(D00), _mm256_castsi256_ps(D04), 49));
        auto D0 = _mm256_add_epi32(c0, c1);

        if (post->scale != nullptr) {
            auto biasValue0 = _mm256_loadu_si256((__m256i*)(bias_dz));
            D0 = _mm256_add_epi32(D0, biasValue0);

            auto scaleValue = _mm256_loadu_ps(scale_dz);
            auto f0 = _mm256_cvtepi32_ps(D0);
            f0 = _mm256_mul_ps(f0, scaleValue);
            f0 = _mm256_min_ps(f0, maxValue);
            f0 = _mm256_max_ps(f0, minValue);
            auto m0 = _mm256_cmp_ps(f0, zero128, 1);
            m0 = _mm256_blendv_ps(plus, minus, m0);

            f0 = _mm256_add_ps(f0, m0);
            // 3: _MM_FROUND_TO_ZERO
            D0 = _mm256_cvtps_epi32(_mm256_round_ps(f0, 3));
            auto offset = _mm256_set1_epi32(128);
            D0 = _mm256_add_epi32(D0, offset);

            // Int32 -> Int8
            D0 = _mm256_packs_epi32(D0, _mm256_castps_si256(_mm256_permute2f128_ps(_mm256_castsi256_ps(D0), _mm256_castsi256_ps(D0), 1)));
            auto d0 = _mm_packus_epi16(_mm256_castsi256_si128(D0), _mm256_castsi256_si128(_mm256_castps_si256(zero128)));
            MNN__mm_storeu_si64(dst_x, d0);
        } else {
            auto biasValue0 = _mm256_loadu_si256((__m256i*)(bias_dz));
            auto f0 = _mm256_cvtepi32_ps(_mm256_add_epi32(D0, biasValue0));
            _mm256_storeu_ps(((float*)dst_x), f0);
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
    if (2 == realDst) {
        for (int dz = 0; dz < dst_depth_quad; ++dz) {
            const auto weight_dz = weight + dz * src_depth_quad * (8 * 16);
            const auto bias_dz = post->bias + dz * 8;
            const float* scale_dz = nullptr;
            if (post->scale != nullptr) {
                scale_dz  = post->scale + dz * 8;
            }
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

            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + (8 * 16) * sz;
                const auto src_z     = src_x + sz * 2 * 16;
                auto w0 = _mm256_loadu_si256((__m256i*)(weight_sz + 16 * 0));
                auto w1 = _mm256_loadu_si256((__m256i*)(weight_sz + 16 * 2));
                auto w2 = _mm256_loadu_si256((__m256i*)(weight_sz + 16 * 4));
                auto w3 = _mm256_loadu_si256((__m256i*)(weight_sz + 16 * 6));

                auto s0 = _mm256_broadcastsi128_si256(mm_loadu_si128(src_z + 16 * 0));
                auto s1 = _mm256_broadcastsi128_si256(mm_loadu_si128(src_z + 16 * 1));

                D00 = _mm256_add_epi32(D00, _mm256_madd_epi16(_mm256_maddubs_epi16(s0, w0), oneValue));
                D01 = _mm256_add_epi32(D01, _mm256_madd_epi16(_mm256_maddubs_epi16(s0, w1), oneValue));
                D02 = _mm256_add_epi32(D02, _mm256_madd_epi16(_mm256_maddubs_epi16(s0, w2), oneValue));
                D03 = _mm256_add_epi32(D03, _mm256_madd_epi16(_mm256_maddubs_epi16(s0, w3), oneValue));
                D10 = _mm256_add_epi32(D10, _mm256_madd_epi16(_mm256_maddubs_epi16(s1, w0), oneValue));
                D11 = _mm256_add_epi32(D11, _mm256_madd_epi16(_mm256_maddubs_epi16(s1, w1), oneValue));
                D12 = _mm256_add_epi32(D12, _mm256_madd_epi16(_mm256_maddubs_epi16(s1, w2), oneValue));
                D13 = _mm256_add_epi32(D13, _mm256_madd_epi16(_mm256_maddubs_epi16(s1, w3), oneValue));
            }

            auto D0 = _mm256_hadd_epi32(_mm256_permute2f128_si256(D00, D01, 32), _mm256_permute2f128_si256(D00, D01, 49));
            auto D1 = _mm256_hadd_epi32(_mm256_permute2f128_si256(D02, D03, 32), _mm256_permute2f128_si256(D02, D03, 49));
            auto D2 = _mm256_hadd_epi32(_mm256_permute2f128_si256(D10, D11, 32), _mm256_permute2f128_si256(D10, D11, 49));
            auto D3 = _mm256_hadd_epi32(_mm256_permute2f128_si256(D12, D13, 32), _mm256_permute2f128_si256(D12, D13, 49));

            D0 = _mm256_hadd_epi32(_mm256_permute2f128_si256(D0, D1, 32), _mm256_permute2f128_si256(D0, D1, 49));
            D2 = _mm256_hadd_epi32(_mm256_permute2f128_si256(D2, D3, 32), _mm256_permute2f128_si256(D2, D3, 49));

            if (post->scale != nullptr) {
                auto biasValue = _mm256_castps_si256(_mm256_loadu_ps((const float*)bias_dz));
                D0 = _mm256_add_epi32(D0, biasValue);
                D2 = _mm256_add_epi32(D2, biasValue);
                auto scaleValue = _mm256_loadu_ps(scale_dz);
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
                auto offset = _mm256_set1_epi32(128);
                D0 = _mm256_add_epi32(D0, offset);
                D2 = _mm256_add_epi32(D2, offset);

                auto d0 = _mm256_extracti128_si256(D0, 0);
                auto d1 = _mm256_extracti128_si256(D0, 1);
                auto d2 = _mm256_extracti128_si256(D2, 0);
                auto d3 = _mm256_extracti128_si256(D2, 1);

                // Int32 -> Int8
                d0 = _mm_packs_epi32(d0, d1);
                d2 = _mm_packs_epi32(d2, d3);
                d0 = _mm_packus_epi16(d0, d2);
                _mm_storeu_si128((__m128i*)dst_x, d0);
            } else {
                auto biasValue = _mm256_castps_si256(_mm256_loadu_ps((const float*)(bias_dz)));
                auto f0 = _mm256_cvtepi32_ps(_mm256_add_epi32(D0, biasValue));
                auto f1 = _mm256_cvtepi32_ps(_mm256_add_epi32(D1, biasValue));
                _mm256_storeu_ps(((float*)dst_x), f0);
                _mm256_storeu_ps(((float*)dst_x + 8), f1);
            }
        }
        return;
    }
    if (1 == realDst) {
        for (int dz = 0; dz < dst_depth_quad; ++dz) {
            const auto weight_dz = weight + dz * src_depth_quad * (8 * 16);
            const auto bias_dz = post->bias + dz * 8;
            const float* scale_dz = nullptr;
            if (post->scale != nullptr) {
                scale_dz  = post->scale + dz * 8;
            }
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

            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + (8 * 16) * sz;
                const auto src_z     = src_x + sz * 2 * 16;
                auto w0 = _mm256_loadu_si256((__m256i*)(weight_sz + 16 * 0));
                auto w1 = _mm256_loadu_si256((__m256i*)(weight_sz + 16 * 2));
                auto w2 = _mm256_loadu_si256((__m256i*)(weight_sz + 16 * 4));
                auto w3 = _mm256_loadu_si256((__m256i*)(weight_sz + 16 * 6));

                auto s0 = _mm256_broadcastsi128_si256(mm_loadu_si128(src_z + 16 * 0));

                D00 = _mm256_add_epi32(D00, _mm256_madd_epi16(_mm256_maddubs_epi16(s0, w0), oneValue));
                D01 = _mm256_add_epi32(D01, _mm256_madd_epi16(_mm256_maddubs_epi16(s0, w1), oneValue));
                D02 = _mm256_add_epi32(D02, _mm256_madd_epi16(_mm256_maddubs_epi16(s0, w2), oneValue));
                D03 = _mm256_add_epi32(D03, _mm256_madd_epi16(_mm256_maddubs_epi16(s0, w3), oneValue));
            }

            auto D0 = _mm256_hadd_epi32(_mm256_permute2f128_si256(D00, D01, 32), _mm256_permute2f128_si256(D00, D01, 49));
            auto D1 = _mm256_hadd_epi32(_mm256_permute2f128_si256(D02, D03, 32), _mm256_permute2f128_si256(D02, D03, 49));
            auto D2 = _mm256_hadd_epi32(_mm256_permute2f128_si256(D10, D11, 32), _mm256_permute2f128_si256(D10, D11, 49));
            auto D3 = _mm256_hadd_epi32(_mm256_permute2f128_si256(D12, D13, 32), _mm256_permute2f128_si256(D12, D13, 49));

            D0 = _mm256_hadd_epi32(_mm256_permute2f128_si256(D0, D1, 32), _mm256_permute2f128_si256(D0, D1, 49));
            D2 = _mm256_hadd_epi32(_mm256_permute2f128_si256(D2, D3, 32), _mm256_permute2f128_si256(D2, D3, 49));

            if (post->scale != nullptr) {
                auto biasValue = _mm256_castps_si256(_mm256_loadu_ps((const float*)bias_dz));
                D0 = _mm256_add_epi32(D0, biasValue);
                D2 = _mm256_add_epi32(D2, biasValue);
                auto scaleValue = _mm256_loadu_ps(scale_dz);
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
                auto offset = _mm256_set1_epi32(128);
                D0 = _mm256_add_epi32(D0, offset);
                D2 = _mm256_add_epi32(D2, offset);

                auto d0 = _mm256_extracti128_si256(D0, 0);
                auto d1 = _mm256_extracti128_si256(D0, 1);
                auto d2 = _mm256_extracti128_si256(D2, 0);
                auto d3 = _mm256_extracti128_si256(D2, 1);

                // Int32 -> Int8
                d0 = _mm_packs_epi32(d0, d1);
                d2 = _mm_packs_epi32(d2, d3);
                d0 = _mm_packus_epi16(d0, d2);
                MNN__mm_storeu_si64((__m128i*)dst_x, d0);
            } else {
                auto biasValue = _mm256_castps_si256(_mm256_loadu_ps((const float*)(bias_dz)));
                auto f0 = _mm256_cvtepi32_ps(_mm256_add_epi32(D0, biasValue));
                _mm256_storeu_ps(((float*)dst_x), f0);
            }
        }
        return;
    }
}

#undef MAIN_COMPUTE
#undef STORE_TEMP
void _AVX_MNNLineDepthWiseInt8AddBiasScaleUnit(int8_t* dstO, const int8_t* srcO, const int8_t* weightO, const QuanPostTreatParameters* parameters, size_t width, size_t src_w_step, size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step) {
    auto dst = dstO;
    auto src = (const int16_t*)srcO;
    int widthC4 = width / 4;
    int widthRemain = width % 4;
    auto weight = (const int16_t*)weightO;
    auto biasValue = _mm256_castps_si256(_mm256_loadu_ps((const float*)parameters->bias));
    auto scaleValue = _mm256_loadu_ps((const float*)parameters->scale);
    __m256i d0, d1, d2, d3;
    int dx, fx, fy;
    __m256i srcValue0;
    auto srcTemp0 = (int64_t*)(&srcValue0);
    __m256i weightValue;
    auto weightTemp = (int64_t*)(&weightValue);
    __m256i zero = _mm256_xor_si256(srcValue0, srcValue0);
    __m256 zero128 = _mm256_set1_ps(0.0f);
    __m128i minValue = _mm_set1_epi16(parameters->minValue + 128);
    __m128i maxValue = _mm_set1_epi16(parameters->maxValue + 128);
    __m256 plus = _mm256_set1_ps(0.5f);
    __m256 minus = _mm256_set1_ps(-0.5f);
    for (dx = 0; dx < widthC4; ++dx) {
        d0 = biasValue;
        d1 = biasValue;
        d2 = biasValue;
        d3 = biasValue;

        auto dst_x          = dst;
        const auto src_z    = src;
        for (fy = 0; fy < fh; ++fy) {
            const auto src_y    = src_z + fy * dilateY_step;
            const auto weight_y = weight + fy * fw * 8;
            for (fx = 0; fx < fw; ++fx) {
                const auto src_x    = src_y + fx * dilateX_step;
                auto S0 = _mm256_castsi128_si256(_mm_castps_si128(_mm_loadu_ps((const float*)(src_x + 0 * src_w_step))));
                auto S1 = _mm256_castsi128_si256(_mm_castps_si128(_mm_loadu_ps((const float*)(src_x + 1 * src_w_step))));
                auto S2 = _mm256_castsi128_si256(_mm_castps_si128(_mm_loadu_ps((const float*)(src_x + 2 * src_w_step))));
                auto S3 = _mm256_castsi128_si256(_mm_castps_si128(_mm_loadu_ps((const float*)(src_x + 3 * src_w_step))));
                const auto weight_x = weight_y + 8 * fx;
                auto W0 = _mm256_castsi128_si256(_mm_castps_si128(_mm_loadu_ps((const float*)weight_x)));
                auto s00 = _mm256_unpacklo_epi16(S0, zero);
                auto s10 = _mm256_unpacklo_epi16(S1, zero);
                auto s20 = _mm256_unpacklo_epi16(S2, zero);
                auto s30 = _mm256_unpacklo_epi16(S3, zero);
                auto w00 = _mm256_unpacklo_epi16(W0, zero);
                auto s01 = _mm256_unpackhi_epi16(S0, zero);
                auto s11 = _mm256_unpackhi_epi16(S1, zero);
                auto s21 = _mm256_unpackhi_epi16(S2, zero);
                auto s31 = _mm256_unpackhi_epi16(S3, zero);
                auto w01 = _mm256_unpackhi_epi16(W0, zero);
                S0 = _mm256_permute2f128_si256(s00, s01, 32);
                S1 = _mm256_permute2f128_si256(s10, s11, 32);
                S2 = _mm256_permute2f128_si256(s20, s21, 32);
                S3 = _mm256_permute2f128_si256(s30, s31, 32);
                W0 = _mm256_permute2f128_si256(w00, w01, 32);
                d0 = _mm256_add_epi32(d0, _mm256_madd_epi16(W0, S0));
                d1 = _mm256_add_epi32(d1, _mm256_madd_epi16(W0, S1));
                d2 = _mm256_add_epi32(d2, _mm256_madd_epi16(W0, S2));
                d3 = _mm256_add_epi32(d3, _mm256_madd_epi16(W0, S3));
            }
        }
        __m256 f0 = _mm256_cvtepi32_ps(d0);
        __m256 f1 = _mm256_cvtepi32_ps(d1);
        __m256 f2 = _mm256_cvtepi32_ps(d2);
        __m256 f3 = _mm256_cvtepi32_ps(d3);
        f0 = _mm256_mul_ps(f0, scaleValue);
        f1 = _mm256_mul_ps(f1, scaleValue);
        f2 = _mm256_mul_ps(f2, scaleValue);
        f3 = _mm256_mul_ps(f3, scaleValue);
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
        src += src_w_step * 4;
    }
    switch (widthRemain) {
        case 3:
        {
            d0 = biasValue;
            d1 = biasValue;
            d2 = biasValue;

            auto dst_x          = dst;
            const auto src_z    = src;
            for (fy = 0; fy < fh; ++fy) {
                const auto src_y    = src_z + fy * dilateY_step;
                const auto weight_y = weight + fy * fw * 8;
                for (fx = 0; fx < fw; ++fx) {
                    const auto src_x    = src_y + fx * dilateX_step;
                    auto S0 = _mm256_castsi128_si256(_mm_castps_si128(_mm_loadu_ps((const float*)(src_x + 0 * src_w_step))));
                    auto S1 = _mm256_castsi128_si256(_mm_castps_si128(_mm_loadu_ps((const float*)(src_x + 1 * src_w_step))));
                    auto S2 = _mm256_castsi128_si256(_mm_castps_si128(_mm_loadu_ps((const float*)(src_x + 2 * src_w_step))));
                    const auto weight_x = weight_y + 8 * fx;
                    auto W0 = _mm256_castsi128_si256(_mm_castps_si128(_mm_loadu_ps((const float*)weight_x)));
                    auto s00 = _mm256_unpacklo_epi16(S0, zero);
                    auto s10 = _mm256_unpacklo_epi16(S1, zero);
                    auto s20 = _mm256_unpacklo_epi16(S2, zero);
                    auto w00 = _mm256_unpacklo_epi16(W0, zero);
                    auto s01 = _mm256_unpackhi_epi16(S0, zero);
                    auto s11 = _mm256_unpackhi_epi16(S1, zero);
                    auto s21 = _mm256_unpackhi_epi16(S2, zero);
                    auto w01 = _mm256_unpackhi_epi16(W0, zero);
                    S0 = _mm256_permute2f128_si256(s00, s01, 32);
                    S1 = _mm256_permute2f128_si256(s10, s11, 32);
                    S2 = _mm256_permute2f128_si256(s20, s21, 32);
                    W0 = _mm256_permute2f128_si256(w00, w01, 32);
                    d0 = _mm256_add_epi32(d0, _mm256_madd_epi16(W0, S0));
                    d1 = _mm256_add_epi32(d1, _mm256_madd_epi16(W0, S1));
                    d2 = _mm256_add_epi32(d2, _mm256_madd_epi16(W0, S2));
                }
            }
            __m256 f0 = _mm256_cvtepi32_ps(d0);
            __m256 f1 = _mm256_cvtepi32_ps(d1);
            __m256 f2 = _mm256_cvtepi32_ps(d2);
            f0 = _mm256_mul_ps(f0, scaleValue);
            f1 = _mm256_mul_ps(f1, scaleValue);
            f2 = _mm256_mul_ps(f2, scaleValue);
            auto m0 = _mm256_cmp_ps(f0, zero128, 1);
            auto m1 = _mm256_cmp_ps(f1, zero128, 1);
            auto m2 = _mm256_cmp_ps(f2, zero128, 1);
            m0 = _mm256_blendv_ps(plus, minus, m0);
            m1 = _mm256_blendv_ps(plus, minus, m1);
            m2 = _mm256_blendv_ps(plus, minus, m2);
            f0 = _mm256_add_ps(f0, m0);
            f1 = _mm256_add_ps(f1, m1);
            f2 = _mm256_add_ps(f2, m2);
            // 3: _MM_FROUND_TO_ZERO
            d0 = _mm256_cvtps_epi32(_mm256_round_ps(f0, 3));
            d1 = _mm256_cvtps_epi32(_mm256_round_ps(f1, 3));
            d2 = _mm256_cvtps_epi32(_mm256_round_ps(f2, 3));

            auto offset = _mm256_set1_epi32(128);
            d0 = _mm256_add_epi32(d0, offset);
            d1 = _mm256_add_epi32(d1, offset);
            d2 = _mm256_add_epi32(d2, offset);

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
            D2 = _mm_max_epi16(D2, minValue);
            D3 = _mm_min_epi16(D3, maxValue);
            D3 = _mm_max_epi16(D3, minValue);
            _mm_storeu_ps((float*)(dst), _mm_castsi128_ps(_mm_packus_epi16(D0, D1)));
            MNN__mm_storeu_si64((float*)(dst + 16), _mm_packus_epi16(D2, D3));
            break;
        }
        case 2:
        {
            d0 = biasValue;
            d1 = biasValue;

            auto dst_x          = dst;
            const auto src_z    = src;
            for (fy = 0; fy < fh; ++fy) {
                const auto src_y    = src_z + fy * dilateY_step;
                const auto weight_y = weight + fy * fw * 8;
                for (fx = 0; fx < fw; ++fx) {
                    const auto src_x    = src_y + fx * dilateX_step;
                    auto S0 = _mm256_castsi128_si256(_mm_castps_si128(_mm_loadu_ps((const float*)(src_x + 0 * src_w_step))));
                    auto S1 = _mm256_castsi128_si256(_mm_castps_si128(_mm_loadu_ps((const float*)(src_x + 1 * src_w_step))));
                    const auto weight_x = weight_y + 8 * fx;
                    auto W0 = _mm256_castsi128_si256(_mm_castps_si128(_mm_loadu_ps((const float*)weight_x)));
                    auto s00 = _mm256_unpacklo_epi16(S0, zero);
                    auto s10 = _mm256_unpacklo_epi16(S1, zero);
                    auto w00 = _mm256_unpacklo_epi16(W0, zero);
                    auto s01 = _mm256_unpackhi_epi16(S0, zero);
                    auto s11 = _mm256_unpackhi_epi16(S1, zero);
                    auto w01 = _mm256_unpackhi_epi16(W0, zero);
                    S0 = _mm256_permute2f128_si256(s00, s01, 32);
                    S1 = _mm256_permute2f128_si256(s10, s11, 32);
                    W0 = _mm256_permute2f128_si256(w00, w01, 32);
                    d0 = _mm256_add_epi32(d0, _mm256_madd_epi16(W0, S0));
                    d1 = _mm256_add_epi32(d1, _mm256_madd_epi16(W0, S1));
                }
            }
            __m256 f0 = _mm256_cvtepi32_ps(d0);
            __m256 f1 = _mm256_cvtepi32_ps(d1);
            f0 = _mm256_mul_ps(f0, scaleValue);
            f1 = _mm256_mul_ps(f1, scaleValue);
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
            break;
        }
        case 1:
        {
            d0 = biasValue;

            auto dst_x          = dst;
            const auto src_z    = src;
            for (fy = 0; fy < fh; ++fy) {
                const auto src_y    = src_z + fy * dilateY_step;
                const auto weight_y = weight + fy * fw * 8;
                for (fx = 0; fx < fw; ++fx) {
                    const auto src_x    = src_y + fx * dilateX_step;
                    auto S0 = _mm256_castsi128_si256(_mm_castps_si128(_mm_loadu_ps((const float*)(src_x + 0 * src_w_step))));
                    const auto weight_x = weight_y + 8 * fx;
                    auto W0 = _mm256_castsi128_si256(_mm_castps_si128(_mm_loadu_ps((const float*)weight_x)));
                    auto s00 = _mm256_unpacklo_epi16(S0, zero);
                    auto w00 = _mm256_unpacklo_epi16(W0, zero);
                    auto s01 = _mm256_unpackhi_epi16(S0, zero);
                    auto w01 = _mm256_unpackhi_epi16(W0, zero);
                    S0 = _mm256_permute2f128_si256(s00, s01, 32);
                    W0 = _mm256_permute2f128_si256(w00, w01, 32);
                    d0 = _mm256_add_epi32(d0, _mm256_madd_epi16(W0, S0));
                }
            }
            __m256 f0 = _mm256_cvtepi32_ps(d0);
            f0 = _mm256_mul_ps(f0, scaleValue);
            auto m0 = _mm256_cmp_ps(f0, zero128, 1);
            m0 = _mm256_blendv_ps(plus, minus, m0);
            f0 = _mm256_add_ps(f0, m0);
            // 3: _MM_FROUND_TO_ZERO
            d0 = _mm256_cvtps_epi32(_mm256_round_ps(f0, 3));
            auto offset = _mm256_set1_epi32(128);
            d0 = _mm256_add_epi32(d0, offset);

            auto e0 = _mm256_permute2f128_si256(d0, d0, 32);
            auto e1 = _mm256_permute2f128_si256(d0, d0, 49);
            // Int32 -> Int8
            d0 = _mm256_packs_epi32(e0, e1);
            auto D0 = _mm256_extracti128_si256(d0, 0);
            auto D1 = _mm256_extracti128_si256(d0, 1);

            D0 = _mm_min_epi16(D0, maxValue);
            D1 = _mm_min_epi16(D1, maxValue);
            D0 = _mm_max_epi16(D0, minValue);
            D1 = _mm_max_epi16(D1, minValue);

            MNN__mm_storeu_si64((float*)(dst), _mm_packus_epi16(D0, D1));
            break;
        }

        default:
            break;
    }
}

void _AVX_MNNFloat2Int8(const float* src, int8_t* dst, size_t sizeQuad, const float* scalep, ssize_t minV, ssize_t maxV, ssize_t zeroPoint) {
    auto zero = _mm256_set1_epi32(0);
    auto minValue = _mm256_set1_ps(minV);
    auto maxValue = _mm256_set1_ps(maxV);
    auto zeroPointValue = _mm256_set1_ps(zeroPoint);
    auto offset = _mm256_set1_epi32(128);
    auto plus = _mm256_set1_ps(0.5f);
    auto minus = _mm256_set1_ps(-0.5f);
    auto scaleValue = _mm256_loadu_ps(scalep);

    for (int i = 0; i < sizeQuad; ++i) {
        auto f0 = _mm256_loadu_ps(src + 8 * i);
        f0 = _mm256_mul_ps(f0, scaleValue);
        f0 = _mm256_add_ps(f0, zeroPointValue);
        f0 = _mm256_min_ps(f0, maxValue);
        f0 = _mm256_max_ps(f0, minValue);
        auto m0 = _mm256_cmp_ps(f0, _mm256_castsi256_ps(zero), 1);
        m0 = _mm256_blendv_ps(plus, minus, m0);
        f0 = _mm256_add_ps(f0, m0);
        // 3: _MM_FROUND_TO_ZERO
        auto d0 = _mm256_cvtps_epi32(_mm256_round_ps(f0, 3));
        d0 = _mm256_add_epi32(d0, offset);
        d0 = _mm256_packs_epi32(d0, _mm256_setzero_si256());
        d0 = _mm256_permute4x64_epi64(d0, 0xD8);
#if defined(_MSC_VER)
        __m256i x = static_cast<__m256i>(_mm256_packus_epi16(d0, _mm256_setzero_si256()));
        *((int64_t*)dst + i) = x.m256i_i64[0];
#else
         __v4di x = static_cast<__v4di>(_mm256_packus_epi16(d0, _mm256_setzero_si256()));
         *((int64_t*)dst + i) = x[0];
#endif
    }
}

void _AVX_MNNInt8ScaleToFloat(float* dst, const int8_t* src, const float* scale, size_t sizeQuad, ssize_t zeroPoint) {
    auto sizeC4 = sizeQuad / 4;
    auto sizeRemain = sizeQuad % 4;
    auto zero = _mm256_set1_epi32(0);
    auto scaleValue = _mm256_loadu_ps(scale);
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
        _mm256_storeu_ps(dst + 8 * 0, _mm256_mul_ps(s0_f, scaleValue));
        _mm256_storeu_ps(dst + 8 * 1, _mm256_mul_ps(s1_f, scaleValue));
        _mm256_storeu_ps(dst + 8 * 2, _mm256_mul_ps(s2_f, scaleValue));
        _mm256_storeu_ps(dst + 8 * 3, _mm256_mul_ps(s3_f, scaleValue));
        src += 32;
        dst += 32;
    }
    if (sizeRemain > 0) {
        int8_t srcTemp[256];
        ::memcpy(srcTemp, src, sizeRemain * 8);
        auto s = *(__m256i*)srcTemp;
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
        switch (sizeRemain) {
            case 3:
                _mm256_storeu_ps(dst + 8 * 0, _mm256_mul_ps(s0_f, scaleValue));
                _mm256_storeu_ps(dst + 8 * 1, _mm256_mul_ps(s1_f, scaleValue));
                _mm256_storeu_ps(dst + 8 * 2, _mm256_mul_ps(s2_f, scaleValue));
                break;
            case 2:
                _mm256_storeu_ps(dst + 8 * 0, _mm256_mul_ps(s0_f, scaleValue));
                _mm256_storeu_ps(dst + 8 * 1, _mm256_mul_ps(s1_f, scaleValue));
                break;
            case 1:
                _mm256_storeu_ps(dst + 8 * 0, _mm256_mul_ps(s0_f, scaleValue));
                break;
            default:
                break;
        }
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
    inputOrigin += xIndexStart * 8;
    int icDiv16 = icDiv8 / 2;
    int icDiv16R = icDiv8 % 2;
    for (int i = 0; i < realDstCount; ++i) {
        auto colAddrI = colAddr + 16 * i;
        auto inputK   = inputOrigin + 8 * i;
        for (int sz = 0; sz < icDiv16; ++sz) {
            auto inputZ0           = inputK + srcZStep * sz * 2;
            auto inputZ1           = inputK + srcZStep * (sz * 2 + 1);
            auto dstK0         = colAddrI + (sz * 2) * 16;
            auto dstK1         = colAddrI + (sz * 2) * 16 + 8;
            *((int64_t*)dstK0) = *((int64_t*)inputZ0);
            *((int64_t*)dstK1) = *((int64_t*)inputZ1);
        }
        if (icDiv16R > 0) {
            auto inputZ0           = inputK + srcZStep * icDiv16 * 2;
            auto dstK0         = colAddrI + (icDiv16 * 2) * 16;
            auto dstK1         = colAddrI + (icDiv16 * 2) * 16 + 8;
            *((int64_t*)dstK0) = *((int64_t*)inputZ0);
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
    constexpr int dstXStepInt32 = 16 * 2 / sizeof(int64_t);
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
        
        auto inputOffset = inputOrigin + (sy + sfy * dilateY) * srcYStep + (sx + sfx * dilateX) * 8;
        auto indexOffset = sfy * kw + sfx;
        for (int fy = 0; fy < fyC; ++fy) {
            for (int fx = 0; fx < fxC; ++fx) {
                auto inputK       = inputOffset + fy * dilateY * srcYStep + fx * dilateX * 8;
                auto indexStart   = indexOffset + fy * kw + fx;
                auto indexInside  = indexStart % 2;
                auto indexOutside = indexStart / 2;
                auto dstK0        = (int64_t*)colAddrI + indexOutside * dstXStepInt32 + indexInside;
                dstK0[0]          = *((int64_t*)inputK);
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
    constexpr int dstXStepInt32 = 16 * 2 / sizeof(int64_t);
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
        
        auto inputOffset = inputOrigin + (sy + sfy * dilateY) * srcYStep + (sx + sfx * dilateX) * 8;
        auto indexOffset = (sfy * kw + sfx) * icDiv4;
        for (int fy = 0; fy < fyC; ++fy) {
            for (int fx = 0; fx < fxC; ++fx) {
                auto inputK     = inputOffset + fy * dilateY * srcYStep + fx * dilateX * 8;
                auto indexStart = indexOffset + (fy * kw + fx) * icDiv4;
                for (int sz = 0; sz < icDiv4; ++sz) {
                    const int yIndex      = indexStart + sz;
                    const int ySubOutside = yIndex / 2;
                    const int ySubInside  = yIndex % 2;
                    auto dstK0            = (int64_t*)colAddrI + ySubOutside * dstXStepInt32 + ySubInside;
                    dstK0[0]              = *((int64_t*)inputK);
                    inputK += srcZStep;
                }
            }
        }
    }
}

static MNN::CoreInt8Functions::Im2ColFunc chooseIm2Col(const MNN::ConvolutionCommon::Im2ColParameter* im2colParam, size_t inputChannel) {
    bool fastIm2Col = im2colParam->kernelX == 1 && im2colParam->kernelY == 1 &&
                      im2colParam->strideX == 1 && im2colParam->strideY == 1 && im2colParam->padX == 0 &&
                      im2colParam->padY == 0;
    int ih = im2colParam->ih, iw = im2colParam->iw;
    fastIm2Col &= im2colParam->srcYStep == iw * 8;
    if (fastIm2Col) {
        return _fastIm2Col;
    } else if (inputChannel <= 8) {
        return _im2colCommonZ1;
    } else {
        return _im2colCommon;
    }
}

static void _AVX2_MNNGetGemmUnit(int* UNIT, int* SRC_UNIT, int* DST_XUNIT) {
    *UNIT = 8;
    *SRC_UNIT = 16;
    *DST_XUNIT = 2;
}

void _AVX_MNNInt8FunctionInit(void* functions) {
    auto gAVX2CoreInt8Functions = (MNN::CoreInt8Functions*)functions;
    // MatMul
    gAVX2CoreInt8Functions->Int8GemmKernel = _AVX_MNNGemmInt8AddBiasScale_16x4_Unit;
    gAVX2CoreInt8Functions->Int8GemmKernelFast = _AVX_MNNGemmInt8AddBiasScale_16x4_Unit_Fast;
    gAVX2CoreInt8Functions->MNNGetGemmUnit = _AVX2_MNNGetGemmUnit;
    // Im2Col
    gAVX2CoreInt8Functions->chooseIm2Col = chooseIm2Col;
    // Int8 <-> Float
    gAVX2CoreInt8Functions->MNNFloat2Int8 = _AVX_MNNFloat2Int8;
    gAVX2CoreInt8Functions->MNNInt8ScaleToFloat = _AVX_MNNInt8ScaleToFloat;

    // conv depthwise
    gAVX2CoreInt8Functions->ConvDepthwiseLineInt8 = _AVX_MNNLineDepthWiseInt8AddBiasScaleUnit;
}
