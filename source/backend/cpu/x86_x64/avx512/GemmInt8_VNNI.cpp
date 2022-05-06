//
//  GemmInt8_VNNI.cpp
//  MNN
//
//  Created by MNN on 2021/09/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_AVX512_VNNI

#include "FunctionSummary.hpp"
#define PACK_UNIT 16
namespace {
static inline __m128i mm_loadu_si128(const void* addr) {
    return _mm_loadu_si128((__m128i const*)addr);
}
}  // namespace

#define _MM256_SET_M128I(__H, __L) _mm256_insertf128_si256(_mm256_castsi128_si256(__L), __H, 1) // for compile compatiable

// GemmInt8 with VNNI
void _AVX512_MNNGemmInt8AddBiasScale_16x4_Unit_VNNI(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, const QuanPostTreatParameters* post, size_t realDst) {
    const auto dst_step_tmp = dst_step / sizeof(int8_t);
    auto zero512 = _mm512_set1_ps(0.0f);
    auto minValue = _mm512_set1_ps(post->minValue);
    auto maxValue = _mm512_set1_ps(post->maxValue);
    auto plus = _mm512_set1_ps(0.5f);
    auto minus = _mm512_set1_ps(-0.5f);
    auto offset = _mm256_set1_epi16(128);
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

            auto _d00 = _MM256_SET_M128I(d10, d00);
            auto _d01 = _MM256_SET_M128I(d11, d01);
            auto _d02 = _MM256_SET_M128I(d12, d02);
            auto _d03 = _MM256_SET_M128I(d13, d03);
            auto _d0  = _mm256_hadd_epi32(_mm256_hadd_epi32(_d00, _d01),
                                          _mm256_hadd_epi32(_d02, _d03));

            auto _d10 = _MM256_SET_M128I(d30, d20);
            auto _d11 = _MM256_SET_M128I(d31, d21);
            auto _d12 = _MM256_SET_M128I(d32, d22);
            auto _d13 = _MM256_SET_M128I(d33, d23);
            auto _d1  = _mm256_hadd_epi32(_mm256_hadd_epi32(_d10, _d11),
                                          _mm256_hadd_epi32(_d12, _d13));

            auto _d20 = _MM256_SET_M128I(d50, d40);
            auto _d21 = _MM256_SET_M128I(d51, d41);
            auto _d22 = _MM256_SET_M128I(d52, d42);
            auto _d23 = _MM256_SET_M128I(d53, d43);
            auto _d2  = _mm256_hadd_epi32(_mm256_hadd_epi32(_d20, _d21),
                                          _mm256_hadd_epi32(_d22, _d23));

            auto _d30 = _MM256_SET_M128I(d70, d60);
            auto _d31 = _MM256_SET_M128I(d71, d61);
            auto _d32 = _MM256_SET_M128I(d72, d62);
            auto _d33 = _MM256_SET_M128I(d73, d63);
            auto _d3  = _mm256_hadd_epi32(_mm256_hadd_epi32(_d30, _d31),
                                          _mm256_hadd_epi32(_d32, _d33));

            auto d0 = _mm512_castsi256_si512(_d0);
            d0 = _mm512_inserti32x8(d0, _d1, 1);
            auto d1 = _mm512_castsi256_si512(_d2);
            d1 = _mm512_inserti32x8(d1, _d3, 1);
            if (post->scale != nullptr) {
                auto biasValue = _mm512_loadu_si512(bias_dz);
                d0 = _mm512_add_epi32(d0, biasValue);
                d1 = _mm512_add_epi32(d1, biasValue);
                auto scaleValue = _mm512_loadu_ps(scale_dz);
                auto f0 = _mm512_cvtepi32_ps(d0);
                auto f1 = _mm512_cvtepi32_ps(d1);
                f0 = _mm512_mul_ps(f0, scaleValue);
                f1 = _mm512_mul_ps(f1, scaleValue);
                f0 = _mm512_min_ps(f0, maxValue);
                f1 = _mm512_min_ps(f1, maxValue);
                f0 = _mm512_max_ps(f0, minValue);
                f1 = _mm512_max_ps(f1, minValue);
                auto m0 = _mm512_cmp_ps_mask(f0, zero512, 1);
                auto m1 = _mm512_cmp_ps_mask(f1, zero512, 1);
                auto b0 = _mm512_mask_blend_ps(m0, plus, minus);
                auto b1 = _mm512_mask_blend_ps(m1, plus, minus);
                f0 = _mm512_add_ps(f0, b0);
                f1 = _mm512_add_ps(f1, b1);
                // 3: _MM_FROUND_TO_ZERO
                d0 = _mm512_cvtps_epi32(_mm512_roundscale_ps(f0, 3));
                d1 = _mm512_cvtps_epi32(_mm512_roundscale_ps(f1, 3));
                // Int32 -> Int8
                auto hd0 = _mm512_cvtsepi32_epi16(d0);
                auto hd1 = _mm512_cvtsepi32_epi16(d1);
                hd0 = _mm256_add_epi16(hd0, offset);
                hd1 = _mm256_add_epi16(hd1, offset);
                auto h0 = _mm256_extracti128_si256(hd0, 0);
                auto h1 = _mm256_extracti128_si256(hd0, 1);
                auto h2 = _mm256_extracti128_si256(hd1, 0);
                auto h3 = _mm256_extracti128_si256(hd1, 1);
                h0 = _mm_packus_epi16(h0, h1);
                h1 = _mm_packus_epi16(h2, h3);

                _mm_storeu_si128((__m128i*)dst_x, h0);
                _mm_storeu_si128((__m128i*)dst_x + 1, h1);
            } else {
                auto biasValue = _mm512_loadu_si512(bias_dz);
                d0 = _mm512_add_epi32(d0, biasValue);
                d1 = _mm512_add_epi32(d1, biasValue);
                auto scaleValue = _mm512_loadu_ps(scale_dz);
                auto f0 = _mm512_cvtepi32_ps(d0);
                auto f1 = _mm512_cvtepi32_ps(d1);
                _mm512_storeu_ps(((float*)dst_x), f0);
                _mm512_storeu_ps(((float*)dst_x) + 16, f1);
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

        auto _d00 = _MM256_SET_M128I(d10, d00);
        auto _d01 = _MM256_SET_M128I(d11, d01);
        auto _d02 = _MM256_SET_M128I(d12, d02);
        auto _d03 = _MM256_SET_M128I(d13, d03);
        auto _d0  = _mm256_hadd_epi32(_mm256_hadd_epi32(_d00, _d01),
                                      _mm256_hadd_epi32(_d02, _d03));

        auto _d10 = _MM256_SET_M128I(d30, d20);
        auto _d11 = _MM256_SET_M128I(d31, d21);
        auto _d12 = _MM256_SET_M128I(d32, d22);
        auto _d13 = _MM256_SET_M128I(d33, d23);
        auto _d1  = _mm256_hadd_epi32(_mm256_hadd_epi32(_d10, _d11),
                                      _mm256_hadd_epi32(_d12, _d13));

        auto d0 = _mm512_castsi256_si512(_d0);
        d0 = _mm512_inserti32x8(d0, _d1, 1);
        if (post->scale != nullptr) {
            auto biasValue = _mm512_loadu_si512(bias_dz);
            d0 = _mm512_add_epi32(d0, biasValue);
            auto scaleValue = _mm512_loadu_ps(scale_dz);
            auto f0 = _mm512_cvtepi32_ps(d0);
            f0 = _mm512_mul_ps(f0, scaleValue);
            f0 = _mm512_min_ps(f0, maxValue);
            f0 = _mm512_max_ps(f0, minValue);
            auto m0 = _mm512_cmp_ps_mask(f0, zero512, 1);
            auto b0 = _mm512_mask_blend_ps(m0, plus, minus);
            f0 = _mm512_add_ps(f0, b0);
            // 3: _MM_FROUND_TO_ZERO
            d0 = _mm512_cvtps_epi32(_mm512_roundscale_ps(f0, 3));
            // Int32 -> Int8
            auto hd0 = _mm512_cvtsepi32_epi16(d0);
            hd0 = _mm256_add_epi16(hd0, offset);
            auto h0 = _mm256_extracti128_si256(hd0, 0);
            auto h1 = _mm256_extracti128_si256(hd0, 1);
            h0 = _mm_packus_epi16(h0, h1);

            _mm_storeu_si128((__m128i*)dst_x, h0);
        } else {
            auto biasValue = _mm512_loadu_si512(bias_dz);
            d0 = _mm512_add_epi32(d0, biasValue);
            auto scaleValue = _mm512_loadu_ps(scale_dz);
            auto f0 = _mm512_cvtepi32_ps(d0);
            _mm512_storeu_ps(((float*)dst_x), f0);
        }
    }
}

void _AVX512_MNNLineDepthWiseInt8AddBiasScaleUnit_VNNI(int8_t* dstO, const int8_t* srcO, const int8_t* weightO, const QuanPostTreatParameters* parameters, size_t width, size_t src_w_step, size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step) {
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
#endif

#undef _MM256_SET_M128I
