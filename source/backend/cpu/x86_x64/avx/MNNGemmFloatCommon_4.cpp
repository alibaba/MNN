//
//  MNNGemmFloatCommon_4.cpp
//  MNN
//
//  Created by MNN on 2019/08/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <immintrin.h>
#include <xmmintrin.h>
#include <smmintrin.h>
#include <stdint.h>
#include "backend/cpu/compute/Int8FunctionsOpt.h"
#include <cmath>
#include <algorithm>
#ifdef MNN_OPTIMIZE_INT8_SSE
void _AVX_MNNGemmInt8AddBiasScale_16x4_Unit(int8_t* dst, const int8_t* src, const int8_t* weight, const int32_t* bias,
                                       const float* scale, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad) {
    const auto dst_step_tmp = dst_step / sizeof(int8_t);
    auto zero = _mm_set1_epi8(0);
    int32_t dstTemp[4];
    auto maxV = _mm_set1_ps(127.0f);
    auto minV = _mm_set1_ps(-127.0f);

    for (int dz = 0; dz < dst_depth_quad; ++dz) {
        const auto weight_dz = weight + dz * src_depth_quad * (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT);
        const auto bias_dz   = bias + dz * GEMM_INT8_UNIT;
        const auto scale_dz  = scale + dz * GEMM_INT8_UNIT;
        auto dst_z           = dst + dz * dst_step_tmp;
        auto biasV = _mm_cvtepi32_ps(*(__m128i *)(bias_dz));
        auto scaleV = _mm_loadu_ps(scale_dz);
        for (int w = 0; w < GEMM_INT8_DST_XUNIT; ++w) {
            const auto src_x   = src + w * GEMM_INT8_SRC_UNIT;
            auto dst_x         = dst_z + w * GEMM_INT8_UNIT;
            auto dst0 = _mm_set1_epi32(0);
            auto dst1 = _mm_set1_epi32(0);
            auto dst2 = _mm_set1_epi32(0);
            auto dst3 = _mm_set1_epi32(0);

            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT) * sz;
                const auto src_z     = src_x + sz * GEMM_INT8_DST_XUNIT * GEMM_INT8_SRC_UNIT;
                auto src = *(__m128i *)(src_z);
                auto weight0 = *(__m128i *)(weight_sz + 16 * 0);
                auto weight1 = *(__m128i *)(weight_sz + 16 * 1);
                auto weight2 = *(__m128i *)(weight_sz + 16 * 2);
                auto weight3 = *(__m128i *)(weight_sz + 16 * 3);
                auto s0 = _mm_cvtepi8_epi16(src);
                auto s1 = _mm_cvtepi8_epi16(_mm_unpackhi_epi64(src, zero));

                auto w00 = _mm_cvtepi8_epi16(weight0);
                auto w01 = _mm_cvtepi8_epi16(_mm_unpackhi_epi64(weight0, zero));
                auto w10 = _mm_cvtepi8_epi16(weight1);
                auto w11 = _mm_cvtepi8_epi16(_mm_unpackhi_epi64(weight1, zero));
                auto w20 = _mm_cvtepi8_epi16(weight2);
                auto w21 = _mm_cvtepi8_epi16(_mm_unpackhi_epi64(weight2, zero));
                auto w30 = _mm_cvtepi8_epi16(weight3);
                auto w31 = _mm_cvtepi8_epi16(_mm_unpackhi_epi64(weight3, zero));
                
                auto d00 = _mm_mullo_epi16(w00, s0);
                auto d01 = _mm_mullo_epi16(w01, s1);
                auto d10 = _mm_mullo_epi16(w10, s0);
                auto d11 = _mm_mullo_epi16(w11, s1);
                auto d20 = _mm_mullo_epi16(w20, s0);
                auto d21 = _mm_mullo_epi16(w21, s1);
                auto d30 = _mm_mullo_epi16(w30, s0);
                auto d31 = _mm_mullo_epi16(w31, s1);
                auto d0 = _mm_add_epi16(d00, d01);
                auto d1 = _mm_add_epi16(d10, d11);
                auto d2 = _mm_add_epi16(d20, d21);
                auto d3 = _mm_add_epi16(d30, d31);

                auto d0Lo = _mm_cvtepi16_epi32(d0);
                auto d0Hi = _mm_cvtepi16_epi32(_mm_unpackhi_epi64(d0, zero));
                auto d1Lo = _mm_cvtepi16_epi32(d1);
                auto d1Hi = _mm_cvtepi16_epi32(_mm_unpackhi_epi64(d1, zero));
                auto d2Lo = _mm_cvtepi16_epi32(d2);
                auto d2Hi = _mm_cvtepi16_epi32(_mm_unpackhi_epi64(d2, zero));
                auto d3Lo = _mm_cvtepi16_epi32(d3);
                auto d3Hi = _mm_cvtepi16_epi32(_mm_unpackhi_epi64(d3, zero));

                dst0 = _mm_add_epi32(dst0, d0Lo);
                dst0 = _mm_add_epi32(dst0, d0Hi);
                dst1 = _mm_add_epi32(dst1, d1Lo);
                dst1 = _mm_add_epi32(dst1, d1Hi);
                dst2 = _mm_add_epi32(dst2, d2Lo);
                dst2 = _mm_add_epi32(dst2, d2Hi);
                dst3 = _mm_add_epi32(dst3, d3Lo);
                dst3 = _mm_add_epi32(dst3, d3Hi);
            }

            _MM_TRANSPOSE4_PS(dst0, dst1, dst2, dst3);

            auto summer = _mm_add_epi32(dst0, _mm_add_epi32(dst1, _mm_add_epi32(dst2, dst3)));
            auto summerFloat = _mm_add_ps(_mm_cvtepi32_ps(summer), biasV) * scaleV;
            summerFloat = _mm_round_ps(_mm_min_ps(_mm_max_ps(summerFloat, minV), maxV), _MM_FROUND_TO_NEAREST_INT);
            summer = _mm_cvtps_epi32(summerFloat);
            *(__m128i*)(dstTemp + 4 * 0) =  summer;
            for (int j = 0; j < 4; ++j) {
                dst_x[j] = dstTemp[j];
            }
        }
    }
}
#endif
void _AVX_MNNGemmFloatCommon_4(float* dst, const float* src, const float* weight, size_t src_depth_quad, size_t dst_step,
                          size_t dst_depth_quad, size_t width, size_t weight_depth_offset) {
    auto src_depth_step = 4 * width;
    int wC8             = width / 8;
    int w8End           = wC8 * 8;
    for (int dz = 0; dz < dst_depth_quad; ++dz) {
        float* dst_z   = dst + dz * dst_step;
        auto weight_dz = weight + dz * (src_depth_quad * 16 + weight_depth_offset);

        for (int dx = 0; dx < wC8; ++dx) {
            float* dst_x        = dst_z + dx * 8 * 4;
            auto dst0           = _mm256_set1_ps(0.0f);
            auto dst1           = _mm256_set1_ps(0.0f);
            auto dst2           = _mm256_set1_ps(0.0f);
            auto dst3           = _mm256_set1_ps(0.0f);
            const float* src_dx = src + 8 * dx * 4;
            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const float* src_z    = src_dx + sz * src_depth_step;
                const float* weight_z = weight_dz + sz * 16;
                auto w0               = _mm256_broadcast_ps((const __m128 *)(weight_z + 4 * 0));
                auto w1               = _mm256_broadcast_ps((const __m128 *)(weight_z + 4 * 1));
                auto w2               = _mm256_broadcast_ps((const __m128 *)(weight_z + 4 * 2));
                auto w3               = _mm256_broadcast_ps((const __m128 *)(weight_z + 4 * 3));
#define AVX_COMPUTE(v)                                   \
    {                                                \
        auto srcValue = _mm256_loadu_ps(src_z + 8 * v); \
        auto s0       = _mm256_shuffle_ps(srcValue, srcValue, _MM_SHUFFLE(0, 0, 0, 0)); \
        auto s1       = _mm256_shuffle_ps(srcValue, srcValue, _MM_SHUFFLE(1, 1, 1, 1)); \
        auto s2       = _mm256_shuffle_ps(srcValue, srcValue, _MM_SHUFFLE(2, 2, 2, 2)); \
        auto s3       = _mm256_shuffle_ps(srcValue, srcValue, _MM_SHUFFLE(3, 3, 3, 3)); \
        auto sw0      = _mm256_mul_ps(s0, w0);          \
        auto sw1      = _mm256_mul_ps(s1, w1);          \
        auto sw2      = _mm256_mul_ps(s2, w2);          \
        auto sw3      = _mm256_mul_ps(s3, w3);          \
        dst##v        = _mm256_add_ps(dst##v, sw0);     \
        dst##v        = _mm256_add_ps(dst##v, sw1);     \
        dst##v        = _mm256_add_ps(dst##v, sw2);     \
        dst##v        = _mm256_add_ps(dst##v, sw3);     \
    }

                AVX_COMPUTE(0);
                AVX_COMPUTE(1);
                AVX_COMPUTE(2);
                AVX_COMPUTE(3);
            }

            _mm256_storeu_ps(dst_x + 8 * 0, dst0);
            _mm256_storeu_ps(dst_x + 8 * 1, dst1);
            _mm256_storeu_ps(dst_x + 8 * 2, dst2);
            _mm256_storeu_ps(dst_x + 8 * 3, dst3);
        }
        _mm256_zeroall();

        for (int dx = w8End; dx < width; ++dx) {
            float* dst_x  = dst_z + dx * 4;
            auto dstValue = _mm_set1_ps(0.0f);

            const float* src_dx = src + 4 * dx;
            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const float* src_z    = src_dx + sz * src_depth_step;
                const float* weight_z = weight_dz + sz * 16;
                auto w0               = _mm_loadu_ps(weight_z + 4 * 0);
                auto w1               = _mm_loadu_ps(weight_z + 4 * 1);
                auto w2               = _mm_loadu_ps(weight_z + 4 * 2);
                auto w3               = _mm_loadu_ps(weight_z + 4 * 3);

                auto s0       = _mm_set1_ps(src_z[0]);
                auto s1       = _mm_set1_ps(src_z[1]);
                auto s2       = _mm_set1_ps(src_z[2]);
                auto s3       = _mm_set1_ps(src_z[3]);

                auto sw0 = _mm_mul_ps(s0, w0);
                auto sw1 = _mm_mul_ps(s1, w1);
                auto sw2 = _mm_mul_ps(s2, w2);
                auto sw3 = _mm_mul_ps(s3, w3);
                dstValue = _mm_add_ps(dstValue, sw0);
                dstValue = _mm_add_ps(dstValue, sw1);
                dstValue = _mm_add_ps(dstValue, sw2);
                dstValue = _mm_add_ps(dstValue, sw3);
            }
            _mm_storeu_ps(dst_x, dstValue);
        }
    }
}
