//
//  MNNGemmFloatCommon_4.cpp
//  MNN
//
//  Created by MNN on 2019/08/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include <stdint.h>
#include "backend/cpu/compute/Int8FunctionsOpt.h"
#include <cmath>
#include <algorithm>

#ifndef _MM_TRANSPOSE4_PS
#define _MM_TRANSPOSE4_PS(row0, row1, row2, row3) \
do { \
  __m128 tmp3, tmp2, tmp1, tmp0; \
  tmp0 = _mm_unpacklo_ps((row0), (row1)); \
  tmp2 = _mm_unpacklo_ps((row2), (row3)); \
  tmp1 = _mm_unpackhi_ps((row0), (row1)); \
  tmp3 = _mm_unpackhi_ps((row2), (row3)); \
  (row0) = _mm_movelh_ps(tmp0, tmp2); \
  (row1) = _mm_movehl_ps(tmp2, tmp0); \
  (row2) = _mm_movelh_ps(tmp1, tmp3); \
  (row3) = _mm_movehl_ps(tmp3, tmp1); \
} while (0)
#endif

#ifdef MNN_OPTIMIZE_INT8_SSE
void _AVX_MNNGemmInt8AddBiasScale_16x4_Unit(int8_t* dst, const int8_t* src, const int8_t* weight, const int32_t* bias,
                                       const float* scale, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad) {
    const auto dst_step_tmp = dst_step / sizeof(int8_t);
    auto zero = _mm_set1_epi8(0);
    float dstTemp[4];
    auto maxV = _mm_set1_ps(127.0f);
    auto minV = _mm_set1_ps(-127.0f);

    for (int dz = 0; dz < dst_depth_quad; ++dz) {
        const auto weight_dz = weight + dz * src_depth_quad * (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT);
        const auto bias_dz   = bias + dz * GEMM_INT8_UNIT;
        const auto scale_dz  = scale + dz * GEMM_INT8_UNIT;
        auto dst_z           = dst + dz * dst_step_tmp;
        auto biasV = *(__m128i *)(bias_dz);
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
            auto summerFloat = _mm_cvtepi32_ps(_mm_add_epi32(summer, biasV)) * scaleV;
            summerFloat = _mm_min_ps(_mm_max_ps(summerFloat, minV), maxV);
            *(__m128*)(dstTemp + 4 * 0) =  summerFloat;
            for (int j = 0; j < 4; ++j) {
                dst_x[j] = static_cast<int8_t>(roundf(dstTemp[j]));
            }
        }
    }
}
#endif

#ifdef MNN_VEC_PRINT
#include <MNN/MNNDefine.h>
static void _dump(__m256 v0) {
    float fv0[8];
    _mm256_store_ps(fv0, v0);
    for (int i=0; i<8; ++i) {
        MNN_PRINT("%f, ", fv0[i]);
    }
    MNN_PRINT("\n");
}
#endif
static __m256 _merge(__m256 v0, __m256 v1, __m256 v2, __m256 v3) {
    auto h0 = _mm256_hadd_ps(v0, v1);
    auto h1 = _mm256_hadd_ps(v2, v3);
    auto res = _mm256_hadd_ps(h0, h1);
    return res;
}

static __m128 merge128(__m128 d0, __m128 d1, __m128 d2, __m128 d3) {
    auto d00 = _mm_hadd_ps(d0, d1);
    auto d01 = _mm_hadd_ps(d2, d3);
    return _mm_hadd_ps(d00, d01);
}

void _AVX_MNNGemmFloatCommon_4(float* dst, const float* src, const float* weight, size_t src_depth_quad, size_t dst_step,
                          size_t dst_depth_quad, size_t width, size_t weight_depth_offset) {
    auto src_depth_step = 4 * width;
    const int unit = 4;
    int wUnit             = width / unit;
    auto wUnitEnd = wUnit * unit;
    for (int dz = 0; dz < dst_depth_quad; ++dz) {
        float* dst_z   = dst + dz * dst_step;
        auto weight_dz = weight + dz * (src_depth_quad * 16 + weight_depth_offset);

        for (int dx = 0; dx < wUnit; ++dx) {
            float* dst_x        = dst_z + dx * 4 * unit;
            const float* src_dx = src + dx * 4 * unit;
            
            auto is0 = _mm256_loadu_ps(src_dx + 8 * 0);
            auto is1 = _mm256_loadu_ps(src_dx + 8 * 1);

            auto iw0 = _mm256_broadcast_ps((const __m128 *)(weight_dz + 4 * 0));
            auto iw1 = _mm256_broadcast_ps((const __m128 *)(weight_dz + 4 * 1));
            auto iw2 = _mm256_broadcast_ps((const __m128 *)(weight_dz + 4 * 2));
            auto iw3 = _mm256_broadcast_ps((const __m128 *)(weight_dz + 4 * 3));
            
#define MNN_INIT_VEC(i, j) auto d##i##j = _mm256_mul_ps(is##i, iw##j)
            MNN_INIT_VEC(0, 0);
            MNN_INIT_VEC(0, 1);
            MNN_INIT_VEC(0, 2);
            MNN_INIT_VEC(0, 3);
            MNN_INIT_VEC(1, 0);
            MNN_INIT_VEC(1, 1);
            MNN_INIT_VEC(1, 2);
            MNN_INIT_VEC(1, 3);
#undef MNN_INIT_VEC
            for (int sz = 1; sz < src_depth_quad; ++sz) {
                const float* src_z    = src_dx + sz * src_depth_step;
                auto s0 = _mm256_loadu_ps(src_z + 8 * 0);
                auto s1 = _mm256_loadu_ps(src_z + 8 * 1);

                const float* weight_z = weight_dz + sz * 16;
                auto w0 = _mm256_broadcast_ps((const __m128 *)(weight_z + 4 * 0));
                auto w1 = _mm256_broadcast_ps((const __m128 *)(weight_z + 4 * 1));
                auto w2 = _mm256_broadcast_ps((const __m128 *)(weight_z + 4 * 2));
                auto w3 = _mm256_broadcast_ps((const __m128 *)(weight_z + 4 * 3));
#ifdef MNN_FMA_ENABLE
#define COMPUTE(i,j) d##i##j = _mm256_fmadd_ps(s##i, w##j, d##i##j)
#else
#define COMPUTE(i,j) d##i##j = _mm256_add_ps(_mm256_mul_ps(s##i, w##j), d##i##j)
#endif
                COMPUTE(0, 0);
                COMPUTE(0, 1);
                COMPUTE(0, 2);
                COMPUTE(0, 3);
                
                COMPUTE(1, 0);
                COMPUTE(1, 1);
                COMPUTE(1, 2);
                COMPUTE(1, 3);
                
#undef COMPUTE
            }

            _mm256_storeu_ps(dst_x + 8 * 0, _merge(d00, d01, d02, d03));
            _mm256_storeu_ps(dst_x + 8 * 1, _merge(d10, d11, d12, d13));
        }
        for (int dx = wUnitEnd; dx < width; ++dx) {
            float* dst_x  = dst_z + dx * 4;
            auto d0 = _mm_set1_ps(0.0f);
            auto d1 = _mm_set1_ps(0.0f);
            auto d2 = _mm_set1_ps(0.0f);
            auto d3 = _mm_set1_ps(0.0f);

            const float* src_dx = src + 4 * dx;
            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const float* src_z    = src_dx + sz * src_depth_step;
                const float* weight_z = weight_dz + sz * 16;
                auto w0               = _mm_loadu_ps(weight_z + 4 * 0);
                auto w1               = _mm_loadu_ps(weight_z + 4 * 1);
                auto w2               = _mm_loadu_ps(weight_z + 4 * 2);
                auto w3               = _mm_loadu_ps(weight_z + 4 * 3);
                auto s = _mm_loadu_ps(src_z);
#ifdef MNN_FMA_ENABLE
#define COMPUTE(i) d##i = _mm_fmadd_ps(s, w##i, d##i)
#else
#define COMPUTE(i) d##i = _mm_add_ps(_mm_mul_ps(s, w##i), d##i)
#endif
                COMPUTE(0);
                COMPUTE(1);
                COMPUTE(2);
                COMPUTE(3);
#undef COMPUTE
            }
            _mm_storeu_ps(dst_x, merge128(d0, d1, d2, d3));
        }
    }
}

void _AVX_MNNGemmFloatUnit_4(float* dst, const float* src, const float* weight, size_t src_depth_quad, size_t dst_step,
                             size_t dst_depth_quad, size_t weight_depth_offset) {
    return _AVX_MNNGemmFloatCommon_4(dst, src, weight, src_depth_quad, dst_step, dst_depth_quad, 8, weight_depth_offset);
}
