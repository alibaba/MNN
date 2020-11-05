//
//  MNNGemmFloatCommon_4.cpp
//  MNN
//
//  Created by MNN on 2019/08/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <algorithm>
#include <cmath>
#include "FunctionSummary.hpp"
#include "backend/cpu/compute/Int8FunctionsOpt.h"

#ifndef _MM_TRANSPOSE4_PS
#define _MM_TRANSPOSE4_PS(row0, row1, row2, row3) \
    do {                                          \
        __m128 tmp3, tmp2, tmp1, tmp0;            \
        tmp0   = _mm_unpacklo_ps((row0), (row1)); \
        tmp2   = _mm_unpacklo_ps((row2), (row3)); \
        tmp1   = _mm_unpackhi_ps((row0), (row1)); \
        tmp3   = _mm_unpackhi_ps((row2), (row3)); \
        (row0) = _mm_movelh_ps(tmp0, tmp2);       \
        (row1) = _mm_movehl_ps(tmp2, tmp0);       \
        (row2) = _mm_movelh_ps(tmp1, tmp3);       \
        (row3) = _mm_movehl_ps(tmp3, tmp1);       \
    } while (0)
#endif

#ifdef MNN_VEC_PRINT
#include <MNN/MNNDefine.h>
static void _dump(__m256 v0) {
    float fv0[8];
    _mm256_store_ps(fv0, v0);
    for (int i = 0; i < 8; ++i) {
        MNN_PRINT("%f, ", fv0[i]);
    }
    MNN_PRINT("\n");
}
#endif
static __m256 _merge(__m256 v0, __m256 v1, __m256 v2, __m256 v3) {
    auto h0  = _mm256_hadd_ps(v0, v1);
    auto h1  = _mm256_hadd_ps(v2, v3);
    auto res = _mm256_hadd_ps(h0, h1);
    return res;
}

static __m128 merge128(__m128 d0, __m128 d1, __m128 d2, __m128 d3) {
    auto d00 = _mm_hadd_ps(d0, d1);
    auto d01 = _mm_hadd_ps(d2, d3);
    return _mm_hadd_ps(d00, d01);
}

void _AVX_MNNGemmFloatCommon_4(float* dst, const float* src, const float* weight, size_t src_depth_quad,
                               size_t dst_step, size_t dst_depth_quad, size_t width, size_t weight_depth_offset) {
    auto src_depth_step = 4 * width;
    const int unit      = 4;
    int wUnit           = width / unit;
    auto wUnitEnd       = wUnit * unit;
    for (int dz = 0; dz < dst_depth_quad; ++dz) {
        float* dst_z   = dst + dz * dst_step;
        auto weight_dz = weight + dz * (src_depth_quad * 16 + weight_depth_offset);

        for (int dx = 0; dx < wUnit; ++dx) {
            float* dst_x        = dst_z + dx * 4 * unit;
            const float* src_dx = src + dx * 4 * unit;

            auto is0 = _mm256_loadu_ps(src_dx + 8 * 0);
            auto is1 = _mm256_loadu_ps(src_dx + 8 * 1);

            auto iw0 = _mm256_broadcast_ps((const __m128*)(weight_dz + 4 * 0));
            auto iw1 = _mm256_broadcast_ps((const __m128*)(weight_dz + 4 * 1));
            auto iw2 = _mm256_broadcast_ps((const __m128*)(weight_dz + 4 * 2));
            auto iw3 = _mm256_broadcast_ps((const __m128*)(weight_dz + 4 * 3));

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
                const float* src_z = src_dx + sz * src_depth_step;
                auto s0            = _mm256_loadu_ps(src_z + 8 * 0);
                auto s1            = _mm256_loadu_ps(src_z + 8 * 1);

                const float* weight_z = weight_dz + sz * 16;
                auto w0               = _mm256_broadcast_ps((const __m128*)(weight_z + 4 * 0));
                auto w1               = _mm256_broadcast_ps((const __m128*)(weight_z + 4 * 1));
                auto w2               = _mm256_broadcast_ps((const __m128*)(weight_z + 4 * 2));
                auto w3               = _mm256_broadcast_ps((const __m128*)(weight_z + 4 * 3));
#define COMPUTE(i, j) d##i##j = _mm256_add_ps(_mm256_mul_ps(s##i, w##j), d##i##j)
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
            float* dst_x = dst_z + dx * 4;
            auto d0      = _mm_set1_ps(0.0f);
            auto d1      = _mm_set1_ps(0.0f);
            auto d2      = _mm_set1_ps(0.0f);
            auto d3      = _mm_set1_ps(0.0f);

            const float* src_dx = src + 4 * dx;
            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const float* src_z    = src_dx + sz * src_depth_step;
                const float* weight_z = weight_dz + sz * 16;
                auto w0               = _mm_loadu_ps(weight_z + 4 * 0);
                auto w1               = _mm_loadu_ps(weight_z + 4 * 1);
                auto w2               = _mm_loadu_ps(weight_z + 4 * 2);
                auto w3               = _mm_loadu_ps(weight_z + 4 * 3);
                auto s                = _mm_loadu_ps(src_z);
#define COMPUTE(i) d##i = _mm_add_ps(_mm_mul_ps(s, w##i), d##i)
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

void _AVX_MNNGemmFloatCommonFMA_4(float* dst, const float* src, const float* weight, size_t src_depth_quad,
                                  size_t dst_step, size_t dst_depth_quad, size_t width, size_t weight_depth_offset) {
    auto src_depth_step = 4 * width;
    const int unit      = 4;
    int wUnit           = width / unit;
    auto wUnitEnd       = wUnit * unit;
    for (int dz = 0; dz < dst_depth_quad; ++dz) {
        float* dst_z   = dst + dz * dst_step;
        auto weight_dz = weight + dz * (src_depth_quad * 16 + weight_depth_offset);

        for (int dx = 0; dx < wUnit; ++dx) {
            float* dst_x        = dst_z + dx * 4 * unit;
            const float* src_dx = src + dx * 4 * unit;

            auto is0 = _mm256_loadu_ps(src_dx + 8 * 0);
            auto is1 = _mm256_loadu_ps(src_dx + 8 * 1);

            auto iw0 = _mm256_broadcast_ps((const __m128*)(weight_dz + 4 * 0));
            auto iw1 = _mm256_broadcast_ps((const __m128*)(weight_dz + 4 * 1));
            auto iw2 = _mm256_broadcast_ps((const __m128*)(weight_dz + 4 * 2));
            auto iw3 = _mm256_broadcast_ps((const __m128*)(weight_dz + 4 * 3));

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
                const float* src_z = src_dx + sz * src_depth_step;
                auto s0            = _mm256_loadu_ps(src_z + 8 * 0);
                auto s1            = _mm256_loadu_ps(src_z + 8 * 1);

                const float* weight_z = weight_dz + sz * 16;
                auto w0               = _mm256_broadcast_ps((const __m128*)(weight_z + 4 * 0));
                auto w1               = _mm256_broadcast_ps((const __m128*)(weight_z + 4 * 1));
                auto w2               = _mm256_broadcast_ps((const __m128*)(weight_z + 4 * 2));
                auto w3               = _mm256_broadcast_ps((const __m128*)(weight_z + 4 * 3));
#define COMPUTE(i, j) d##i##j = _mm256_fmadd_ps(s##i, w##j, d##i##j)
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
            float* dst_x = dst_z + dx * 4;
            auto d0      = _mm_set1_ps(0.0f);
            auto d1      = _mm_set1_ps(0.0f);
            auto d2      = _mm_set1_ps(0.0f);
            auto d3      = _mm_set1_ps(0.0f);

            const float* src_dx = src + 4 * dx;
            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const float* src_z    = src_dx + sz * src_depth_step;
                const float* weight_z = weight_dz + sz * 16;
                auto w0               = _mm_loadu_ps(weight_z + 4 * 0);
                auto w1               = _mm_loadu_ps(weight_z + 4 * 1);
                auto w2               = _mm_loadu_ps(weight_z + 4 * 2);
                auto w3               = _mm_loadu_ps(weight_z + 4 * 3);
                auto s                = _mm_loadu_ps(src_z);
#define COMPUTE(i) d##i = _mm_fmadd_ps(s, w##i, d##i)
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
    return _AVX_MNNGemmFloatCommon_4(dst, src, weight, src_depth_quad, dst_step, dst_depth_quad, 8,
                                     weight_depth_offset);
}

void _AVX_MNNGemmFloatUnitFMA_4(float* dst, const float* src, const float* weight, size_t src_depth_quad,
                                size_t dst_step, size_t dst_depth_quad, size_t weight_depth_offset) {
    return _AVX_MNNGemmFloatCommonFMA_4(dst, src, weight, src_depth_quad, dst_step, dst_depth_quad, 8,
                                        weight_depth_offset);
}
