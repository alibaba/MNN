//
//  MNNGemmFloatCommon_4.cpp
//  MNN
//
//  Created by MNN on 2018/11/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_USE_SSE

#include <immintrin.h>
#include <stdint.h>
#include "CommonHelperSSE.hpp"
#include "ConvOpt.h"

TargetBegin("sse2")
static void _SSE_MNNGemmFloatCommon_4(float* dst, const float* src, const float* weight, size_t src_depth_quad, size_t dst_step,
                          size_t dst_depth_quad, size_t width, size_t weight_depth_offset) {
    auto src_depth_step = 4 * width;
    int wC4             = width / 4;
    int w4End           = wC4 * 4;
    for (int dz = 0; dz < dst_depth_quad; ++dz) {
        float* dst_z   = dst + dz * dst_step;
        auto weight_dz = weight + dz * (src_depth_quad * 16 + weight_depth_offset);

        for (int dx = 0; dx < wC4; ++dx) {
            float* dst_x        = dst_z + dx * 4 * 4;
            auto dst0           = _mm_set1_ps(0.0f);
            auto dst1           = _mm_set1_ps(0.0f);
            auto dst2           = _mm_set1_ps(0.0f);
            auto dst3           = _mm_set1_ps(0.0f);
            const float* src_dx = src + 4 * dx * 4;
            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const float* src_z    = src_dx + sz * src_depth_step;
                const float* weight_z = weight_dz + sz * 16;
                auto w0               = _mm_loadu_ps(weight_z + 4 * 0);
                auto w1               = _mm_loadu_ps(weight_z + 4 * 1);
                auto w2               = _mm_loadu_ps(weight_z + 4 * 2);
                auto w3               = _mm_loadu_ps(weight_z + 4 * 3);
#define SSE_COMPUTE(v)                                   \
    {                                                \
        const float* srcValue = src_z + 4 * v;       \
        auto s0       = _mm_set1_ps(srcValue[0]);    \
        auto s1       = _mm_set1_ps(srcValue[1]);    \
        auto s2       = _mm_set1_ps(srcValue[2]);    \
        auto s3       = _mm_set1_ps(srcValue[3]);    \
        auto sw0      = _mm_mul_ps(s0, w0);          \
        auto sw1      = _mm_mul_ps(s1, w1);          \
        auto sw2      = _mm_mul_ps(s2, w2);          \
        auto sw3      = _mm_mul_ps(s3, w3);          \
        dst##v        = _mm_add_ps(dst##v, sw0);     \
        dst##v        = _mm_add_ps(dst##v, sw1);     \
        dst##v        = _mm_add_ps(dst##v, sw2);     \
        dst##v        = _mm_add_ps(dst##v, sw3);     \
    }

                SSE_COMPUTE(0);
                SSE_COMPUTE(1);
                SSE_COMPUTE(2);
                SSE_COMPUTE(3);
            }

            _mm_store_ps(dst_x + 4 * 0, dst0);
            _mm_store_ps(dst_x + 4 * 1, dst1);
            _mm_store_ps(dst_x + 4 * 2, dst2);
            _mm_store_ps(dst_x + 4 * 3, dst3);
        }

        for (int dx = w4End; dx < width; ++dx) {
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
            _mm_store_ps(dst_x, dstValue);
        }
    }
}
TargetEnd("sse2")

TargetBegin("avx")
static void _AVX_MNNGemmFloatCommon_4(float* dst, const float* src, const float* weight, size_t src_depth_quad, size_t dst_step,
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
            _mm_store_ps(dst_x, dstValue);
        }
    }
}
TargetEnd("avx")

void MNNGemmFloatCommon_4(float* dst, const float* src, const float* weight, size_t src_depth_quad, size_t dst_step,
                          size_t dst_depth_quad, size_t width, size_t weight_depth_offset) {
    if (cpu_feature_available(AVX)) {
        _AVX_MNNGemmFloatCommon_4(dst, src, weight, src_depth_quad, dst_step, dst_depth_quad, width, weight_depth_offset);
    } else {
        _SSE_MNNGemmFloatCommon_4(dst, src, weight, src_depth_quad, dst_step, dst_depth_quad, width, weight_depth_offset);
    }
}
#endif
