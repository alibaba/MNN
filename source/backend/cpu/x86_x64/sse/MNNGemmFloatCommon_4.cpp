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

static __m128 merge(__m128 d0, __m128 d1, __m128 d2, __m128 d3) {
    auto d00 = _mm_hadd_ps(d0, d1);
    auto d01 = _mm_hadd_ps(d2, d3);
    return _mm_hadd_ps(d00, d01);
}

#ifdef MNN_FMA_ENABLE
#define COMPUTE(i) { \
d0##i = _mm_fmadd_ps(w##i, s0, d0##i);\
d1##i = _mm_fmadd_ps(w##i, s1, d1##i);\
d2##i = _mm_fmadd_ps(w##i, s2, d2##i);\
d3##i = _mm_fmadd_ps(w##i, s3, d3##i);\
}
#else
#define COMPUTE(i) { \
d0##i = _mm_add_ps(_mm_mul_ps(w##i, s0), d0##i);\
d1##i = _mm_add_ps(_mm_mul_ps(w##i, s1), d1##i);\
d2##i = _mm_add_ps(_mm_mul_ps(w##i, s2), d2##i);\
d3##i = _mm_add_ps(_mm_mul_ps(w##i, s3), d3##i);\
}
#endif
#define STORE(i) _mm_storeu_ps(dst_x + 4 * i, merge(d##i##0, d##i##1, d##i##2, d##i##3));

void _SSE_MNNGemmFloatCommon_4(float* dst, const float* src, const float* weight, size_t src_depth_quad, size_t dst_step,
                          size_t dst_depth_quad, size_t width, size_t weight_depth_offset) {
    auto src_depth_step = 4 * width;
    int wC4             = width / 4;
    int w4End           = wC4 * 4;
    for (int dz = 0; dz < dst_depth_quad; ++dz) {
        float* dst_z   = dst + dz * dst_step;
        auto weight_dz = weight + dz * (src_depth_quad * 16 + weight_depth_offset);
        for (int dx = 0; dx< wC4; ++dx) {
            float* dst_x  = dst_z + dx * 16;
            const float* src_dx = src + 16 * dx;
            auto iw0               = _mm_loadu_ps(weight_dz + 4 * 0);
            auto iw1               = _mm_loadu_ps(weight_dz + 4 * 1);
            auto iw2               = _mm_loadu_ps(weight_dz + 4 * 2);
            auto iw3               = _mm_loadu_ps(weight_dz + 4 * 3);
            auto is0 = _mm_loadu_ps(src_dx + 4*0);
            auto is1 = _mm_loadu_ps(src_dx + 4*1);
            auto is2 = _mm_loadu_ps(src_dx + 4*2);
            auto is3 = _mm_loadu_ps(src_dx + 4*3);

            auto d00 = _mm_mul_ps(is0, iw0);
            auto d01 = _mm_mul_ps(is0, iw1);
            auto d02 = _mm_mul_ps(is0, iw2);
            auto d03 = _mm_mul_ps(is0, iw3);

            auto d10 = _mm_mul_ps(is1, iw0);
            auto d11 = _mm_mul_ps(is1, iw1);
            auto d12 = _mm_mul_ps(is1, iw2);
            auto d13 = _mm_mul_ps(is1, iw3);

            auto d20 = _mm_mul_ps(is2, iw0);
            auto d21 = _mm_mul_ps(is2, iw1);
            auto d22 = _mm_mul_ps(is2, iw2);
            auto d23 = _mm_mul_ps(is2, iw3);

            auto d30 = _mm_mul_ps(is3, iw0);
            auto d31 = _mm_mul_ps(is3, iw1);
            auto d32 = _mm_mul_ps(is3, iw2);
            auto d33 = _mm_mul_ps(is3, iw3);

            for (int sz = 1; sz < src_depth_quad; ++sz) {
                const float* src_z    = src_dx + sz * src_depth_step;
                const float* weight_z = weight_dz + sz * 16;
                auto w0               = _mm_loadu_ps(weight_z + 4 * 0);
                auto w1               = _mm_loadu_ps(weight_z + 4 * 1);
                auto w2               = _mm_loadu_ps(weight_z + 4 * 2);
                auto w3               = _mm_loadu_ps(weight_z + 4 * 3);
                auto s0 = _mm_loadu_ps(src_z + 4*0);
                auto s1 = _mm_loadu_ps(src_z + 4*1);
                auto s2 = _mm_loadu_ps(src_z + 4*2);
                auto s3 = _mm_loadu_ps(src_z + 4*3);

                COMPUTE(0);
                COMPUTE(1);
                COMPUTE(2);
                COMPUTE(3);
            }
            STORE(0);
            STORE(1);
            STORE(2);
            STORE(3);
        }
        for (int dx = w4End; dx < width; ++dx) {
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

                auto sw0 = _mm_mul_ps(s, w0);
                auto sw1 = _mm_mul_ps(s, w1);
                auto sw2 = _mm_mul_ps(s, w2);
                auto sw3 = _mm_mul_ps(s, w3);
                d0 = _mm_add_ps(d0, sw0);
                d1 = _mm_add_ps(d1, sw1);
                d2 = _mm_add_ps(d2, sw2);
                d3 = _mm_add_ps(d3, sw3);
            }
            _mm_storeu_ps(dst_x, merge(d0, d1, d2, d3));
        }
    }
}

void _SSE_MNNGemmFloatUnit_4(float* dst, const float* src, const float* weight, size_t src_depth_quad, size_t dst_step,
                             size_t dst_depth_quad, size_t weight_depth_offset) {
    return _SSE_MNNGemmFloatCommon_4(dst, src, weight, src_depth_quad, dst_step, dst_depth_quad, 8, weight_depth_offset);
}
