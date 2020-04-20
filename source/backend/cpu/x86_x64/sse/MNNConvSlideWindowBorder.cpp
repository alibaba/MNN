//
//  MNNConvSlideWindowBorder.cpp
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

void _SSE_MNNConvSlideWindowBorder(float* dst, const float* src, const float* weight, size_t src_depth_quad,
                              size_t src_depth_step, size_t fw, size_t fh, size_t weight_y_step, size_t weight_z_step,
                              size_t dilateX_step, size_t dilateY_step, float* alpha) {
    int sz, fx, fy;
    auto d0 = _mm_set1_ps(0.0f);
    auto d1 = _mm_set1_ps(0.0f);
    auto d2 = _mm_set1_ps(0.0f);
    auto d3 = _mm_set1_ps(0.0f);

    for (sz = 0; sz < src_depth_quad; ++sz) {
        const float* src_z    = src + sz * src_depth_step;
        const float* weight_z = weight + sz * weight_z_step;
        for (fy = 0; fy < fh; ++fy) {
            const float* src_y    = src_z + fy * dilateY_step;
            const float* weight_y = weight_z + fy * weight_y_step;
            for (fx = 0; fx < fw; ++fx) {
                const float* weight_x = weight_y + 16 * fx;
                const float* src_x    = src_y + fx * dilateX_step;
                auto w0               = _mm_loadu_ps(weight_x + 4 * 0);
                auto w1               = _mm_loadu_ps(weight_x + 4 * 1);
                auto w2               = _mm_loadu_ps(weight_x + 4 * 2);
                auto w3               = _mm_loadu_ps(weight_x + 4 * 3);
                auto s = _mm_loadu_ps(src_x);
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
        }
    }
    auto h0 = _mm_hadd_ps(d0, d1);
    auto h1 = _mm_hadd_ps(d2, d3);
    _mm_storeu_ps(dst, _mm_hadd_ps(h0, h1));
}
