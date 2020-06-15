//
//  MNNConvSlideWindowMiddle.cpp
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

void _SSE_MNNConvSlideWindowMiddle(float* dst, const float* src, const float* weight, size_t width, size_t src_w_setup,
                              size_t src_depth_quad, size_t src_depth_step, size_t fw, size_t fh, size_t dilateX_step,
                              size_t dilateY_step, float* alpha) {
    for (int dx = 0; dx < width; ++dx) {
        float* dst_x  = dst + dx * 4;
        auto d0 = _mm_set1_ps(0.0f);
        auto d1 = _mm_set1_ps(0.0f);
        auto d2 = _mm_set1_ps(0.0f);
        auto d3 = _mm_set1_ps(0.0f);

        const float* src_dx = src + src_w_setup * dx;
        for (int sz = 0; sz < src_depth_quad; ++sz) {
            const float* src_z    = src_dx + sz * src_depth_step;
            const float* weight_z = weight + sz * fh * fw * 16;
            for (int fy = 0; fy < fh; ++fy) {
                const float* src_y    = src_z + fy * dilateY_step;
                const float* weight_y = weight_z + fy * fw * 16;
                for (int fx = 0; fx < fw; ++fx) {
                    const float* weight_x = weight_y + 16 * fx;
                    const float* src_x    = src_y + fx * dilateX_step;
                    auto w0               = _mm_loadu_ps(weight_x + 4 * 0);
                    auto w1               = _mm_loadu_ps(weight_x + 4 * 1);
                    auto w2               = _mm_loadu_ps(weight_x + 4 * 2);
                    auto w3               = _mm_loadu_ps(weight_x + 4 * 3);
                    auto s = _mm_loadu_ps(src_x);

                    auto sw0 = _mm_mul_ps(s, w0);
                    d0 = _mm_add_ps(d0, sw0);
                    auto sw1 = _mm_mul_ps(s, w1);
                    d1 = _mm_add_ps(d1, sw1);
                    auto sw2 = _mm_mul_ps(s, w2);
                    d2 = _mm_add_ps(d2, sw2);
                    auto sw3 = _mm_mul_ps(s, w3);
                    d3 = _mm_add_ps(d3, sw3);
                }
            }
        }
        auto h0 = _mm_hadd_ps(d0, d1);
        auto h1 = _mm_hadd_ps(d2, d3);
        _mm_storeu_ps(dst_x, _mm_hadd_ps(h0, h1));
    }
}
