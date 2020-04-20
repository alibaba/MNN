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
#include <MNN/MNNDefine.h>
static inline __m256
MNN_mm256_loadu2_m128(float const *__addr_hi, float const *__addr_lo)
{
  return _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_loadu_ps(__addr_lo)), _mm_loadu_ps(__addr_hi), 1);
}
void _AVX_MNNConvSlideWindowMiddle(float* dst, const float* src, const float* weight, size_t width, size_t src_w_setup,
                              size_t src_depth_quad, size_t src_depth_step, size_t fw, size_t fh, size_t dilateX_step,
                              size_t dilateY_step, float* alpha) {
    int wC2 = width / 4;
    for (int dx2 = 0; dx2 < wC2; ++dx2) {
        auto dx = dx2 * 4;
        float* dst_x  = dst + dx * 4;
        auto d0 = _mm256_set1_ps(0.0f);
        auto d1 = _mm256_set1_ps(0.0f);
        auto d2 = _mm256_set1_ps(0.0f);
        auto d3 = _mm256_set1_ps(0.0f);

        auto d4 = _mm256_set1_ps(0.0f);
        auto d5 = _mm256_set1_ps(0.0f);
        auto d6 = _mm256_set1_ps(0.0f);
        auto d7 = _mm256_set1_ps(0.0f);

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
                    auto w0               = _mm256_broadcast_ps((const __m128 *)(weight_x + 4 * 0));
                    auto w1               = _mm256_broadcast_ps((const __m128 *)(weight_x + 4 * 1));
                    auto w2               = _mm256_broadcast_ps((const __m128 *)(weight_x + 4 * 2));
                    auto w3               = _mm256_broadcast_ps((const __m128 *)(weight_x + 4 * 3));

                    auto s0 = MNN_mm256_loadu2_m128(src_x + src_w_setup, src_x);
                    auto s1 = MNN_mm256_loadu2_m128(src_x + 3 * src_w_setup, src_x + 2 * src_w_setup);
#ifdef MNN_FMA_ENABLE
#define COMPUTE(i, j, k) d##k = _mm256_fmadd_ps(s##i, w##j, d##k)
#else
#define COMPUTE(i, j, k) d##k = _mm256_add_ps(_mm256_mul_ps(s##i, w##j), d##k)
#endif
                    COMPUTE(0, 0, 0);
                    COMPUTE(0, 1, 1);
                    COMPUTE(0, 2, 2);
                    COMPUTE(0, 3, 3);
                    COMPUTE(1, 0, 4);
                    COMPUTE(1, 1, 5);
                    COMPUTE(1, 2, 6);
                    COMPUTE(1, 3, 7);
#undef COMPUTE
                }
            }
        }
        auto h0 = _mm256_hadd_ps(d0, d1);
        auto h1 = _mm256_hadd_ps(d2, d3);
        _mm256_storeu_ps(dst_x + 8 * 0, _mm256_hadd_ps(h0, h1));
        auto h2 = _mm256_hadd_ps(d4, d5);
        auto h3 = _mm256_hadd_ps(d6, d7);
        _mm256_storeu_ps(dst_x + 8 * 1, _mm256_hadd_ps(h2, h3));
    }
    
    for (int dx = wC2 * 4; dx < width; ++dx) {
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
        _mm_storeu_ps(dst_x, _mm_hadd_ps(h0, h1));
    }
}
