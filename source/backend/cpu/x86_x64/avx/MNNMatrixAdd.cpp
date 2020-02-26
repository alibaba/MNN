//
//  MNNMatrixAdd.cpp
//  MNN
//
//  Created by MNN on 2019/08/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <immintrin.h>
#include <stdint.h>

void _AVX_MNNMatrixAdd(float* C, const float* A, const float* B, size_t widthC4, size_t cStride, size_t aStride,
                  size_t bStride, size_t height) {
    for (int y = 0; y < height; ++y) {
        auto a = A + aStride * y;
        auto b = B + bStride * y;
        auto c = C + cStride * y;
        for (int x = 0; x < widthC4 - 1; x += 2) {
            _mm256_storeu_ps(c + 4 * x, _mm256_add_ps(_mm256_loadu_ps(b + 4 * x), _mm256_loadu_ps(a + 4 * x)));
        }
        if (widthC4 % 2 == 1) {
            _mm256_zeroall();
            auto dst = _mm_add_ps(_mm_loadu_ps(a + 4 * (widthC4 - 1)), _mm_loadu_ps(b + 4 * (widthC4 - 1)));
            _mm_storeu_ps(c + 4 * (widthC4 - 1), dst);
        }
    }
    _mm256_zeroall();
}
