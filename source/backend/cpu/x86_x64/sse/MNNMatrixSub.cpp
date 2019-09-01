//
//  MNNMatrixSub.cpp
//  MNN
//
//  Created by MNN on 2018/11/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <immintrin.h>
#include <stdint.h>

void _SSE_MNNMatrixSub(float* C, const float* A, const float* B, size_t widthC4, size_t cStride, size_t aStride,
                  size_t bStride, size_t height) {
    for (int y = 0; y < height; ++y) {
        auto a = A + aStride * y;
        auto b = B + bStride * y;
        auto c = C + cStride * y;
        for (int x = 0; x < widthC4; ++x) {
            _mm_store_ps(c + 4 * x, _mm_sub_ps(_mm_load_ps(a + 4 * x), _mm_load_ps(b + 4 * x)));
        }
    }
}
