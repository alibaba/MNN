//
//  MNNMatrixSub.cpp
//  MNN
//
//  Created by MNN on 2019/08/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "FunctionSummary.hpp"

void _AVX_MNNMatrixSub(float* C, const float* A, const float* B, size_t widthC4, size_t cStride, size_t aStride,
                       size_t bStride, size_t height) {
    for (int y = 0; y < height; ++y) {
        auto a = A + aStride * y;
        auto b = B + bStride * y;
        auto c = C + cStride * y;
        for (int x = 0; x < widthC4; ++x) {
            _mm256_storeu_ps(c + 8 * x, _mm256_sub_ps(_mm256_loadu_ps(a + 8 * x), _mm256_loadu_ps(b + 8 * x)));
        }
    }
}
