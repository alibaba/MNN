//
//  MNNMatrixAdd.cpp
//  MNN
//
//  Created by MNN on 2018/11/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_USE_SSE

#include <immintrin.h>
#include <stdint.h>
#include "ConvOpt.h"
#include "CommonHelperSSE.hpp"

TargetBegin("sse2")
static void _SSE_MNNMatrixAdd(float* C, const float* A, const float* B, size_t widthC4, size_t cStride, size_t aStride,
                  size_t bStride, size_t height) {
    for (int y = 0; y < height; ++y) {
        auto a = A + aStride * y;
        auto b = B + bStride * y;
        auto c = C + cStride * y;
        for (int x = 0; x < widthC4; ++x) {
            _mm_store_ps(c + 4 * x, _mm_add_ps(_mm_load_ps(b + 4 * x), _mm_load_ps(a + 4 * x)));
        }
    }
}
TargetEnd("sse2")

TargetBegin("avx")
static void _AVX_MNNMatrixAdd(float* C, const float* A, const float* B, size_t widthC4, size_t cStride, size_t aStride,
                  size_t bStride, size_t height) {
    for (int y = 0; y < height; ++y) {
        auto a = A + aStride * y;
        auto b = B + bStride * y;
        auto c = C + cStride * y;
        for (int x = 0; x < widthC4; x += 2) {
            _mm256_storeu_ps(c + 4 * x, _mm256_add_ps(_mm256_loadu_ps(b + 4 * x), _mm256_loadu_ps(a + 4 * x)));
        }
    }
    _mm256_zeroall();
}
TargetEnd("avx")

void MNNMatrixAdd(float* C, const float* A, const float* B, size_t widthC4, size_t cStride, size_t aStride,
                  size_t bStride, size_t height) {
    if (widthC4 % 2 == 0 && cpu_feature_available(AVX)) {
        _AVX_MNNMatrixAdd(C, A, B, widthC4, cStride, aStride, bStride, height);
    } else {
        _SSE_MNNMatrixAdd(C, A, B, widthC4, cStride, aStride, bStride, height);
    }
}

#endif
