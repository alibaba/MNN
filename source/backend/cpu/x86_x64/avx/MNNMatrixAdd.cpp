//
//  MNNMatrixAdd.cpp
//  MNN
//
//  Created by MNN on 2019/08/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "FunctionSummary.hpp"
void _AVX_MNNMatrixAdd(float* C, const float* A, const float* B, size_t widthC4, size_t cStride, size_t aStride,
                       size_t bStride, size_t height) {
    for (int y = 0; y < height; ++y) {
        auto a = A + aStride * y;
        auto b = B + bStride * y;
        auto c = C + cStride * y;
        for (int x = 0; x < widthC4; ++x) {
            _mm256_storeu_ps(c + 8 * x, _mm256_add_ps(_mm256_loadu_ps(b + 8 * x), _mm256_loadu_ps(a + 8 * x)));
        }
    }
}

void _AVX_MNNStrassenMergeCFunction(float* c11, float* c12, float* c21, float* c22, float* xAddr, size_t cStride, size_t eSub, size_t hSub) {
    const int unit = 8;
    for (int y=0; y<hSub; ++y) {
        auto c11Y = c11 + y * cStride;
        auto c12Y = c12 + y * cStride;
        auto c22Y = c22 + y * cStride;
        auto c21Y = c21 + y * cStride;
        auto xY = xAddr + y * eSub * unit;
        for (int x=0; x<eSub; ++x) {
            auto xv = _mm256_loadu_ps(xY + unit*x);
            auto c21v = _mm256_loadu_ps(c21Y + unit*x);
            auto c11v = _mm256_loadu_ps(c11Y + unit*x);
            auto c22v = _mm256_loadu_ps(c22Y + unit*x);
            auto c12v = _mm256_loadu_ps(c12Y + unit*x);
            c12v = _mm256_add_ps(c12v, xv);
            c21v = _mm256_add_ps(c12v, c21v);
            c12v = _mm256_add_ps(c22v, c12v);
            c22v = _mm256_add_ps(c22v, c21v);
            c12v = _mm256_add_ps(c11v, c12v);
            _mm256_storeu_ps(c12Y + unit*x, c12v);
            _mm256_storeu_ps(c22Y + unit*x, c22v);
            _mm256_storeu_ps(c21Y + unit*x, c21v);
        }
    }
}
