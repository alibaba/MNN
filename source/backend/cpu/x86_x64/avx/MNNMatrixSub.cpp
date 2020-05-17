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
        for (int x = 0; x < widthC4 - 1; x += 2) {
            _mm256_storeu_ps(c + 4 * x, _mm256_sub_ps(_mm256_loadu_ps(a + 4 * x), _mm256_loadu_ps(b + 4 * x)));
        }
        if (widthC4 % 2 == 1) {
            auto dst = _mm_sub_ps(_mm_loadu_ps(a + 4 * (widthC4 - 1)), _mm_loadu_ps(b + 4 * (widthC4 - 1)));
            _mm_storeu_ps(c + 4 * (widthC4 - 1), dst);
        }
    }
}
void _AVX_MNNStrassenMergeCFunction(float* c11, float* c12, float* c21, float* c22, float* xAddr, size_t cStride,
                                    size_t length, size_t hSub) {
    auto lengthC4 = length / 8;
    for (int y=0; y<hSub; ++y) {
        auto c11Y = c11 + y * cStride;
        auto c12Y = c12 + y * cStride;
        auto c22Y = c22 + y * cStride;
        auto c21Y = c21 + y * cStride;
        auto xY = xAddr + y * length;
        for (int x=0; x<lengthC4; ++x) {
            auto xv = _mm256_loadu_ps(xY + 8*x);
            auto c21v = _mm256_loadu_ps(c21Y + 8*x);
            auto c11v = _mm256_loadu_ps(c11Y + 8*x);
            auto c22v = _mm256_loadu_ps(c22Y + 8*x);
            auto c12v = _mm256_loadu_ps(c12Y + 8*x);
            c12v = c12v + xv;
            c21v = c12v + c21v;
            c12v = c22v + c12v;
            c22v = c22v + c21v;
            c12v = c11v + c12v;
            _mm256_storeu_ps(c12Y + 8*x, c12v);
            _mm256_storeu_ps(c22Y + 8*x, c22v);
            _mm256_storeu_ps(c21Y + 8*x, c21v);
            _mm256_storeu_ps(c11Y + 8*x, c11v);
        }
        for (int x=lengthC4*8; x<length; ++x) {
            auto xv = xY[x];
            auto c21v = c21Y[x];
            auto c11v = c11Y[x];
            auto c22v = c22Y[x];
            auto c12v = c12Y[x];
            c12v = c12v + xv;
            c21v = c12v + c21v;
            c12v = c22v + c12v;
            c22v = c22v + c21v;
            c12v = c11v + c12v;
            c12Y[x] = c12v;
            c22Y[x] = c22v;
            c21Y[x] = c21v;
            c11Y[x] = c11v;
        }
    }
}
