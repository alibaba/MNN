//
//  CommonOptFunctionSSE.cpp
//  MNN
//
//  Created by MNN on 2018/11/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_USE_SSE

#include <emmintrin.h>
#include "CommonOptFunction.h"

void MNNAddBias(float* dst, const float* bias, size_t planeNumber, size_t biasNumber) {
    for (int z = 0; z < biasNumber; ++z) {
        auto biasV   = _mm_load_ps(bias + 4 * z);
        float* dst_z = dst + planeNumber * 4 * z;
        for (int p = 0; p < planeNumber; ++p) {
            auto dstV = _mm_add_ps(_mm_load_ps(dst_z + 4 * p), biasV);
            _mm_store_ps(dst_z + 4 * p, dstV);
        }
    }
}

void MNNAddBiasRelu(float* dst, const float* bias, size_t planeNumber, size_t biasNumber) {
    auto maxV = _mm_set1_ps(0.0f);
    for (int z = 0; z < biasNumber; ++z) {
        auto biasV   = _mm_load_ps(bias + 4 * z);
        float* dst_z = dst + planeNumber * 4 * z;
        for (int p = 0; p < planeNumber; ++p) {
            auto dstV = _mm_add_ps(_mm_load_ps(dst_z + 4 * p), biasV);
            dstV      = _mm_max_ps(dstV, maxV);
            _mm_store_ps(dst_z + 4 * p, dstV);
        }
    }
}

void MNNAddBiasRelu6(float* dst, const float* bias, size_t planeNumber, size_t biasNumber) {
    auto maxV = _mm_set1_ps(0.0f);
    auto minV = _mm_set1_ps(6.0f);
    for (int z = 0; z < biasNumber; ++z) {
        auto biasV   = _mm_load_ps(bias + 4 * z);
        float* dst_z = dst + planeNumber * 4 * z;
        for (int p = 0; p < planeNumber; ++p) {
            auto dstV = _mm_add_ps(_mm_load_ps(dst_z + 4 * p), biasV);
            dstV      = _mm_max_ps(dstV, maxV);
            dstV      = _mm_min_ps(dstV, minV);
            _mm_store_ps(dst_z + 4 * p, dstV);
        }
    }
}

void MNNCopyC4WithStride(const float* source, float* dest, size_t srcStride, size_t dstStride, size_t count) {
    for (int i = 0; i < count; ++i) {
        auto s = source + i * srcStride;
        auto d = dest + i * dstStride;
        _mm_store_ps(d, _mm_load_ps(s));
    }
}

void MNNAddC4WithStride(const float* source, float* dest, size_t srcStride, size_t dstStride, size_t count) {
    for (int i = 0; i < count; ++i) {
        auto s = source + i * srcStride;
        auto d = dest + i * dstStride;
        _mm_store_ps(d, _mm_add_ps(_mm_load_ps(s), _mm_load_ps(d)));
    }
}

#endif
