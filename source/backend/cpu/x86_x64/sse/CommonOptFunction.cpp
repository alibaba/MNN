//
//  CommonOptFunction.cpp
//  MNN
//
//  Created by MNN on 2019/08/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <emmintrin.h>

void _SSE_MNNAddBias(float* dst, const float* bias, size_t planeNumber, size_t biasNumber) {
    for (int z = 0; z < biasNumber; ++z) {
        auto biasV   = _mm_loadu_ps(bias + 4 * z);
        float* dst_z = dst + planeNumber * 4 * z;
        for (int p = 0; p < planeNumber; ++p) {
            auto dstV = _mm_add_ps(_mm_loadu_ps(dst_z + 4 * p), biasV);
            _mm_storeu_ps(dst_z + 4 * p, dstV);
        }
    }
}

void _SSE_MNNAddBiasRelu(float* dst, const float* bias, size_t planeNumber, size_t biasNumber) {
    auto maxV = _mm_set1_ps(0.0f);
    for (int z = 0; z < biasNumber; ++z) {
        auto biasV   = _mm_loadu_ps(bias + 4 * z);
        float* dst_z = dst + planeNumber * 4 * z;
        for (int p = 0; p < planeNumber; ++p) {
            auto dstV = _mm_add_ps(_mm_loadu_ps(dst_z + 4 * p), biasV);
            dstV      = _mm_max_ps(dstV, maxV);
            _mm_storeu_ps(dst_z + 4 * p, dstV);
        }
    }
}

void _SSE_MNNAddBiasRelu6(float* dst, const float* bias, size_t planeNumber, size_t biasNumber) {
    auto maxV = _mm_set1_ps(0.0f);
    auto minV = _mm_set1_ps(6.0f);
    for (int z = 0; z < biasNumber; ++z) {
        auto biasV   = _mm_loadu_ps(bias + 4 * z);
        float* dst_z = dst + planeNumber * 4 * z;
        for (int p = 0; p < planeNumber; ++p) {
            auto dstV = _mm_add_ps(_mm_loadu_ps(dst_z + 4 * p), biasV);
            dstV      = _mm_max_ps(dstV, maxV);
            dstV      = _mm_min_ps(dstV, minV);
            _mm_storeu_ps(dst_z + 4 * p, dstV);
        }
    }
}

void _SSE_MNNCopyC4WithStride(const float* source, float* dest, size_t srcStride, size_t dstStride, size_t count) {
    for (int i = 0; i < count; ++i) {
        auto s = source + i * srcStride;
        auto d = dest + i * dstStride;
        _mm_storeu_ps(d, _mm_loadu_ps(s));
    }
}

void _SSE_MNNAddC4WithStride(const float* source, float* dest, size_t srcStride, size_t dstStride, size_t count) {
    for (int i = 0; i < count; ++i) {
        auto s = source + i * srcStride;
        auto d = dest + i * dstStride;
        _mm_storeu_ps(d, _mm_add_ps(_mm_loadu_ps(s), _mm_loadu_ps(d)));
    }
}

void _SSE_MNNReluWithSlopeChannel(float* dst, const float* src, const float* slope, size_t sizeQuad, size_t depthQuad) {
    auto zero = _mm_set1_ps(0.0f);
    for (int j = 0; j < depthQuad; j++) {
        auto slopeZ = _mm_loadu_ps(slope);
        const float* srcZ   = src + 4 * j * sizeQuad;
        float* dstZ         = dst + 4 * j * sizeQuad;
        for (int i = 0; i < sizeQuad; i++) {
            auto src = _mm_loadu_ps(srcZ + 4 * i);
            auto mask0 = _mm_cmplt_ps(src, zero);
            auto mask1 = _mm_cmpge_ps(src, zero);
            auto other = _mm_mul_ps(src, slopeZ);
            _mm_storeu_ps(dstZ + 4 * i, _mm_add_ps(_mm_and_ps(other, mask0), _mm_and_ps(src, mask1)));
        }
    }
}
