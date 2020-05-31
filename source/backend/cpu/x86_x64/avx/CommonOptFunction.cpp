//
//  CommonOptFunction.cpp
//  MNN
//
//  Created by MNN on 2019/08/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <immintrin.h>

void _AVX_MNNAddBias(float* dst, const float* bias, size_t planeNumber, size_t biasNumber) {
    if (planeNumber == 0) {
        return;
    }
    for (int z = 0; z < biasNumber; ++z) {
        auto biasV   = _mm256_broadcast_ps((const __m128 *)(bias + 4 * z));
        float* dst_z = dst + planeNumber * 4 * z;
        for (int p = 0; p < planeNumber - 1; p += 2) {
            auto dstV = _mm256_add_ps(_mm256_loadu_ps(dst_z + 4 * p), biasV);
            _mm256_storeu_ps(dst_z + 4 * p, dstV);
        }
        if (planeNumber % 2 == 1) {
            _mm256_zeroall();
            auto biasV = _mm_loadu_ps(bias + 4 * z);
            auto dstV = _mm_add_ps(_mm_loadu_ps(dst_z + 4 * (planeNumber - 1)), biasV);
            _mm_storeu_ps(dst_z + 4 * (planeNumber - 1), dstV);
        }
    }
    _mm256_zeroall();
}

void _AVX_MNNAddBiasRelu(float* dst, const float* bias, size_t planeNumber, size_t biasNumber) {
    if (planeNumber == 0) {
        return;
    }
    auto maxV = _mm256_set1_ps(0.0f);
    for (int z = 0; z < biasNumber; ++z) {
        auto biasV   = _mm256_broadcast_ps((const __m128 *)(bias + 4 * z));
        float* dst_z = dst + planeNumber * 4 * z;
        for (int p = 0; p < planeNumber - 1; p += 2) {
            auto dstV = _mm256_add_ps(_mm256_loadu_ps(dst_z + 4 * p), biasV);
            dstV      = _mm256_max_ps(dstV, maxV);
            _mm256_storeu_ps(dst_z + 4 * p, dstV);
        }
        if (planeNumber % 2 == 1) {
            _mm256_zeroall();
            auto biasV = _mm_loadu_ps(bias + 4 * z);
            auto dstV  = _mm_add_ps(_mm_loadu_ps(dst_z + 4 * (planeNumber - 1)), biasV);
            dstV       = _mm_max_ps(dstV, _mm_set1_ps(0.0f));
            _mm_storeu_ps(dst_z + 4 * (planeNumber - 1), dstV);
            maxV = _mm256_set1_ps(0.0f);
        }
    }
    _mm256_zeroall();
}

void _AVX_MNNAddBiasRelu6(float* dst, const float* bias, size_t planeNumber, size_t biasNumber) {
    if (planeNumber == 0) {
        return;
    }
    auto maxV = _mm256_set1_ps(0.0f);
    auto minV = _mm256_set1_ps(6.0f);
    for (int z = 0; z < biasNumber; ++z) {
        auto biasV   = _mm256_broadcast_ps((const __m128 *)(bias + 4 * z));
        float* dst_z = dst + planeNumber * 4 * z;
        for (int p = 0; p < planeNumber - 1; p += 2) {
            auto dstV = _mm256_add_ps(_mm256_loadu_ps(dst_z + 4 * p), biasV);
            dstV      = _mm256_max_ps(dstV, maxV);
            dstV      = _mm256_min_ps(dstV, minV);
            _mm256_storeu_ps(dst_z + 4 * p, dstV);
        }
        if (planeNumber % 2 == 1) {
            _mm256_zeroall();
            auto biasV = _mm_loadu_ps(bias + 4 * z);
            auto dstV  = _mm_add_ps(_mm_loadu_ps(dst_z + 4 * (planeNumber - 1)), biasV);
            dstV       = _mm_min_ps(_mm_max_ps(dstV, _mm_set1_ps(0.0f)), _mm_set1_ps(6.0f));
            _mm_storeu_ps(dst_z + 4 * (planeNumber - 1), dstV);
            maxV = _mm256_set1_ps(0.0f);
            minV = _mm256_set1_ps(6.0f);
        }
    }
    _mm256_zeroall();
}
