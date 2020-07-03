//
//  CommonOptFunction.cpp
//  MNN
//
//  Created by MNN on 2019/08/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <emmintrin.h>
#include <string.h>
#include <algorithm>
#include "core/Macro.h"

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
void _SSE_MNNPackedMatMulRemain(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, float* cache, const float* postParameters, const float* bias) {
    auto h = parameter[2];
    auto l = parameter[1];
    auto cStride = parameter[3] / sizeof(float);
    auto hRemain = parameter[4];
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride = bExtraStride + l * 6;
    auto hC4 = UP_DIV(h, 4);
    for (int y=0; y<hC4; ++y) {
        ::memset(C + y * cStride, 0, eSize * 4 * sizeof(float));
    }
    float alpha = 1.0f;
    float beta = 0.0f;
    float minValue = -std::numeric_limits<float>().max();
    float maxValue = std::numeric_limits<float>().max();
    if (nullptr != postParameters) {
        minValue = postParameters[2];
        maxValue = postParameters[3];
        alpha = postParameters[0];
        beta = postParameters[1];
    }
    
    for (int x=0; x<eSize; ++x) {
        auto dst = C + 4 * x;
        auto src = A + x;
        for (int ry=0; ry<h; ++ry) {
            auto y = ry / 4;
            auto yRemain = ry % 4;
            auto bY = B + y * bStride;
            auto dstY = dst + y * cStride;
            int wdy = ry / 6;
            int wdyRemain = ry % 6;
            auto weight = B + wdy * bStride + wdyRemain;
            float summer = 0.0f;
            for (int z=0; z<l; ++z) {
                auto aZ = src + z * 16;
                auto wZ = weight + z * 6;
                summer += wZ[0] * aZ[0];
            }
            float originValue = dstY[yRemain];
            if (nullptr != bias) {
                originValue = bias[ry];
            }
            auto dstValue = originValue * beta + alpha * summer;
            dstValue = std::min(dstValue, maxValue);
            dstValue = std::max(dstValue, minValue);
            dstY[yRemain] = dstValue;
        }
    }
}

void _SSE_MNNPackedMatMul(float* C, const float* A, const float* B, const size_t* parameter, float* cache, const float* postParameters, const float* bias) {
    return _SSE_MNNPackedMatMulRemain(C, A, B, 16, parameter, cache, postParameters, bias);
}
