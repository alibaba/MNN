//
//  ReorderFunctions.cpp
//  MNN
//
//  Created by MNN on b'2021/07/09'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "FunctionSummary.hpp"
#include "core/Macro.h"
#include "backend/cpu/compute/CommonOptFunction.h"
#include <algorithm>
#include <cmath>

void MNNTranspose32Bit(int32_t* dstO, const int32_t* srcO, int32_t* dim) {
    int w         = dim[0];
    int h         = dim[1];
    int srcStride = dim[2];
    int dstStride = dim[3];
    auto wC4      = w / 4;
    auto hC4      = h / 4;
    for (int y = 0; y < hC4; ++y) {
        auto sy = (float*)srcO + 4 * y;
        auto dy = (float*)dstO + 4 * y * dstStride;
        for (int x = 0; x < wC4; ++x) {
            auto sx = sy + x * 4 * srcStride;
            auto dx = dy + 4 * x;
            auto s0 = _mm_loadu_ps(sx + srcStride * 0);
            auto s1 = _mm_loadu_ps(sx + srcStride * 1);
            auto s2 = _mm_loadu_ps(sx + srcStride * 2);
            auto s3 = _mm_loadu_ps(sx + srcStride * 3);
            _MM_TRANSPOSE4_PS(s0, s1, s2, s3);

            _mm_storeu_ps(dx + dstStride * 0, s0);
            _mm_storeu_ps(dx + dstStride * 1, s1);
            _mm_storeu_ps(dx + dstStride * 2, s2);
            _mm_storeu_ps(dx + dstStride * 3, s3);
        }
    }
    // Down
    for (int i = hC4 * 4; i < h; ++i) {
        auto si = srcO + i;
        auto di = dstO + i * dstStride;
        for (int j = 0; j < w; ++j) {
            auto sj = si + j * srcStride;
            auto dj = di + j;
            *dj     = *sj;
        }
    }
    // Right
    for (int i = 0; i < hC4 * 4; ++i) {
        auto si = srcO + i;
        auto di = dstO + i * dstStride;
        for (int j = wC4 * 4; j < w; ++j) {
            auto sj = si + j * srcStride;
            auto dj = di + j;
            *dj     = *sj;
        }
    }
}

void MNNTranspose16Bit(int16_t* dstO, const int16_t* srcO, int32_t* dim) {
    // TODO: support sse
    int w = dim[0];
    int h = dim[1];
    int srcStride = dim[2];
    int dstStride = dim[3];
    for (int i=0; i<h; ++i) {
        auto si = srcO + i;
        auto di = dstO + i * dstStride;
        for (int j=0; j<w; ++j) {
            auto sj = si + j * srcStride;
            auto dj = di + j;
            *dj = *sj;
        }
    }
}

void MNNUnpackC4(float* dst, const float* src, size_t area, size_t depth, int* areaOffset) {
    auto areaC4  = area / 4;
    auto depthC4 = depth / 4;
    int srcAreaOffset = areaOffset[0];
    int dstAreaOffset = areaOffset[1];
    for (int z = 0; z < depthC4; ++z) {
        auto dstPlane = dst + z * dstAreaOffset * 4;
        auto srcPlane = src + z * srcAreaOffset * 4;
        for (int x = 0; x < areaC4; ++x) {
            auto s  = srcPlane + 16 * x;
            auto d  = dstPlane + 4 * x;
            auto s0 = _mm_loadu_ps(s + 0 * 4);
            auto s1 = _mm_loadu_ps(s + 1 * 4);
            auto s2 = _mm_loadu_ps(s + 2 * 4);
            auto s3 = _mm_loadu_ps(s + 3 * 4);

            _MM_TRANSPOSE4_PS(s0, s1, s2, s3);

            _mm_storeu_ps(d + 0 * dstAreaOffset, s0);
            _mm_storeu_ps(d + 1 * dstAreaOffset, s1);
            _mm_storeu_ps(d + 2 * dstAreaOffset, s2);
            _mm_storeu_ps(d + 3 * dstAreaOffset, s3);
        }
    }
    auto areaRemain  = areaC4 * 4;
    auto depthRemain = depthC4 * 4;
    // Down
    int remain = depth - depthRemain;
    if (remain > 0) {
        float* dstPlane       = depthC4 * dstAreaOffset * 4 + dst;
        const float* srcPlane = src + depthC4 * srcAreaOffset * 4;
        for (int x = 0; x < area; ++x) {
            for (int y = 0; y < remain; y++) {
                dstPlane[y * dstAreaOffset + x] = srcPlane[4 * x + y];
            }
        }
    }
    // Right
    for (int z = 0; z < depthC4; ++z) {
        const float* srcPlane = z * srcAreaOffset * 4 + src;
        float* dstPlane       = dst + z * dstAreaOffset * 4;
        for (int x = areaRemain; x < area; ++x) {
            for (int y = 0; y < 4; y++) {
                dstPlane[y * dstAreaOffset + x] = srcPlane[4 * x + y];
            }
        }
    }
}

void MNNPackC4(float* dst, const float* src, size_t area, size_t depth, int* areaOffset) {
    auto areaC4  = area / 4;
    auto depthC4 = depth / 4;
    int srcAreaOffset = areaOffset[0];
    int dstAreaOffset = areaOffset[1];
    for (int z = 0; z < depthC4; ++z) {
        auto dstPlane = dst + z * dstAreaOffset * 4;
        auto srcPlane = src + z * srcAreaOffset * 4;
        for (int x = 0; x < areaC4; ++x) {
            auto s  = srcPlane + 4 * x;
            auto d  = dstPlane + 16 * x;
            auto s0 = _mm_loadu_ps(s + 0 * srcAreaOffset);
            auto s1 = _mm_loadu_ps(s + 1 * srcAreaOffset);
            auto s2 = _mm_loadu_ps(s + 2 * srcAreaOffset);
            auto s3 = _mm_loadu_ps(s + 3 * srcAreaOffset);

            _MM_TRANSPOSE4_PS(s0, s1, s2, s3);

            _mm_storeu_ps(d + 4 * 0, s0);
            _mm_storeu_ps(d + 4 * 1, s1);
            _mm_storeu_ps(d + 4 * 2, s2);
            _mm_storeu_ps(d + 4 * 3, s3);
        }
    }
    auto areaRemain  = areaC4 * 4;
    auto depthRemain = depthC4 * 4;
    // Down
    int remain = depth - depthRemain;
    if (remain > 0) {
        float* dstPlane       = depthC4 * dstAreaOffset * 4 + dst;
        const float* srcPlane = src + depthC4 * srcAreaOffset * 4;
        for (int x = 0; x < area; ++x) {
            for (int y = 0; y < remain; y++) {
                dstPlane[4 * x + y] = srcPlane[y * srcAreaOffset + x];
            }
            for (int y = remain; y < 4; y++) {
                dstPlane[4 * x + y] = 0;
            }
        }
    }
    // Right
    for (int z = 0; z < depthC4; ++z) {
        float* dstPlane       = z * dstAreaOffset * 4 + dst;
        const float* srcPlane = src + z * srcAreaOffset * 4;
        for (int x = areaRemain; x < area; ++x) {
            float s0 = srcPlane[x];
            float s1 = srcPlane[x + srcAreaOffset];
            float s2 = srcPlane[x + srcAreaOffset * 2];
            float s3 = srcPlane[x + srcAreaOffset * 3];
            _mm_storeu_ps(dstPlane + 4 * x, _mm_set_ps(s3, s2, s1, s0));
        }
    }
}
