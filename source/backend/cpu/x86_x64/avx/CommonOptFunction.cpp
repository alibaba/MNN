//
//  CommonOptFunction.cpp
//  MNN
//
//  Created by MNN on 2019/08/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <float.h>
#include <string.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include "FunctionSummary.hpp"
#include "core/Macro.h"
#include "backend/cpu/CPUPool.hpp"
#include "backend/cpu/BinaryUtils.hpp"
#include "Vec8.hpp"
#include "backend/cpu/x86_x64/sse/FunctionSummary.hpp"

void _AVX_MNNCopyC4WithStride(const float* source, float* dest, size_t srcStride, size_t dstStride, size_t count) {
    for (int i = 0; i < count; ++i) {
        auto s = source + i * srcStride;
        auto d = dest + i * dstStride;
        _mm256_storeu_ps(d, _mm256_loadu_ps(s));
    }
}
void _AVX_MNNAddC4WithStride(const float* source, float* dest, size_t srcStride, size_t dstStride, size_t count) {
    for (int i = 0; i < count; ++i) {
        auto s = source + i * srcStride;
        auto d = dest + i * dstStride;
        _mm256_storeu_ps(d, _mm256_add_ps(_mm256_loadu_ps(s), _mm256_loadu_ps(d)));
    }
}

#define PACK_UNIT 8
void _AVX_MNNPackCUnit(float* dst, const float* src, size_t area, size_t depth, int* areaOffset) {
    auto areaC4  = area / PACK_UNIT;
    auto depthC4 = depth / PACK_UNIT;
    auto srcAreaOffset = areaOffset[0];
    auto dstAreaOffset = areaOffset[1];
    __m256 t0, t1, t2, t3, t4, t5, t6, t7;
    for (int z = 0; z < depthC4; ++z) {
        auto dstPlane = dst + z * dstAreaOffset * PACK_UNIT;
        auto srcPlane = src + z * srcAreaOffset * PACK_UNIT;
        for (int x = 0; x < areaC4; ++x) {
            auto s  = srcPlane + PACK_UNIT * x;
            auto d  = dstPlane + PACK_UNIT * PACK_UNIT * x;
            auto r0 = _mm256_loadu_ps(s + 0 * srcAreaOffset);
            auto r1 = _mm256_loadu_ps(s + 1 * srcAreaOffset);
            auto r2 = _mm256_loadu_ps(s + 2 * srcAreaOffset);
            auto r3 = _mm256_loadu_ps(s + 3 * srcAreaOffset);
            auto r4 = _mm256_loadu_ps(s + 4 * srcAreaOffset);
            auto r5 = _mm256_loadu_ps(s + 5 * srcAreaOffset);
            auto r6 = _mm256_loadu_ps(s + 6 * srcAreaOffset);
            auto r7 = _mm256_loadu_ps(s + 7 * srcAreaOffset);

            TRANSPOSE_8x8;

            _mm256_storeu_ps(d + PACK_UNIT * 0, t0);
            _mm256_storeu_ps(d + PACK_UNIT * 1, t1);
            _mm256_storeu_ps(d + PACK_UNIT * 2, t2);
            _mm256_storeu_ps(d + PACK_UNIT * 3, t3);
            _mm256_storeu_ps(d + PACK_UNIT * 4, t4);
            _mm256_storeu_ps(d + PACK_UNIT * 5, t5);
            _mm256_storeu_ps(d + PACK_UNIT * 6, t6);
            _mm256_storeu_ps(d + PACK_UNIT * 7, t7);
        }
    }
    auto areaRemain  = areaC4 * PACK_UNIT;
    auto depthRemain = depthC4 * PACK_UNIT;
    // Down
    int remain = depth - depthRemain;
    if (remain > 0) {
        float* dstPlane       = depthC4 * dstAreaOffset * PACK_UNIT + dst;
        const float* srcPlane = src + depthC4 * srcAreaOffset * PACK_UNIT;
        {
            for (int x = 0; x < areaC4; ++x) {
                auto s  = srcPlane + PACK_UNIT * x;
                auto d  = dstPlane + PACK_UNIT * PACK_UNIT * x;
                auto r0 = _mm256_loadu_ps(s + 0 * srcAreaOffset);
                auto r1 = _mm256_setzero_ps();
                auto r2 = _mm256_setzero_ps();
                auto r3 = _mm256_setzero_ps();
                auto r4 = _mm256_setzero_ps();
                auto r5 = _mm256_setzero_ps();
                auto r6 = _mm256_setzero_ps();
                auto r7 = _mm256_setzero_ps();
                switch (remain) {
                    case 7:
                        r6 = _mm256_loadu_ps(s + 6 * srcAreaOffset);
                    case 6:
                        r5 = _mm256_loadu_ps(s + 5 * srcAreaOffset);
                    case 5:
                        r4 = _mm256_loadu_ps(s + 4 * srcAreaOffset);
                    case 4:
                        r3 = _mm256_loadu_ps(s + 3 * srcAreaOffset);
                    case 3:
                        r2 = _mm256_loadu_ps(s + 2 * srcAreaOffset);
                    case 2:
                        r1 = _mm256_loadu_ps(s + 1 * srcAreaOffset);
                    default:
                        break;
                }

                TRANSPOSE_8x8;

                _mm256_storeu_ps(d + PACK_UNIT * 7, t7);
                _mm256_storeu_ps(d + PACK_UNIT * 6, t6);
                _mm256_storeu_ps(d + PACK_UNIT * 5, t5);
                _mm256_storeu_ps(d + PACK_UNIT * 4, t4);
                _mm256_storeu_ps(d + PACK_UNIT * 3, t3);
                _mm256_storeu_ps(d + PACK_UNIT * 2, t2);
                _mm256_storeu_ps(d + PACK_UNIT * 1, t1);
                _mm256_storeu_ps(d + PACK_UNIT * 0, t0);
            }
        }
        for (int x = areaRemain; x < area; ++x) {
            for (int y = 0; y < remain; y++) {
                dstPlane[PACK_UNIT * x + y] = srcPlane[y * srcAreaOffset + x];
            }
            for (int y = remain; y < PACK_UNIT; y++) {
                dstPlane[PACK_UNIT * x + y] = 0;
            }
        }
    }
    // Right
    for (int z = 0; z < depthC4; ++z) {
        float* dstPlane       = z * dstAreaOffset * PACK_UNIT + dst;
        const float* srcPlane = src + z * srcAreaOffset * PACK_UNIT;
        for (int x = areaRemain; x < area; ++x) {
            float s0 = srcPlane[x];
            float s1 = srcPlane[x + srcAreaOffset];
            float s2 = srcPlane[x + srcAreaOffset * 2];
            float s3 = srcPlane[x + srcAreaOffset * 3];
            float s4 = srcPlane[x + srcAreaOffset * 4];
            float s5 = srcPlane[x + srcAreaOffset * 5];
            float s6 = srcPlane[x + srcAreaOffset * 6];
            float s7 = srcPlane[x + srcAreaOffset * 7];
            _mm256_storeu_ps(dstPlane + PACK_UNIT * x, _mm256_set_ps(s7, s6, s5, s4, s3, s2, s1, s0));
        }
    }
}
void _AVX_MNNUnpackCUnit(float* dst, const float* src, size_t area, size_t depth, int* areaOffset) {
    auto areaC4  = area / PACK_UNIT;
    auto depthC4 = depth / PACK_UNIT;
    auto srcAreaOffset = areaOffset[0];
    auto dstAreaOffset = areaOffset[1];
    __m256 t0, t1, t2, t3, t4, t5, t6, t7;
    for (int z = 0; z < depthC4; ++z) {
        auto dstPlane = dst + z * dstAreaOffset * PACK_UNIT;
        auto srcPlane = src + z * srcAreaOffset * PACK_UNIT;
        for (int x = 0; x < areaC4; ++x) {
            auto s  = srcPlane + PACK_UNIT * PACK_UNIT * x;
            auto d  = dstPlane + PACK_UNIT * x;
            auto r0 = _mm256_loadu_ps(s + 0 * PACK_UNIT);
            auto r1 = _mm256_loadu_ps(s + 1 * PACK_UNIT);
            auto r2 = _mm256_loadu_ps(s + 2 * PACK_UNIT);
            auto r3 = _mm256_loadu_ps(s + 3 * PACK_UNIT);
            auto r4 = _mm256_loadu_ps(s + 4 * PACK_UNIT);
            auto r5 = _mm256_loadu_ps(s + 5 * PACK_UNIT);
            auto r6 = _mm256_loadu_ps(s + 6 * PACK_UNIT);
            auto r7 = _mm256_loadu_ps(s + 7 * PACK_UNIT);

            TRANSPOSE_8x8;

            _mm256_storeu_ps(d + 0 * dstAreaOffset, t0);
            _mm256_storeu_ps(d + 1 * dstAreaOffset, t1);
            _mm256_storeu_ps(d + 2 * dstAreaOffset, t2);
            _mm256_storeu_ps(d + 3 * dstAreaOffset, t3);
            _mm256_storeu_ps(d + 4 * dstAreaOffset, t4);
            _mm256_storeu_ps(d + 5 * dstAreaOffset, t5);
            _mm256_storeu_ps(d + 6 * dstAreaOffset, t6);
            _mm256_storeu_ps(d + 7 * dstAreaOffset, t7);
        }
    }
    auto areaRemain  = areaC4 * PACK_UNIT;
    auto depthRemain = depthC4 * PACK_UNIT;
    // Down
    int remain = depth - depthRemain;
    if (remain > 0) {
        float* dstPlane       = depthC4 * dstAreaOffset * PACK_UNIT + dst;
        const float* srcPlane = src + depthC4 * srcAreaOffset * PACK_UNIT;
        for (int x = 0; x < areaC4; ++x) {
            auto s  = srcPlane + PACK_UNIT * PACK_UNIT * x;
            auto d  = dstPlane + PACK_UNIT * x;
            auto r0 = _mm256_loadu_ps(s + 0 * PACK_UNIT);
            auto r1 = _mm256_loadu_ps(s + 1 * PACK_UNIT);
            auto r2 = _mm256_loadu_ps(s + 2 * PACK_UNIT);
            auto r3 = _mm256_loadu_ps(s + 3 * PACK_UNIT);
            auto r4 = _mm256_loadu_ps(s + 4 * PACK_UNIT);
            auto r5 = _mm256_loadu_ps(s + 5 * PACK_UNIT);
            auto r6 = _mm256_loadu_ps(s + 6 * PACK_UNIT);
            auto r7 = _mm256_loadu_ps(s + 7 * PACK_UNIT);

            TRANSPOSE_8x8;

            switch (remain) {
                case 7:
                    _mm256_storeu_ps(d + 6 * dstAreaOffset, t6);
                case 6:
                    _mm256_storeu_ps(d + 5 * dstAreaOffset, t5);
                case 5:
                    _mm256_storeu_ps(d + 4 * dstAreaOffset, t4);
                case 4:
                    _mm256_storeu_ps(d + 3 * dstAreaOffset, t3);
                case 3:
                    _mm256_storeu_ps(d + 2 * dstAreaOffset, t2);
                case 2:
                    _mm256_storeu_ps(d + 1 * dstAreaOffset, t1);
                case 1:
                    _mm256_storeu_ps(d + 0 * dstAreaOffset, t0);
                default:
                    break;
            }
        }
        for (int x = areaRemain; x < area; ++x) {
            for (int y = 0; y < remain; y++) {
                dstPlane[y * dstAreaOffset + x] = srcPlane[PACK_UNIT * x + y];
            }
        }
    }
    // Right
    for (int z = 0; z < depthC4; ++z) {
        const float* srcPlane = z * srcAreaOffset * PACK_UNIT + src;
        float* dstPlane       = dst + z * dstAreaOffset * PACK_UNIT;
        for (int x = areaRemain; x < area; ++x) {
            for (int y = 0; y < PACK_UNIT; y++) {
                dstPlane[y * dstAreaOffset + x] = srcPlane[PACK_UNIT * x + y];
            }
        }
    }
}
void _AVX_MNNPackCUnitTranspose(float* dst, const float* src, size_t area, size_t depth, int* areaOffset) {
    int c      = (int)depth;
    int cDiv4  = c / PACK_UNIT;
    int cAlign = cDiv4 * PACK_UNIT;
    auto srcAreaOffset = areaOffset[0];
    auto dstAreaOffset = areaOffset[1];
    for (int hi = 0; hi < area; ++hi) {
        const float* srcHeight = src + hi * c;
        float* dstHeight       = dst + hi * PACK_UNIT;
        for (int ci = 0; ci < cDiv4; ++ci) {
            _mm256_storeu_ps(dstHeight + PACK_UNIT * ci * dstAreaOffset, _mm256_loadu_ps(srcHeight + PACK_UNIT * ci));
        }
    }

    if (cAlign == c) {
        return;
    }

    int cReamin   = c - cAlign;
    auto srcAlign = src + cAlign;
    auto dstAlign = dst + dstAreaOffset * cAlign;

    for (int hi = 0; hi < area; ++hi) {
        const float* srcHeight = srcAlign + hi * c;
        float* dstHeight       = dstAlign + hi * PACK_UNIT;
        for (int i = 0; i < PACK_UNIT; ++i) {
            dstHeight[i] = 0;
        }
        for (int ci = 0; ci < cReamin; ++ci) {
            dstHeight[ci] = srcHeight[ci];
        }
    }

}
void _AVX_MNNUnpackCUnitTranspose(float* dst, const float* src, size_t area, size_t depth, int* areaOffset) {
    int c      = (int)depth;
    int cDiv4  = c / PACK_UNIT;
    int cAlign = cDiv4 * PACK_UNIT;
    auto srcAreaOffset = areaOffset[0];
    auto dstAreaOffset = areaOffset[1];
    for (int hi = 0; hi < area; ++hi) {
        const float* srcHeight = src + hi * PACK_UNIT;
        float* dstHeight       = dst + hi * c;
        for (int ci = 0; ci < cDiv4; ++ci) {
            _mm256_storeu_ps(dstHeight + PACK_UNIT * ci, _mm256_loadu_ps(srcHeight + PACK_UNIT * ci * srcAreaOffset));
        }
    }

    if (cAlign == c) {
        return;
    }

    int cReamin   = c - cAlign;
    auto srcAlign = src + srcAreaOffset * cAlign;
    auto dstAlign = dst + cAlign;

    for (int hi = 0; hi < area; ++hi) {
        const float* srcHeight = srcAlign + hi * PACK_UNIT;
        float* dstHeight       = dstAlign + hi * c;

        for (int ci = 0; ci < cReamin; ++ci) {
            dstHeight[ci] = srcHeight[ci];
        }
    }
}

void _AVX_MNNReluWithSlopeChannel(float* dst, const float* src, const float* slope, size_t sizeQuad, size_t depthQuad) {
    auto zero = _mm_set1_ps(0.0f);
    auto zero2 = _mm256_set1_ps(0.0f);
    int sizeC8 = sizeQuad;
    for (int j = 0; j < depthQuad; j++) {
        auto slopeZ       = _mm256_loadu_ps(slope + 8 * j);
        const float* srcZ = src + 8 * j * sizeQuad;
        float* dstZ       = dst + 8 * j * sizeQuad;
        for (int i = 0; i < sizeC8; i++) {
            auto src   = _mm256_loadu_ps(srcZ);
            auto mask0 = _mm256_cmp_ps(src, zero2, 0x01);
            auto mask1 = _mm256_cmp_ps(src, zero2, 0x0D);
            auto other = _mm256_mul_ps(src, slopeZ);
            _mm256_storeu_ps(dstZ, _mm256_add_ps(_mm256_and_ps(other, mask0), _mm256_and_ps(src, mask1)));
            srcZ += 8;
            dstZ += 8;
        }
    }
}

void _AVX_MNNGelu(float *dst, const float *src, size_t size) {
    auto var1 = _mm256_set1_ps(0.044715f);
    auto var2 = _mm256_set1_ps(0.79788458f);
    auto var3 = _mm256_set1_ps(378.f);
    auto var4 = _mm256_set1_ps(17325.f);
    auto var5 = _mm256_set1_ps(135135.f);
    auto var6 = _mm256_set1_ps(28.f);
    auto var7 = _mm256_set1_ps(3150.f);
    auto var8 = _mm256_set1_ps(62370.f);
    auto var9 = _mm256_set1_ps(135135.f);
    auto var10 = _mm256_set1_ps(0.5);
    auto varOne = _mm256_set1_ps(1.f);
    auto varNegOne = _mm256_set1_ps(-1.f);
    for (int i = 0; i < size; i++) {
        auto x = _mm256_loadu_ps(src + i * 8);
        auto y = _mm256_mul_ps(x, x);
        y = _mm256_mul_ps(y, x);
        y = _mm256_mul_ps(y, var1);
        y = _mm256_add_ps(y, x);
        y = _mm256_mul_ps(y, var2);
        // y = tanh(y)
        {
            auto y2 = _mm256_mul_ps(y, y);
            auto w = _mm256_add_ps(y2, var3);
            w = _mm256_mul_ps(w, y2);
            w = _mm256_add_ps(w, var4);
            w = _mm256_mul_ps(w, y2);
            w = _mm256_add_ps(w, var5);
            w = _mm256_mul_ps(w, y);
            auto z = _mm256_mul_ps(y2, var6);
            z = _mm256_add_ps(z, var7);
            z = _mm256_mul_ps(z, y2);
            z = _mm256_add_ps(z, var8);
            z = _mm256_mul_ps(z, y2);
            z = _mm256_add_ps(z, var9);
            z = _mm256_div_ps(w, z);
            z = _mm256_max_ps(z, varNegOne);
            y = _mm256_min_ps(z, varOne);
        }
        y = _mm256_add_ps(y, varOne);
        y = _mm256_mul_ps(y, x);
        y = _mm256_mul_ps(y, var10);
        _mm256_storeu_ps(dst + i * 8, y);
    }
}

void _AVX_MNNAxByClampBroadcastUnit(float* C, const float* A, const float* B, size_t width, size_t cStride, size_t aStride, size_t height, const float* parameters) {
    auto minF = _mm256_broadcast_ss(parameters + 2);
    auto maxF = _mm256_broadcast_ss(parameters + 3);
    for (int y = 0; y < height; ++y) {
        auto a = A + aStride * y;
        auto b = B + 8 * y;
        auto bv = _mm256_loadu_ps(b);
        auto c = C + cStride * y;
        for (int x = 0; x < width; ++x) {
            auto av = _mm256_loadu_ps(a);
            auto cv = _mm256_add_ps(av, bv);
            cv = _mm256_min_ps(cv, maxF);
            cv = _mm256_max_ps(cv, minF);
            _mm256_storeu_ps(c, cv);
            a += 8;
            c += 8;
        }
    }
}

void _AVX_MNNExpC8(float* dest, const float* source, const float* parameters, size_t countC8) {
    auto count = countC8;
    auto p0    = _mm256_set1_ps(parameters[0]);
    auto p1    = _mm256_set1_ps(parameters[1]);
    auto p2    = _mm256_set1_ps(parameters[2]);
    auto p3    = _mm256_set1_ps(parameters[3]);
    auto p4    = _mm256_set1_ps(parameters[4]);
    auto p5    = _mm256_set1_ps(parameters[5]);
    auto p6    = _mm256_set1_ps(parameters[6]);
    auto p7    = _mm256_set1_ps(parameters[7]);
    auto xMax  = _mm256_set1_ps(87);
    auto xMin  = _mm256_set1_ps(-87);
    auto basic = _mm256_set1_epi32(1 << 23);
    auto temp127 = _mm256_set1_epi32(127);
    auto negZero = _mm256_set1_ps(-0.f);
    for (int i = 0; i < count; ++i) {
        auto x            = _mm256_xor_ps(_mm256_loadu_ps(source + i * 8), negZero);
        x                 = _mm256_max_ps(x, xMin);
        x                 = _mm256_min_ps(x, xMax);
        auto div          = _mm256_mul_ps(x, p1);
        auto divInt       = _mm256_cvtps_epi32(div);
        div               = _mm256_cvtepi32_ps(divInt);
        auto div2         = _mm256_add_epi32(divInt, temp127);
        div2 = _mm256_mullo_epi32(div2, basic);
        auto expBasic  = _mm256_castsi256_ps(div2);
        auto xReamin   = _mm256_sub_ps(x, _mm256_mul_ps(div, p0));
        auto t         = xReamin;
        auto c0        = _mm256_mul_ps(p7, t);
        auto c1        = _mm256_add_ps(c0, p6);
        auto c2        = _mm256_mul_ps(c1, t);
        auto c3        = _mm256_add_ps(c2, p5);
        auto c4        = _mm256_mul_ps(c3, t);
        auto c5        = _mm256_add_ps(c4, p4);
        auto c6        = _mm256_mul_ps(c5, t);
        auto c7        = _mm256_add_ps(c6, p3);
        auto c8        = _mm256_mul_ps(c7, t);
        auto c9        = _mm256_add_ps(c8, p2);
        auto expRemain = c9;
        _mm256_storeu_ps(dest + 8 * i, _mm256_mul_ps(expBasic, expRemain));
    }
}

void _AVX_MNNSoftmax(float* dest, const float* source, size_t size) {
    float tmpfloat8[8];
    int count  = size / 8;
    int remain = count * 8;
    // step 1: get maxValue
    float maxValue = 0.f;
    if (count > 0) {
        auto maxVal = _mm256_loadu_ps(source);
        for (int i = 1; i < count; i++) {
            maxVal = _mm256_max_ps(maxVal, _mm256_loadu_ps(source + i * 8));
        }
        _mm256_storeu_ps(tmpfloat8, maxVal);
        maxValue = tmpfloat8[0] > tmpfloat8[1] ? tmpfloat8[0] : tmpfloat8[1];
        for (int i = 2; i < 8; i++) {
            maxValue = maxValue > tmpfloat8[i] ? maxValue : tmpfloat8[i];
        }
    }
    for (int i = remain; i < size; i++) {
        maxValue = maxValue > source[i] ? maxValue : source[i];
    }

    // step 2: get exp(x - maxValue) and sum(exp(x - maxValue))
    float sumValue = 0.f;
    if (count > 0) {
        auto sumVal = _mm256_set1_ps(0.f);
        auto p0    = _mm256_set1_ps(0.6931471805599453);
        auto p1    = _mm256_set1_ps(1.4426950408889634);
        auto p2    = _mm256_set1_ps(1.f);
        auto p3    = _mm256_set1_ps(1.f);
        auto p4    = _mm256_set1_ps(0.5);
        auto p5    = _mm256_set1_ps(0.1666666666666666);
        auto p6    = _mm256_set1_ps(0.041666666666666664);
        auto p7    = _mm256_set1_ps(0.008333333333333333);
        auto xMax  = _mm256_set1_ps(87);
        auto xMin  = _mm256_set1_ps(-87);
        auto basic = _mm256_set1_epi32(1 << 23);
        auto temp127 = _mm256_set1_epi32(127);
        for (int i = 0; i < count; ++i) {
            auto x            = _mm256_sub_ps(_mm256_loadu_ps(source + i * 8), _mm256_set1_ps(maxValue));
            x                 = _mm256_max_ps(x, xMin);
            x                 = _mm256_min_ps(x, xMax);
            auto div          = _mm256_mul_ps(x, p1);
            auto divInt       = _mm256_cvtps_epi32(div);
            div               = _mm256_cvtepi32_ps(divInt);
            auto div2         = _mm256_add_epi32(divInt, temp127);
            div2 = _mm256_mullo_epi32(div2, basic);
            auto expBasic  = _mm256_castsi256_ps(div2);
            auto xReamin   = _mm256_sub_ps(x, _mm256_mul_ps(div, p0));
            auto t         = xReamin;
            auto c0        = _mm256_mul_ps(p7, t);
            auto c1        = _mm256_add_ps(c0, p6);
            auto c2        = _mm256_mul_ps(c1, t);
            auto c3        = _mm256_add_ps(c2, p5);
            auto c4        = _mm256_mul_ps(c3, t);
            auto c5        = _mm256_add_ps(c4, p4);
            auto c6        = _mm256_mul_ps(c5, t);
            auto c7        = _mm256_add_ps(c6, p3);
            auto c8        = _mm256_mul_ps(c7, t);
            auto c9        = _mm256_add_ps(c8, p2);
            auto expRemain = c9;
            auto expRes    = _mm256_mul_ps(expBasic, expRemain);
            sumVal         = _mm256_add_ps(expRes, sumVal);
            _mm256_storeu_ps(dest + 8 * i, expRes);
        }
        _mm256_storeu_ps(tmpfloat8, sumVal);
        for (int i = 0; i < 8; i++) {
            sumValue += tmpfloat8[i];
        }
    }
    auto param = 0.6931471805599453;
    float xLimit = 87;
    for (int i = remain; i < size; i++) {
        auto x         = source[i] - maxValue;
        x = x > -xLimit ? x : -xLimit;
        x = x < xLimit ? x : xLimit;

        int div        = (x / param);
        int div2       = (div + 127) << 23;
        auto xReamin   = x - div * param;
        float expBasic = *(float*)(&div2);

        auto t         = xReamin;
        auto expRemain = ((((1.0f / 120 * t + 1.0f / 24) * t + 1.0f / 6) * t + 0.5f) * t + 1.0f) * t + 1.0f;
        dest[i]  = expBasic * expRemain;
        sumValue += dest[i];
    }
    // step 3: get x / sum and store
    for (int i = 0; i < count; ++i) {
        // using  1 / ((1 / x) * sum) instead x * (1 / sum) or x / sum for some bugs in intel cpu
        auto x = _mm256_rcp_ps(_mm256_loadu_ps(dest + 8 * i));
        auto y = _mm256_set1_ps(sumValue);
        auto z = _mm256_rcp_ps(_mm256_mul_ps(x, y));
        _mm256_storeu_ps(dest + 8 * i, z);
    }
    sumValue = 1.f / sumValue;
    for (int i = remain; i < size; i++) {
        dest[i] *= sumValue;
    }
}

void _AVX_MNNNorm(float *dst, const float *src, const float *gamma, const float *beta, float epsilon, size_t size) {
    float tmpfloat8[8];
    int count  = size / 8;
    int remain = count * 8;
    // step 1: get sum
    float sum = 0.f;
    if (count > 0) {
        auto sumVal = _mm256_set1_ps(0.f);
        for (int i = 0; i < count; i++) {
            sumVal = _mm256_add_ps(sumVal, _mm256_loadu_ps(src + i * 8));
        }
        _mm256_storeu_ps(tmpfloat8, sumVal);
        for (int i = 0; i < 8; i++) {
            sum += tmpfloat8[i];
        }
    }
    for (int i = remain; i < size; i++) {
        sum += src[i];
    }
    // step 2: get square_sum
    float mean = sum / size;
    float square_sum = 0.f;
    auto meanVal = _mm256_set1_ps(mean);
    if (count > 0) {
        auto sumVal = _mm256_set1_ps(0.f);
        for (int i = 0; i < count; i++) {
            auto x = _mm256_sub_ps(_mm256_loadu_ps(src + i * 8), meanVal);
            sumVal = _mm256_add_ps(sumVal, _mm256_mul_ps(x, x));
        }
        _mm256_storeu_ps(tmpfloat8, sumVal);
        for (int i = 0; i < 8; i++) {
            square_sum += tmpfloat8[i];
        }
    }
    for (int i = remain; i < size; i++) {
        float x = (src[i] - mean);
        square_sum += x * x;
    }
    // step 3: get result
    float variable = square_sum / size;
    variable = 1.f / std::sqrt(variable + epsilon);
    auto variableVal = _mm256_set1_ps(variable);
    if (gamma && beta) {
        for (int i = 0; i < count; i++) {
            auto x = _mm256_sub_ps(_mm256_loadu_ps(src + i * 8), meanVal);
            auto g = _mm256_loadu_ps(gamma + i * 8);
            auto b = _mm256_loadu_ps(beta + i * 8);
            auto y = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(x, g), variableVal), b);
            _mm256_storeu_ps(dst + i * 8, y);
        }
        for (int i = remain; i < size; i++) {
            dst[i] = (src[i] - mean) * gamma[i] * variable + beta[i] ;
        }
    } else {
        for (int i = 0; i < count; i++) {
            auto x = _mm256_sub_ps(_mm256_loadu_ps(src + i * 8), meanVal);
            auto y = _mm256_mul_ps(x, variableVal);
            _mm256_storeu_ps(dst + i * 8, y);
        }
        for (int i = remain; i < size; i++) {
            dst[i] = (src[i] - mean) * variable;
        }
    }
}

void _AVX_MNNFloat2Int8(const float* src, int8_t* dst, size_t sizeQuad, const float* scalep, ssize_t minV, ssize_t maxV, ssize_t zeroPoint) {
    auto zero = _mm256_set1_epi32(0);
    auto minValue = _mm256_set1_ps(minV);
    auto maxValue = _mm256_set1_ps(maxV);
    auto zeroPointValue = _mm256_set1_ps(zeroPoint);
    auto plus = _mm256_set1_ps(0.5f);
    auto minus = _mm256_set1_ps(-0.5f);
    auto sclaeVal = _mm_loadu_ps(scalep);
    auto scaleValue = _mm256_insertf128_ps(_mm256_castps128_ps256(sclaeVal), sclaeVal, 1);
    for (int i = 0; i < sizeQuad / 2; ++i) {
        auto f0 = _mm256_loadu_ps(src + 8 * i);
        f0 = _mm256_mul_ps(f0, scaleValue);
        f0 = _mm256_add_ps(f0, zeroPointValue);
        f0 = _mm256_min_ps(f0, maxValue);
        f0 = _mm256_max_ps(f0, minValue);
        auto m0 = _mm256_cmp_ps(f0, _mm256_castsi256_ps(zero), 1);
        m0 = _mm256_blendv_ps(plus, minus, m0);
        f0 = _mm256_add_ps(f0, m0);
        // 3: _MM_FROUND_TO_ZERO
        auto d0 = _mm256_cvtps_epi32(_mm256_round_ps(f0, 3));
        d0 = _mm256_packs_epi32(d0, _mm256_setzero_si256());
        d0 = _mm256_permute4x64_epi64(d0, 0xD8);
#if defined(_MSC_VER)
        __m256i x = static_cast<__m256i>(_mm256_packs_epi16(d0, _mm256_setzero_si256()));
        *((int64_t*)dst + i) = x.m256i_i64[0];
#else
        __v4di x = static_cast<__v4di>(_mm256_packs_epi16(d0, _mm256_setzero_si256()));
        *((int64_t*)dst + i) = x[0];
#endif
    }
    if (sizeQuad % 2) {
        unsigned int offset = sizeQuad * 4 - 4;
        _SSE_MNNFloat2Int8(src + offset, dst + offset, 1, scalep, minV, maxV, zeroPoint);
    }
}

void _AVX_MNNInt8ScaleToFloat(float* dst, const int8_t* src, const float* scale, size_t sizeQuad, ssize_t zeroPoint) {
    auto sizeC4 = sizeQuad / 8;
    auto sizeRemain = sizeQuad % 8;
    auto zero = _mm256_set1_epi32(0);
    auto sclaeVal = _mm_loadu_ps(scale);
    auto scaleValue = _mm256_insertf128_ps(_mm256_castps128_ps256(sclaeVal), sclaeVal, 1);
    auto zeroPointValue = _mm256_set1_epi32(zeroPoint);
    for (int i = 0; i < sizeC4; ++i) {
        auto s = _mm256_castps_si256(_mm256_loadu_ps((const float*)(src)));
        auto s0_16 = _mm256_permute4x64_epi64(_mm256_srai_epi16(_mm256_unpacklo_epi8(zero, s), 8), 0XD8);
        auto s1_16 = _mm256_permute4x64_epi64(_mm256_srai_epi16(_mm256_unpackhi_epi8(zero, s), 8), 0xD8);
        auto s0_32 = _mm256_srai_epi32(_mm256_unpacklo_epi16(zero, s0_16), 16);
        auto s1_32 = _mm256_srai_epi32(_mm256_unpacklo_epi16(zero, s1_16), 16);
        auto s2_32 = _mm256_srai_epi32(_mm256_unpackhi_epi16(zero, s0_16), 16);
        auto s3_32 = _mm256_srai_epi32(_mm256_unpackhi_epi16(zero, s1_16), 16);
        s0_32 = _mm256_sub_epi32(s0_32, zeroPointValue);
        s1_32 = _mm256_sub_epi32(s1_32, zeroPointValue);
        s2_32 = _mm256_sub_epi32(s2_32, zeroPointValue);
        s3_32 = _mm256_sub_epi32(s3_32, zeroPointValue);
        auto s0_f = _mm256_cvtepi32_ps(s0_32);
        auto s1_f = _mm256_cvtepi32_ps(s1_32);
        auto s2_f = _mm256_cvtepi32_ps(s2_32);
        auto s3_f = _mm256_cvtepi32_ps(s3_32);
        _mm256_storeu_ps(dst + 8 * 0, _mm256_mul_ps(s0_f, scaleValue));
        _mm256_storeu_ps(dst + 8 * 1, _mm256_mul_ps(s1_f, scaleValue));
        _mm256_storeu_ps(dst + 8 * 2, _mm256_mul_ps(s2_f, scaleValue));
        _mm256_storeu_ps(dst + 8 * 3, _mm256_mul_ps(s3_f, scaleValue));
        src += 32;
        dst += 32;
    }
    if (sizeRemain > 0) {
        _SSE_MNNInt8ScaleToFloat(dst, src, scale, sizeRemain, zeroPoint);
    }
}


void _AVX_MNNConvRunForUnitDepthWise(float* dst, const float* src, const float* weight, size_t fw, size_t fh,
                                  size_t weight_y_step, size_t dilateX_step, size_t dilateY_step) {
    int fx, fy;
    __m256 dstValue = _mm256_setzero_ps();
    const float* src_z    = src;
    const float* weight_z = weight;
    for (fy = 0; fy < fh; ++fy) {
        const float* src_y    = src_z + fy * dilateY_step;
        const float* weight_y = weight_z + fy * weight_y_step;
        for (fx = 0; fx < fw; ++fx) {
            const float* weight_x = weight_y + 8 * fx;
            const float* src_x    = src_y + fx * dilateX_step;
            dstValue = _mm256_add_ps(dstValue, _mm256_mul_ps(_mm256_loadu_ps(src_x), _mm256_loadu_ps(weight_x)));
        }
    }
    _mm256_storeu_ps(dst, dstValue);
}

void _AVX_MNNConvRunForLineDepthwise(float* dst, const float* src, const float* weight, size_t width, size_t src_w_setup,
                                size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step, size_t height,
                                     size_t srcHStep, size_t dstHStep) {
    int dx, fx, fy;
    const int unit = 4;
    int widthUnit = width / unit;
    int widthRemain = width - widthUnit * unit;
    const float* weight_z = weight;
    if (src_w_setup == 8) {
        for (int y = 0; y < height; ++y) {
            auto srcY = src + y * srcHStep;
            auto dstY = dst + y * dstHStep;
            for (dx = 0; dx < widthUnit; ++dx) {
                auto dstValue0 = _mm256_setzero_ps();
                auto dstValue1 = _mm256_setzero_ps();
                auto dstValue2 = _mm256_setzero_ps();
                auto dstValue3 = _mm256_setzero_ps();
                for (fy = 0; fy < fh; ++fy) {
                    const float* src_y    = srcY + fy * dilateY_step;
                    const float* weight_y = weight_z + fy * fw * 8;
                    for (fx = 0; fx < fw; ++fx) {
                        const float* src_x    = src_y + fx * dilateX_step;
                        const float* weight_x = weight_y + 8 * fx;
                        auto weightValue = _mm256_loadu_ps(weight_x);
                        dstValue0 = _mm256_add_ps(dstValue0, _mm256_mul_ps(_mm256_loadu_ps(src_x + 0 * 8), weightValue));
                        dstValue1 = _mm256_add_ps(dstValue1, _mm256_mul_ps(_mm256_loadu_ps(src_x + 1 * 8), weightValue));
                        dstValue2 = _mm256_add_ps(dstValue2, _mm256_mul_ps(_mm256_loadu_ps(src_x + 2 * 8), weightValue));
                        dstValue3 = _mm256_add_ps(dstValue3, _mm256_mul_ps(_mm256_loadu_ps(src_x + 3 * 8), weightValue));
                    }
                }
                _mm256_storeu_ps(dstY + 8 * 0, dstValue0);
                _mm256_storeu_ps(dstY + 8 * 1, dstValue1);
                _mm256_storeu_ps(dstY + 8 * 2, dstValue2);
                _mm256_storeu_ps(dstY + 8 * 3, dstValue3);
                dstY += 8 * unit;
                srcY += unit * src_w_setup;
            }
            for (dx = 0; dx < widthRemain; ++dx) {
                float* dst_x          = dstY + dx * 8;
                auto dstValue = _mm256_setzero_ps();
                const float* src_z    = srcY + src_w_setup * dx;
                const float* weight_z = weight;
                for (fy = 0; fy < fh; ++fy) {
                    const float* src_y    = src_z + fy * dilateY_step;
                    const float* weight_y = weight_z + fy * fw * 8;
                    for (fx = 0; fx < fw; ++fx) {
                        const float* weight_x = weight_y + 8 * fx;
                        const float* src_x    = src_y + fx * dilateX_step;
                        dstValue = _mm256_add_ps(dstValue, _mm256_mul_ps(_mm256_loadu_ps(src_x), _mm256_loadu_ps(weight_x)));
                    }
                }
                _mm256_storeu_ps(dst_x, dstValue);
            }
        }
        return;
    }
    for (int y = 0; y < height; ++y) {
        auto srcY = src + y * srcHStep;
        auto dstY = dst + y * dstHStep;
        for (dx = 0; dx < widthUnit; ++dx) {
            auto dstValue0 = _mm256_setzero_ps();
            auto dstValue1 = _mm256_setzero_ps();
            auto dstValue2 = _mm256_setzero_ps();
            auto dstValue3 = _mm256_setzero_ps();
            for (fy = 0; fy < fh; ++fy) {
                const float* src_y    = srcY + fy * dilateY_step;
                const float* weight_y = weight_z + fy * fw * 8;
                for (fx = 0; fx < fw; ++fx) {
                    const float* src_x    = src_y + fx * dilateX_step;
                    const float* weight_x = weight_y + 8 * fx;
                    auto weightValue = _mm256_loadu_ps(weight_x);
                    dstValue0 = _mm256_add_ps(dstValue0, _mm256_mul_ps(_mm256_loadu_ps(src_x + 0 * src_w_setup), weightValue));
                    dstValue1 = _mm256_add_ps(dstValue1, _mm256_mul_ps(_mm256_loadu_ps(src_x + 1 * src_w_setup), weightValue));
                    dstValue2 = _mm256_add_ps(dstValue2, _mm256_mul_ps(_mm256_loadu_ps(src_x + 2 * src_w_setup), weightValue));
                    dstValue3 = _mm256_add_ps(dstValue3, _mm256_mul_ps(_mm256_loadu_ps(src_x + 3 * src_w_setup), weightValue));
                }
            }
            _mm256_storeu_ps(dstY + 8 * 0, dstValue0);
            _mm256_storeu_ps(dstY + 8 * 1, dstValue1);
            _mm256_storeu_ps(dstY + 8 * 2, dstValue2);
            _mm256_storeu_ps(dstY + 8 * 3, dstValue3);
            dstY += 8 * unit;
            srcY += unit * src_w_setup;
        }
        for (dx = 0; dx < widthRemain; ++dx) {
            float* dst_x          = dstY + dx * 8;
            auto dstValue = _mm256_setzero_ps();
            const float* src_z    = srcY + src_w_setup * dx;
            const float* weight_z = weight;
            for (fy = 0; fy < fh; ++fy) {
                const float* src_y    = src_z + fy * dilateY_step;
                const float* weight_y = weight_z + fy * fw * 8;
                for (fx = 0; fx < fw; ++fx) {
                    const float* weight_x = weight_y + 8 * fx;
                    const float* src_x    = src_y + fx * dilateX_step;
                    dstValue = _mm256_add_ps(dstValue, _mm256_mul_ps(_mm256_loadu_ps(src_x), _mm256_loadu_ps(weight_x)));
                }
            }
            _mm256_storeu_ps(dst_x, dstValue);
        }
    }
}

void _AVX_MNNMultiAndDestTransformCommon23(float **cacheLine, const float *weigth, float *dest, int cacheLineSize, int ow, const float* bias, const float* parameter) {
    int unit = ow / 2;
    MNN_ASSERT(cacheLineSize >= 1);
    auto biasF = Vec8::load(bias);
    auto minF = Vec8(parameter[2]);
    auto maxF = Vec8(parameter[3]);
    for (int x = 0; x < unit; ++x) {
        auto offset = 4 * 8 * x;
        int i = 0;
        Vec8 m0     = Vec8::load(weigth + i * 32 + 8 * 0) * Vec8::load(cacheLine[i] + offset + 8 * 0);
        Vec8 m1     = Vec8::load(weigth + i * 32 + 8 * 1) * Vec8::load(cacheLine[i] + offset + 8 * 1);
        Vec8 m2     = Vec8::load(weigth + i * 32 + 8 * 2) * Vec8::load(cacheLine[i] + offset + 8 * 2);
        Vec8 m3     = Vec8::load(weigth + i * 32 + 8 * 3) * Vec8::load(cacheLine[i] + offset + 8 * 3);

        for (i = 1; i < cacheLineSize; ++i) {
            m0 = m0 + Vec8::load(weigth + i * 32 + 8 * 0) * Vec8::load(cacheLine[i] + offset + 8 * 0);
            m1 = m1 + Vec8::load(weigth + i * 32 + 8 * 1) * Vec8::load(cacheLine[i] + offset + 8 * 1);
            m2 = m2 + Vec8::load(weigth + i * 32 + 8 * 2) * Vec8::load(cacheLine[i] + offset + 8 * 2);
            m3 = m3 + Vec8::load(weigth + i * 32 + 8 * 3) * Vec8::load(cacheLine[i] + offset + 8 * 3);
        }
        auto o0 = m0 + m1 + m2 + biasF;
        auto o1 = m1 - m2 + m3 + biasF;
        o0 = Vec8::min(maxF, o0);
        o1 = Vec8::min(maxF, o1);
        o0 = Vec8::max(minF, o0);
        o1 = Vec8::max(minF, o1);

        Vec8::save(dest + 16 * x + 0 * 8, o0);
        Vec8::save(dest + 16 * x + 1 * 8, o1);
    }
    if (unit * 2 < ow) {
        auto offset = 8 * 4 * unit;
        int i = 0;
        Vec8 m0     = Vec8::load(weigth + i * 32 + 8 * 0) * Vec8::load(cacheLine[i] + offset + 8 * 0);
        Vec8 m1     = Vec8::load(weigth + i * 32 + 8 * 1) * Vec8::load(cacheLine[i] + offset + 8 * 1);
        Vec8 m2     = Vec8::load(weigth + i * 32 + 8 * 2) * Vec8::load(cacheLine[i] + offset + 8 * 2);

        for (i = 1; i < cacheLineSize; ++i) {
            m0 = m0 + Vec8::load(weigth + i * 32 + 8 * 0) * Vec8::load(cacheLine[i] + offset + 8 * 0);
            m1 = m1 + Vec8::load(weigth + i * 32 + 8 * 1) * Vec8::load(cacheLine[i] + offset + 8 * 1);
            m2 = m2 + Vec8::load(weigth + i * 32 + 8 * 2) * Vec8::load(cacheLine[i] + offset + 8 * 2);
        }
        auto o0 = m0 + m1 + m2 + biasF;
        o0 = Vec8::min(maxF, o0);
        o0 = Vec8::max(minF, o0);
        Vec8::save(dest + 16 * unit + 0 * 8, o0);
    }
}
static void _AVX_MNNConvDwF23SourceTransUnit(const float *source, float *dest, size_t unit) {
    if (unit <= 0) {
        return;
    }
    Vec8 v0 = Vec8::load(source + 8 * 0);
    Vec8 v1 = Vec8::load(source + 8 * 1);
    Vec8 v2;
    Vec8 v3;
    source += 16;

    for (int x = 0; x < unit; ++x) {
        v2 = Vec8::load(source + 0 * 8);
        v3 = Vec8::load(source + 1 * 8);
        auto m0 = v0 - v2;
        auto m1 = v1 + v2;
        auto m2 = v2 - v1;
        auto m3 = v3 - v1;

        Vec8::save(dest + 8 * 0, m0);
        Vec8::save(dest + 8 * 1, m1);
        Vec8::save(dest + 8 * 2, m2);
        Vec8::save(dest + 8 * 3, m3);

        source += 16;
        dest += 32;

        v0 = v2;
        v1 = v3;
    }
}

void _AVX_MNNSourceTransformCommonF23(const float *source, float *dest, int unit, int iw, int pad, int su, int eu) {
    for (int x = 0; x < su; ++x) {
        auto dstX = dest + 4 * 8 * x;
        auto sx   = x * 2 - (int)pad;
        auto ex   = sx + 4;

        auto clampSx = std::max(sx, 0);
        auto clampEx = std::min(ex, (int)iw);

        Vec8 v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int i = clampSx; i < clampEx; ++i) {
            v[i - sx] = Vec8::load(source + 8 * i);
        }
        auto m0 = v[0] - v[2];
        auto m1 = v[1] + v[2];
        auto m2 = v[2] - v[1];
        auto m3 = v[3] - v[1];

        Vec8::save(dstX + 8 * 0, m0);
        Vec8::save(dstX + 8 * 1, m1);
        Vec8::save(dstX + 8 * 2, m2);
        Vec8::save(dstX + 8 * 3, m3);
    }
    _AVX_MNNConvDwF23SourceTransUnit(source + 8 * (su * 2 - pad), dest + 8 * 4 * su, eu - su);

    for (int x = eu; x < unit; ++x) {
        auto dstX = dest + 8 * 4 * x;
        auto sx   = x * 2 - (int)pad;
        auto ex   = sx + 4;

        auto clampSx = std::max(sx, 0);
        auto clampEx = std::min(ex, (int)iw);

        Vec8 v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int i = clampSx; i < clampEx; ++i) {
            v[i - sx] = Vec8::load(source + 8 * i);
        }
        auto m0 = v[0] - v[2];
        auto m1 = v[1] + v[2];
        auto m2 = v[2] - v[1];
        auto m3 = v[3] - v[1];

        Vec8::save(dstX + 8 * 0, m0);
        Vec8::save(dstX + 8 * 1, m1);
        Vec8::save(dstX + 8 * 2, m2);
        Vec8::save(dstX + 8 * 3, m3);
    }
}

void _AVX_MNNConvDwF23MulTransUnit(float **cacheLine, const float *weigth, float *dest, size_t ow, const float* bias, const float* parameter) {
    int unit = ow / 2;
    auto w00 = Vec8::load(weigth + 0 * 32 + 8 * 0);
    auto w01 = Vec8::load(weigth + 0 * 32 + 8 * 1);
    auto w02 = Vec8::load(weigth + 0 * 32 + 8 * 2);
    auto w03 = Vec8::load(weigth + 0 * 32 + 8 * 3);
    auto w10 = Vec8::load(weigth + 1 * 32 + 8 * 0);
    auto w11 = Vec8::load(weigth + 1 * 32 + 8 * 1);
    auto w12 = Vec8::load(weigth + 1 * 32 + 8 * 2);
    auto w13 = Vec8::load(weigth + 1 * 32 + 8 * 3);
    auto w20 = Vec8::load(weigth + 2 * 32 + 8 * 0);
    auto w21 = Vec8::load(weigth + 2 * 32 + 8 * 1);
    auto w22 = Vec8::load(weigth + 2 * 32 + 8 * 2);
    auto w23 = Vec8::load(weigth + 2 * 32 + 8 * 3);
    auto biasF = Vec8::load(bias);
    auto minF = Vec8(parameter[2]);
    auto maxF = Vec8(parameter[3]);

    for (int x = 0; x < unit; ++x) {
        auto offset = 8 * 4 * x;
        int i = 0;
        Vec8 m0     = w00 * Vec8::load(cacheLine[0] + offset + 8 * 0);
        Vec8 m1     = w01 * Vec8::load(cacheLine[0] + offset + 8 * 1);
        Vec8 m2     = w02 * Vec8::load(cacheLine[0] + offset + 8 * 2);
        Vec8 m3     = w03 * Vec8::load(cacheLine[0] + offset + 8 * 3);

        m0 = m0 + w10 * Vec8::load(cacheLine[1] + offset + 8 * 0);
        m1 = m1 + w11 * Vec8::load(cacheLine[1] + offset + 8 * 1);
        m2 = m2 + w12 * Vec8::load(cacheLine[1] + offset + 8 * 2);
        m3 = m3 + w13 * Vec8::load(cacheLine[1] + offset + 8 * 3);

        m0 = m0 + w20 * Vec8::load(cacheLine[2] + offset + 8 * 0);
        m1 = m1 + w21 * Vec8::load(cacheLine[2] + offset + 8 * 1);
        m2 = m2 + w22 * Vec8::load(cacheLine[2] + offset + 8 * 2);
        m3 = m3 + w23 * Vec8::load(cacheLine[2] + offset + 8 * 3);

        auto o0 = m0 + m1 + m2 + biasF;
        auto o1 = m1 - m2 + m3 + biasF;
        o0 = Vec8::min(maxF, o0);
        o1 = Vec8::min(maxF, o1);
        o0 = Vec8::max(minF, o0);
        o1 = Vec8::max(minF, o1);
        Vec8::save(dest + 16 * x + 0 * 8, o0);
        Vec8::save(dest + 16 * x + 1 * 8, o1);
    }
    if (unit * 2 < ow) {
        auto offset = 8 * 4 * unit;
        Vec8 m0     = w00 * Vec8::load(cacheLine[0] + offset + 8 * 0);
        Vec8 m1     = w01 * Vec8::load(cacheLine[0] + offset + 8 * 1);
        Vec8 m2     = w02 * Vec8::load(cacheLine[0] + offset + 8 * 2);

        m0 = m0 + w10 * Vec8::load(cacheLine[1] + offset + 8 * 0);
        m1 = m1 + w11 * Vec8::load(cacheLine[1] + offset + 8 * 1);
        m2 = m2 + w12 * Vec8::load(cacheLine[1] + offset + 8 * 2);

        m0 = m0 + w20 * Vec8::load(cacheLine[2] + offset + 8 * 0);
        m1 = m1 + w21 * Vec8::load(cacheLine[2] + offset + 8 * 1);
        m2 = m2 + w22 * Vec8::load(cacheLine[2] + offset + 8 * 2);
        auto o0 = m0 + m1 + m2 + biasF;
        o0 = Vec8::min(maxF, o0);
        o0 = Vec8::max(minF, o0);
        Vec8::save(dest + 16 * unit + 0 * 8, o0);
    }
}

static MNNBinaryExecute _AVX2_MNNSelectBinaryFunctionForFloat(int opType) {
    auto vecF = MNN::selectVector<Vec8, 8>(opType);
    if (nullptr != vecF) {
        return vecF;
    }
    return MNN::MNNGetCoreFunctions()->MNNSelectBinaryFunctionForFloat(opType);
}


void _AVX_ExtraInit(void* functions) {
    auto coreFunction = static_cast<MNN::CoreFunctions*>(functions);
    coreFunction->MNNPoolingAvg = (decltype(coreFunction->MNNPoolingAvg))(MNN::poolingAvg<float, Vec8, 8>);
    // Set min value as 1 << 24
    coreFunction->MNNPoolingMax = (decltype(coreFunction->MNNPoolingMax))(MNN::poolingMax<float, Vec8, 8, -16777216>);
    coreFunction->MNNSelectBinaryFunctionForFloat = _AVX2_MNNSelectBinaryFunctionForFloat;
}

void _AVX_MNNScaleAndAddBias(float* dst, const float* src, const float* bias, const float* alpha, size_t planeNumber,
                        size_t biasNumber) {
    for (int z = 0; z < biasNumber; ++z) {
        float* dstZ         = dst + planeNumber * 8 * z;
        const float* srcZ   = src + planeNumber * 8 * z;
        auto biasZ = Vec8::load(bias + 8 * z);
        auto alphaZ = Vec8::load(alpha + 8 * z);
        for (int p = 0; p < planeNumber; ++p) {
            float* dstX       = dstZ + 8 * p;
            const float* srcX = srcZ + 8 * p;
            Vec8::save(dstX, (Vec8::load(srcX) * alphaZ) + biasZ);
        }
    }
}

void _AVX_MNNDeconvRunForUnitDepthWise(const float* dst, float* src, const float* weight, size_t fw, size_t fh,
                                  size_t weight_y_step, size_t dilateX_step, size_t dilateY_step) {
    int fx, fy;
    float* src_z          = src;
    const float* weight_z = weight;
    Vec8 dstV             = Vec8::load(dst);
    for (fy = 0; fy < fh; ++fy) {
        float* src_y          = src_z + fy * dilateY_step;
        const float* weight_y = weight_z + fy * weight_y_step;
        for (fx = 0; fx < fw; ++fx) {
            Vec8 weight_x = Vec8::load(weight_y + 8 * fx);
            Vec8 src_x    = Vec8::load(src_y + fx * dilateX_step);
            Vec8::save(src_y + fx * dilateX_step, src_x + weight_x * dstV);
        }
    }
}
void _AVX_MNNDeconvRunForLineDepthwise(const float* dst, float* src, const float* weight, size_t width, size_t src_w_setup,
                                  size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step) {
    int dx;
    for (dx = 0; dx < width; ++dx) {
        const float* dst_x = dst + dx * 8;
        float* src_dx      = src + src_w_setup * dx;
        _AVX_MNNDeconvRunForUnitDepthWise(dst_x, src_dx, weight, fw, fh, fw * 8, dilateX_step, dilateY_step);
    }
}

static __m256 MNNGridSampleLoadSample(int h, int w, const float *buffer, int height, int width, bool padMode) {
    if (h < 0 || h >= height || w < 0 || w >= width) {
        if(padMode == true) { //padMode == BorderMode_ZEROS
            return _mm256_setzero_ps();
        }
        // Clearly, CLAMP is the right way to go for GridSamplePaddingMode_BORDER
        // For GridSamplePaddingMode_REFLECTION, since we have reflected the values into (-1, 1),
        // the leftover reflections degrade to GridSamplePaddingMode_BORDER
        h = h < 0 ? 0 : ( h > (height - 1) ? (height - 1) : h);
        w = w < 0 ? 0 : ( w > (width - 1) ? (width - 1) : w);
    }

    return _mm256_loadu_ps(buffer + h * width * 8 + w * 8);
}
void _AVX_MNNGridSampleInterp(float* outputPtr, const float* inputPtr, const float* cordPtr, size_t inH, size_t inW, size_t outW, bool sampleMode, bool padMode) {
    for (auto ow = 0; ow < outW; ++ow) {
        auto w = cordPtr[2 * ow + 0];
        auto h = cordPtr[2 * ow + 1];
        __m256 interp;

        if (sampleMode == true) { //sampleMode == SampleMode_NEAREST
            int nh = ::floor(h + 0.5f);
            int nw = ::floor(w + 0.5f);
            interp = MNNGridSampleLoadSample(nh, nw, inputPtr, inH, inW, padMode);
        } else { //sampleMode == GridSampleMode_BILINEAR
            int w0_h = ::floor(h);
            int w0_w = ::floor(w);
            int w1_h = ::ceil(h);
            int w1_w = ::ceil(w);
            auto oneV = _mm256_set1_ps(1.0f);

            __m256 i00 = MNNGridSampleLoadSample(w0_h, w0_w, inputPtr, inH, inW, padMode);
            __m256 i01 = MNNGridSampleLoadSample(w0_h, w1_w, inputPtr, inH, inW, padMode);
            __m256 i10 = MNNGridSampleLoadSample(w1_h, w0_w, inputPtr, inH, inW, padMode);
            __m256 i11 = MNNGridSampleLoadSample(w1_h, w1_w, inputPtr, inH, inW, padMode);
            auto f0 = _mm256_set1_ps((float)w1_w - w);
            auto f1 = _mm256_sub_ps(oneV, f0);
            auto h0 = _mm256_set1_ps((float)w1_h - h);
            auto h1 = _mm256_sub_ps(oneV, h0);

            __m256 i0 = _mm256_add_ps(_mm256_mul_ps(i00, f0), _mm256_mul_ps(i01, f1));
            __m256 i1 = _mm256_add_ps(_mm256_mul_ps(i10, f0), _mm256_mul_ps(i11, f1));
            interp = _mm256_add_ps(_mm256_mul_ps(i0, h0), _mm256_mul_ps(i1, h1));
        }

        _mm256_storeu_ps(outputPtr + 8 * ow, interp);
    }
}
