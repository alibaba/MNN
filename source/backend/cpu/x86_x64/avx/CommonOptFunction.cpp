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
#include <limits>
#include <vector>
#include "FunctionSummary.hpp"
#include "core/Macro.h"
#include "backend/cpu/CPUPool.hpp"
#include "backend/cpu/BinaryUtils.hpp"
#include "Vec8.hpp"

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
void _AVX_MNNPackCUnit(float* dst, const float* src, size_t area, size_t depth) {
    auto areaC4  = area / PACK_UNIT;
    auto depthC4 = depth / PACK_UNIT;
    __m256 t0, t1, t2, t3, t4, t5, t6, t7;
    for (int z = 0; z < depthC4; ++z) {
        auto dstPlane = dst + z * area * PACK_UNIT;
        auto srcPlane = src + z * area * PACK_UNIT;
        for (int x = 0; x < areaC4; ++x) {
            auto s  = srcPlane + PACK_UNIT * x;
            auto d  = dstPlane + PACK_UNIT * PACK_UNIT * x;
            auto r0 = _mm256_loadu_ps(s + 0 * area);
            auto r1 = _mm256_loadu_ps(s + 1 * area);
            auto r2 = _mm256_loadu_ps(s + 2 * area);
            auto r3 = _mm256_loadu_ps(s + 3 * area);
            auto r4 = _mm256_loadu_ps(s + 4 * area);
            auto r5 = _mm256_loadu_ps(s + 5 * area);
            auto r6 = _mm256_loadu_ps(s + 6 * area);
            auto r7 = _mm256_loadu_ps(s + 7 * area);
            
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
        float* dstPlane       = depthC4 * area * PACK_UNIT + dst;
        const float* srcPlane = src + depthC4 * area * PACK_UNIT;
        {
            for (int x = 0; x < areaC4; ++x) {
                auto s  = srcPlane + PACK_UNIT * x;
                auto d  = dstPlane + PACK_UNIT * PACK_UNIT * x;
                auto r0 = _mm256_loadu_ps(s + 0 * area);
                auto r1 = _mm256_setzero_ps();
                auto r2 = _mm256_setzero_ps();
                auto r3 = _mm256_setzero_ps();
                auto r4 = _mm256_setzero_ps();
                auto r5 = _mm256_setzero_ps();
                auto r6 = _mm256_setzero_ps();
                auto r7 = _mm256_setzero_ps();
                switch (remain) {
                    case 7:
                        r6 = _mm256_loadu_ps(s + 6 * area);
                    case 6:
                        r5 = _mm256_loadu_ps(s + 5 * area);
                    case 5:
                        r4 = _mm256_loadu_ps(s + 4 * area);
                    case 4:
                        r3 = _mm256_loadu_ps(s + 3 * area);
                    case 3:
                        r2 = _mm256_loadu_ps(s + 2 * area);
                    case 2:
                        r1 = _mm256_loadu_ps(s + 1 * area);
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
                dstPlane[PACK_UNIT * x + y] = srcPlane[y * area + x];
            }
            for (int y = remain; y < PACK_UNIT; y++) {
                dstPlane[PACK_UNIT * x + y] = 0;
            }
        }
    }
    // Right
    for (int z = 0; z < depthC4; ++z) {
        float* dstPlane       = z * area * PACK_UNIT + dst;
        const float* srcPlane = src + z * area * PACK_UNIT;
        for (int x = areaRemain; x < area; ++x) {
            float s0 = srcPlane[x];
            float s1 = srcPlane[x + area];
            float s2 = srcPlane[x + area * 2];
            float s3 = srcPlane[x + area * 3];
            float s4 = srcPlane[x + area * 4];
            float s5 = srcPlane[x + area * 5];
            float s6 = srcPlane[x + area * 6];
            float s7 = srcPlane[x + area * 7];
            _mm256_storeu_ps(dstPlane + PACK_UNIT * x, _mm256_set_ps(s7, s6, s5, s4, s3, s2, s1, s0));
        }
    }
}
void _AVX_MNNUnpackCUnit(float* dst, const float* src, size_t area, size_t depth) {
    auto areaC4  = area / PACK_UNIT;
    auto depthC4 = depth / PACK_UNIT;
    __m256 t0, t1, t2, t3, t4, t5, t6, t7;
    for (int z = 0; z < depthC4; ++z) {
        auto dstPlane = dst + z * area * PACK_UNIT;
        auto srcPlane = src + z * area * PACK_UNIT;
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

            _mm256_storeu_ps(d + 0 * area, t0);
            _mm256_storeu_ps(d + 1 * area, t1);
            _mm256_storeu_ps(d + 2 * area, t2);
            _mm256_storeu_ps(d + 3 * area, t3);
            _mm256_storeu_ps(d + 4 * area, t4);
            _mm256_storeu_ps(d + 5 * area, t5);
            _mm256_storeu_ps(d + 6 * area, t6);
            _mm256_storeu_ps(d + 7 * area, t7);
        }
    }
    auto areaRemain  = areaC4 * PACK_UNIT;
    auto depthRemain = depthC4 * PACK_UNIT;
    // Down
    int remain = depth - depthRemain;
    if (remain > 0) {
        float* dstPlane       = depthC4 * area * PACK_UNIT + dst;
        const float* srcPlane = src + depthC4 * area * PACK_UNIT;
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
                    _mm256_storeu_ps(d + 6 * area, t6);
                case 6:
                    _mm256_storeu_ps(d + 5 * area, t5);
                case 5:
                    _mm256_storeu_ps(d + 4 * area, t4);
                case 4:
                    _mm256_storeu_ps(d + 3 * area, t3);
                case 3:
                    _mm256_storeu_ps(d + 2 * area, t2);
                case 2:
                    _mm256_storeu_ps(d + 1 * area, t1);
                case 1:
                    _mm256_storeu_ps(d + 0 * area, t0);
                default:
                    break;
            }
        }
        for (int x = areaRemain; x < area; ++x) {
            for (int y = 0; y < remain; y++) {
                dstPlane[y * area + x] = srcPlane[PACK_UNIT * x + y];
            }
        }
    }
    // Right
    for (int z = 0; z < depthC4; ++z) {
        const float* srcPlane = z * area * PACK_UNIT + src;
        float* dstPlane       = dst + z * area * PACK_UNIT;
        for (int x = areaRemain; x < area; ++x) {
            for (int y = 0; y < PACK_UNIT; y++) {
                dstPlane[y * area + x] = srcPlane[PACK_UNIT * x + y];
            }
        }
    }
}
void _AVX_MNNPackCUnitTranspose(float* dst, const float* src, size_t area, size_t depth) {
    int c      = (int)depth;
    int cDiv4  = c / PACK_UNIT;
    int cAlign = cDiv4 * PACK_UNIT;
    for (int hi = 0; hi < area; ++hi) {
        const float* srcHeight = src + hi * c;
        float* dstHeight       = dst + hi * PACK_UNIT;
        for (int ci = 0; ci < cDiv4; ++ci) {
            _mm256_storeu_ps(dstHeight + PACK_UNIT * ci * area, _mm256_loadu_ps(srcHeight + PACK_UNIT * ci));
        }
    }

    if (cAlign == c) {
        return;
    }

    int cReamin   = c - cAlign;
    auto srcAlign = src + cAlign;
    auto dstAlign = dst + area * cAlign;

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
void _AVX_MNNUnpackCUnitTranspose(float* dst, const float* src, size_t area, size_t depth) {
    int c      = (int)depth;
    int cDiv4  = c / PACK_UNIT;
    int cAlign = cDiv4 * PACK_UNIT;
    for (int hi = 0; hi < area; ++hi) {
        const float* srcHeight = src + hi * PACK_UNIT;
        float* dstHeight       = dst + hi * c;
        for (int ci = 0; ci < cDiv4; ++ci) {
            _mm256_storeu_ps(dstHeight + PACK_UNIT * ci, _mm256_loadu_ps(srcHeight + PACK_UNIT * ci * area));
        }
    }

    if (cAlign == c) {
        return;
    }

    int cReamin   = c - cAlign;
    auto srcAlign = src + area * cAlign;
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
        auto x = _mm256_load_ps(src + i * 8);
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
