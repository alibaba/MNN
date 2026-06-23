//
//  ReorderFunctions.cpp
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
#include "Vec8.hpp"
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
void _AVX_MNNUnpackCUnitTranspose(float* dst, const float* src, size_t area, size_t depth, int* offset) {
    int c      = (int)depth;
    int cDiv4  = c / PACK_UNIT;
    int cAlign = cDiv4 * PACK_UNIT;
    auto srcAreaOffset = offset[0];
    auto dstDepthOffset = offset[1];
    for (int hi = 0; hi < area; ++hi) {
        const float* srcHeight = src + hi * PACK_UNIT;
        float* dstHeight       = dst + hi * dstDepthOffset;
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
        float* dstHeight       = dstAlign + hi * dstDepthOffset;

        for (int ci = 0; ci < cReamin; ++ci) {
            dstHeight[ci] = srcHeight[ci];
        }
    }
}


void _AVX_MNNPackCUnitInt8(int8_t* dst, const int8_t* src, size_t area, size_t depth, int* areaOffset) {
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
            for (int i=0; i<PACK_UNIT; ++i) {
                for (int j=0; j<PACK_UNIT; ++j) {
                    d[PACK_UNIT*i +j] = s[i + j * srcAreaOffset];
                }
            }
        }
    }
    auto areaRemain  = areaC4 * PACK_UNIT;
    auto depthRemain = depthC4 * PACK_UNIT;
    // Down
    int remain = depth - depthRemain;
    if (remain > 0) {
        auto dstPlane       = depthC4 * dstAreaOffset * PACK_UNIT + dst;
        const auto srcPlane = src + depthC4 * srcAreaOffset * PACK_UNIT;
        {
            for (int x = 0; x < areaC4; ++x) {
                auto s  = srcPlane + PACK_UNIT * x;
                auto d  = dstPlane + PACK_UNIT * PACK_UNIT * x;
                ::memset(d, 0, PACK_UNIT * PACK_UNIT * sizeof(int8_t));
                for (int i=0; i<PACK_UNIT; ++i) {
                    for (int j=0; j<remain; ++j) {
                        d[PACK_UNIT*i +j] = s[i + j * srcAreaOffset];
                    }
                }
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
        auto dstPlane       = z * dstAreaOffset * PACK_UNIT + dst;
        auto srcPlane = src + z * srcAreaOffset * PACK_UNIT;
        for (int x = areaRemain; x < area; ++x) {
            for (int j=0; j<PACK_UNIT; ++j) {
                dstPlane[PACK_UNIT * x + j] = srcPlane[x + srcAreaOffset * j];
            }
        }
    }
}
void _AVX_MNNUnpackCUnitInt8(int8_t* dst, const int8_t* src, size_t area, size_t depth, int* areaOffset) {
    auto areaC4  = area / PACK_UNIT;
    auto depthC4 = depth / PACK_UNIT;
    auto srcAreaOffset = areaOffset[0];
    auto dstAreaOffset = areaOffset[1];
    for (int z = 0; z < depthC4; ++z) {
        auto dstPlane = dst + z * dstAreaOffset * PACK_UNIT;
        auto srcPlane = src + z * srcAreaOffset * PACK_UNIT;
        for (int x = 0; x < areaC4; ++x) {
            auto s  = srcPlane + PACK_UNIT * PACK_UNIT * x;
            auto d  = dstPlane + PACK_UNIT * x;
            for (int i=0; i<PACK_UNIT; ++i) {
                for (int j=0; j<PACK_UNIT; ++j) {
                    d[i+j*dstAreaOffset] = s[PACK_UNIT*i+j];
                }
            }
        }
    }
    auto areaRemain  = areaC4 * PACK_UNIT;
    auto depthRemain = depthC4 * PACK_UNIT;
    // Down
    int remain = depth - depthRemain;
    if (remain > 0) {
        auto dstPlane       = dst + depthC4 * dstAreaOffset * PACK_UNIT;
        const auto srcPlane = src + depthC4 * srcAreaOffset * PACK_UNIT;
        {
            for (int x = 0; x < areaC4; ++x) {
                auto s  = srcPlane + PACK_UNIT * PACK_UNIT * x;
                auto d  = dstPlane + PACK_UNIT * x;
                for (int i=0; i<PACK_UNIT; ++i) {
                    for (int j=0; j<remain; ++j) {
                        d[i + j * dstAreaOffset] = s[PACK_UNIT*i +j];
                    }
                }
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
        auto dstPlane = dst + z * dstAreaOffset * PACK_UNIT;
        auto srcPlane = src + z * srcAreaOffset * PACK_UNIT;
        for (int x = areaRemain; x < area; ++x) {
            for (int j=0; j<PACK_UNIT; ++j) {
                dstPlane[x + dstAreaOffset * j] = srcPlane[PACK_UNIT * x + j];
            }
        }
    }
}

void _AVX_MNNPackCUnitTransposeInt8(int8_t* dst, const int8_t* src, size_t area, size_t depth, int* areaOffset) {
    int c      = (int)depth;
    int cDiv4  = c / PACK_UNIT;
    int cAlign = cDiv4 * PACK_UNIT;
    auto srcAreaOffset = areaOffset[0];
    auto dstAreaOffset = areaOffset[1];
    if (cAlign == c) {
        for (int hi = 0; hi < area; ++hi) {
            const int8_t* srcHeight = src + hi * c;
            int8_t* dstHeight       = dst + hi * PACK_UNIT;
            for (int ci = 0; ci < cDiv4; ++ci) {
                *(int64_t*)(dstHeight + PACK_UNIT * ci * dstAreaOffset) = *(int64_t*)(srcHeight + PACK_UNIT * ci);
            }
        }
        return;
    }
    for (int hi = 0; hi < area; ++hi) {
        const int8_t* srcHeight = src + hi * c;
        int8_t* dstHeight       = dst + hi * PACK_UNIT;
        for (int ci = 0; ci < cDiv4; ++ci) {
            for (int k=0; k<PACK_UNIT; ++k) {
                dstHeight[PACK_UNIT * ci * dstAreaOffset + k] = srcHeight[PACK_UNIT * ci + k];
            }
        }
    }
    int cReamin   = c - cAlign;
    auto srcAlign = src + cAlign;
    auto dstAlign = dst + dstAreaOffset * cAlign;

    for (int hi = 0; hi < area; ++hi) {
        const int8_t* srcHeight = srcAlign + hi * c;
        int8_t* dstHeight       = dstAlign + hi * PACK_UNIT;
        for (int i = 0; i < PACK_UNIT; ++i) {
            dstHeight[i] = 0;
        }
        for (int ci = 0; ci < cReamin; ++ci) {
            dstHeight[ci] = srcHeight[ci];
        }
    }

}

void _AVX_MNNUnpackCUnitTransposeInt8(int8_t* dst, const int8_t* src, size_t area, size_t depth, int* areaOffset) {
    int c      = (int)depth;
    int cDiv4  = c / PACK_UNIT;
    int cAlign = cDiv4 * PACK_UNIT;
    auto srcAreaOffset = areaOffset[0];
    auto dstAreaOffset = areaOffset[1];
    if (cAlign == c) {
        for (int hi = 0; hi < area; ++hi) {
            const int8_t* srcHeight = src + hi * PACK_UNIT;
            int8_t* dstHeight       = dst + hi * c;
            for (int ci = 0; ci < cDiv4; ++ci) {
                *(int64_t*)(dstHeight + PACK_UNIT * ci) = *(int64_t*)(srcHeight + PACK_UNIT * ci * srcAreaOffset);
            }
        }
        return;
    }
    for (int hi = 0; hi < area; ++hi) {
        const int8_t* srcHeight = src + hi * PACK_UNIT;
        int8_t* dstHeight       = dst + hi * c;
        for (int ci = 0; ci < cDiv4; ++ci) {
            for (int k=0; k<PACK_UNIT; ++k) {
                dstHeight[PACK_UNIT * ci + k] = srcHeight[PACK_UNIT * ci * srcAreaOffset + k];
            }
        }
    }
    int cReamin   = c - cAlign;
    auto srcAlign = src + srcAreaOffset * cAlign;
    auto dstAlign = dst + cAlign;

    for (int hi = 0; hi < area; ++hi) {
        const int8_t* srcHeight = srcAlign + hi * PACK_UNIT;
        int8_t* dstHeight       = dstAlign + hi * c;

        for (int ci = 0; ci < cReamin; ++ci) {
            dstHeight[ci] = srcHeight[ci];
        }
    }
}
static void _AVX_MNNSumWeightInt8(float* kernelsum, int8_t* source, size_t outside, size_t reduceAxis, size_t hP, size_t lP) {
    auto inside = hP * lP;
    auto stride0 = inside * reduceAxis;
    if (lP == 4 && hP % 8 == 0) {
        for (int i = 0; i < outside; ++i) {
            memset(kernelsum + i * hP, 0, hP * sizeof(float));
            for (int j = 0; j < reduceAxis; ++j) {
                int8_t* src_j = source + j * inside + i * stride0;
                for (int k = 0; k < hP; k += 8) {
                    __m256i v = _mm256_loadu_si256((const __m256i*)(src_j + k * lP));
                    __m128i v_lo = _mm256_castsi256_si128(v);
                    __m128i v_hi = _mm256_extracti128_si256(v, 1);
                    __m256i v16_0 = _mm256_cvtepi8_epi16(v_lo);
                    __m256i v16_1 = _mm256_cvtepi8_epi16(v_hi);
                    __m256i ones = _mm256_set1_epi16(1);
                    __m256i sum32_0 = _mm256_madd_epi16(v16_0, ones);
                    __m256i sum32_1 = _mm256_madd_epi16(v16_1, ones);
                    __m256i final32_0 = _mm256_add_epi32(sum32_0, _mm256_srli_epi64(sum32_0, 32));
                    __m256i final32_1 = _mm256_add_epi32(sum32_1, _mm256_srli_epi64(sum32_1, 32));
                    __m256 final_f0 = _mm256_cvtepi32_ps(final32_0);
                    __m256 final_f1 = _mm256_cvtepi32_ps(final32_1);
                    __m256 packed = _mm256_shuffle_ps(final_f0, final_f1, _MM_SHUFFLE(2, 0, 2, 0));
                    __m256 result = _mm256_permutevar8x32_ps(packed, _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7));
                    __m256 acc = _mm256_loadu_ps(kernelsum + i * hP + k);
                    acc = _mm256_add_ps(acc, result);
                    _mm256_storeu_ps(kernelsum + i * hP + k, acc);
                }
            }
        }
        return;
    }
    std::vector<float> accum(hP);
    for (int i = 0; i < outside; ++i) {
        memset(accum.data(), 0, hP * sizeof(float));
        for (int j = 0; j < reduceAxis; ++j) {
            for (int k = 0; k < hP; ++k) {
                for (int x = 0; x < lP; ++x) {
                    accum[k] += (float)source[x + k * lP + j * inside + i * stride0];
                }
            }
        }
        memcpy(kernelsum + i * hP, accum.data(), hP * sizeof(float));
    }
}

static void _AVX_MNNReorderWeightInt4(uint8_t* dest, const uint8_t* source, int32_t* shape, size_t size, float* kernelsum) {
    auto blocknum = shape[0];
    auto hu       = shape[1];
    auto lu       = shape[2];
    auto hp       = shape[3];
    auto lp       = shape[4];
    auto ic       = blocknum * lu * lp;
    auto stride0  = blocknum * hp * lu * lp;
    auto stride1  = lu * hp * lp;
    auto stride2  = hp * lp;
    for (int i = 0; i < hu; ++i) {
        for (int bl = 0; bl < blocknum; ++bl) {
            for (int j = 0; j < lu; ++j) {
                int srcindex_base = i * hp * ic + bl * lu * lp + j * lp;
                int dstindex_base = i * stride0 + bl * stride1 + j * stride2;
                if (lp == 2 && hp == 64) {
                    uint16_t* dst16 = (uint16_t*)(dest + dstindex_base);
                    const uint8_t* src8 = source + srcindex_base;
                    for (int k = 0; k < 64; ++k) {
                        dst16[k] = *(const uint16_t*)(src8 + k * ic);
                    }
                } else {
                    for (int k = 0; k < hp; ++k) {
                        int srcindex = srcindex_base + k * ic;
                        int dstindex = dstindex_base + k * lp;
                        memcpy(dest + dstindex, source + srcindex, lp);
                    }
                }
            }
        }
    }
    auto inside = lp * hp;
    auto outside = blocknum * hu;
    if (lp == 2 && hp == 64) {
        for (int i = 0; i < outside; ++i) {
            memset(kernelsum + i * hp, 0, hp * sizeof(float));
            for (int k = 0; k < lu; ++k) {
                uint8_t* D = dest + (i * lu + k) * inside;
                __m256i v1 = _mm256_loadu_si256((const __m256i*)(D)); 
                __m256i v2 = _mm256_loadu_si256((const __m256i*)(D + 64)); 
                __m256i mask_0f = _mm256_set1_epi8(0x0f);
                __m256i w0 = _mm256_and_si256(_mm256_srli_epi16(v1, 4), mask_0f); 
                __m256i w1 = _mm256_and_si256(v1, mask_0f);
                __m256i w2 = _mm256_and_si256(_mm256_srli_epi16(v2, 4), mask_0f);
                __m256i w3 = _mm256_and_si256(v2, mask_0f);
                __m256i b0 = _mm256_or_si256(_mm256_slli_epi16(w0, 4), w2);
                __m256i b1 = _mm256_or_si256(_mm256_slli_epi16(w1, 4), w3);
                __m256i lo = _mm256_unpacklo_epi8(b0, b1);
                __m256i hi = _mm256_unpackhi_epi8(b0, b1);
                __m256i out0 = _mm256_permute2x128_si256(lo, hi, 0x20); 
                __m256i out1 = _mm256_permute2x128_si256(lo, hi, 0x31); 
                __m256i v3 = _mm256_loadu_si256((const __m256i*)(D + 32)); 
                __m256i v4 = _mm256_loadu_si256((const __m256i*)(D + 96)); 
                __m256i w0_2 = _mm256_and_si256(_mm256_srli_epi16(v3, 4), mask_0f); 
                __m256i w1_2 = _mm256_and_si256(v3, mask_0f);
                __m256i w2_2 = _mm256_and_si256(_mm256_srli_epi16(v4, 4), mask_0f);
                __m256i w3_2 = _mm256_and_si256(v4, mask_0f);
                __m256i b0_2 = _mm256_or_si256(_mm256_slli_epi16(w0_2, 4), w2_2);
                __m256i b1_2 = _mm256_or_si256(_mm256_slli_epi16(w1_2, 4), w3_2);
                __m256i lo_2 = _mm256_unpacklo_epi8(b0_2, b1_2);
                __m256i hi_2 = _mm256_unpackhi_epi8(b0_2, b1_2);
                __m256i out2 = _mm256_permute2x128_si256(lo_2, hi_2, 0x20); 
                __m256i out3 = _mm256_permute2x128_si256(lo_2, hi_2, 0x31); 
                _mm256_storeu_si256((__m256i*)(D), out0);
                _mm256_storeu_si256((__m256i*)(D + 32), out1);
                _mm256_storeu_si256((__m256i*)(D + 64), out2);
                _mm256_storeu_si256((__m256i*)(D + 96), out3);
                
                auto do_acc = [&](__m256i a0, __m256i a1, __m256i a2, __m256i a3, int off) {
                    __m256i sum1 = _mm256_add_epi8(a0, a1);
                    __m256i sum1_16 = _mm256_maddubs_epi16(sum1, _mm256_set1_epi8(1));
                    __m256i sum1_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(sum1_16));
                    __m256i sum1_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(sum1_16, 1));
                    __m256 acc_lo = _mm256_loadu_ps(kernelsum + i * hp + off);
                    __m256 acc_hi = _mm256_loadu_ps(kernelsum + i * hp + off + 8);
                    acc_lo = _mm256_add_ps(acc_lo, _mm256_cvtepi32_ps(sum1_lo));
                    acc_hi = _mm256_add_ps(acc_hi, _mm256_cvtepi32_ps(sum1_hi));
                    _mm256_storeu_ps(kernelsum + i * hp + off, acc_lo);
                    _mm256_storeu_ps(kernelsum + i * hp + off + 8, acc_hi);
                    
                    __m256i sum2 = _mm256_add_epi8(a2, a3);
                    __m256i sum2_16 = _mm256_maddubs_epi16(sum2, _mm256_set1_epi8(1));
                    __m256i sum2_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(sum2_16));
                    __m256i sum2_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(sum2_16, 1));
                    __m256 acc2_lo = _mm256_loadu_ps(kernelsum + i * hp + off + 32);
                    __m256 acc2_hi = _mm256_loadu_ps(kernelsum + i * hp + off + 40);
                    acc2_lo = _mm256_add_ps(acc2_lo, _mm256_cvtepi32_ps(sum2_lo));
                    acc2_hi = _mm256_add_ps(acc2_hi, _mm256_cvtepi32_ps(sum2_hi));
                    _mm256_storeu_ps(kernelsum + i * hp + off + 32, acc2_lo);
                    _mm256_storeu_ps(kernelsum + i * hp + off + 40, acc2_hi);
                };
                do_acc(w0, w1, w2, w3, 0);
                do_acc(w0_2, w1_2, w2_2, w3_2, 16);
            }
        }
        return;
    }

    std::vector<uint8_t> buffer(inside);
    std::vector<float> accum(hp);
    for (int i = 0; i < outside; ++i) {
        memset(accum.data(), 0, hp * sizeof(float));
        for (int k = 0; k < lu; ++k) {
            for (int j = 0; j < inside / 2; ++j) {
                auto w0 = dest[j + (i * lu + k) * inside] >> 4;
                auto w1 = dest[j + (i * lu + k) * inside] & 0x0f;
                auto w2 = dest[(i * lu + k) * inside + j + inside / 2] >> 4;
                auto w3 = dest[(i * lu + k) * inside + j + inside / 2] & 0x0f;
                buffer[2 * j + 0] = w0 * 16 + w2;
                buffer[2 * j + 1] = w1 * 16 + w3;
                accum[j / lp] += ((float)w0 + (float)w1);
                accum[(j + inside / 2) / lp] += ((float)w2 + (float)w3);
            }
            memcpy(dest + (i * lu + k) * inside, buffer.data(), inside);
        }
        memcpy(kernelsum + i * hp, accum.data(), hp * sizeof(float));
    }
}

void _AVX_ReorderInit(void* functions) {
    auto coreFunction = static_cast<MNN::CoreFunctions*>(functions);
    coreFunction->MNNPackCUnit = _AVX_MNNPackCUnit;
    coreFunction->MNNUnpackCUnit = _AVX_MNNUnpackCUnit;
    coreFunction->MNNPackCUnitTranspose = _AVX_MNNPackCUnitTranspose;
    coreFunction->MNNUnpackCUnitTranspose = _AVX_MNNUnpackCUnitTranspose;

    coreFunction->MNNUnpackCUnitTransposeInt8 = _AVX_MNNUnpackCUnitTransposeInt8;
    coreFunction->MNNPackCUnitInt8 = _AVX_MNNPackCUnitInt8;
    coreFunction->MNNUnpackCUnitInt8 = _AVX_MNNUnpackCUnitInt8;
    coreFunction->MNNPackCUnitTransposeInt8 = _AVX_MNNPackCUnitTransposeInt8;

    coreFunction->int8MatmulRelatedFunctions.MNNSumWeightInt8 = _AVX_MNNSumWeightInt8;
    coreFunction->int8MatmulRelatedFunctions.MNNReorderWeightInt4 = _AVX_MNNReorderWeightInt4;
}
