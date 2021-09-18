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
}
