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
#include "Vec16.hpp"
#define PACK_UNIT 16

void _AVX512_MNNPackCUnit(float* dst, const float* src, size_t area, size_t depth, int* areaOffset) {
    auto areaC4  = area / PACK_UNIT;
    auto depthC4 = depth / PACK_UNIT;
    auto srcAreaOffset = areaOffset[0];
    auto dstAreaOffset = areaOffset[1];
    for (int z = 0; z < depthC4; ++z) {
        auto dstPlane = dst + z * dstAreaOffset * PACK_UNIT;
        auto srcPlane = src + z * srcAreaOffset * PACK_UNIT;
        for (int x = 0; x < areaC4; ++x) {
            auto s  = srcPlane + PACK_UNIT * x;
            auto d  = dstPlane + PACK_UNIT * PACK_UNIT * x;
#define LOAD_CASE(i) auto r##i = _mm512_loadu_ps(s + i * srcAreaOffset)
            LOAD_CASE(0);
            LOAD_CASE(1);
            LOAD_CASE(2);
            LOAD_CASE(3);
            LOAD_CASE(4);
            LOAD_CASE(5);
            LOAD_CASE(6);
            LOAD_CASE(7);
            LOAD_CASE(8);
            LOAD_CASE(9);
            LOAD_CASE(10);
            LOAD_CASE(11);
            LOAD_CASE(12);
            LOAD_CASE(13);
            LOAD_CASE(14);
            LOAD_CASE(15);
#undef LOAD_CASE
            transpose16x16F(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15);
            
#define SAVE_CASE(i) _mm512_storeu_ps(d + PACK_UNIT * i, r##i)
            SAVE_CASE(0);
            SAVE_CASE(1);
            SAVE_CASE(2);
            SAVE_CASE(3);
            SAVE_CASE(4);
            SAVE_CASE(5);
            SAVE_CASE(6);
            SAVE_CASE(7);
            SAVE_CASE(8);
            SAVE_CASE(9);
            SAVE_CASE(10);
            SAVE_CASE(11);
            SAVE_CASE(12);
            SAVE_CASE(13);
            SAVE_CASE(14);
            SAVE_CASE(15);
#undef SAVE_CASE
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
                auto r0 = _mm512_loadu_ps(s + 0 * srcAreaOffset);
                auto r1 = _mm512_setzero_ps();
                auto r2 = _mm512_setzero_ps();
                auto r3 = _mm512_setzero_ps();
                auto r4 = _mm512_setzero_ps();
                auto r5 = _mm512_setzero_ps();
                auto r6 = _mm512_setzero_ps();
                auto r7 = _mm512_setzero_ps();
                auto r8 = _mm512_setzero_ps();
                auto r9 = _mm512_setzero_ps();
                auto r10 = _mm512_setzero_ps();
                auto r11 = _mm512_setzero_ps();
                auto r12 = _mm512_setzero_ps();
                auto r13 = _mm512_setzero_ps();
                auto r14 = _mm512_setzero_ps();
                auto r15 = _mm512_setzero_ps();
#define LOAD_CASE(i) case (i+1):{r##i = _mm512_loadu_ps(s + i * srcAreaOffset);}
                switch (remain) {
                        LOAD_CASE(14);
                        LOAD_CASE(13);
                        LOAD_CASE(12);
                        LOAD_CASE(11);
                        LOAD_CASE(10);
                        LOAD_CASE(9);
                        LOAD_CASE(8);
                        LOAD_CASE(7);
                        LOAD_CASE(6);
                        LOAD_CASE(5);
                        LOAD_CASE(4);
                        LOAD_CASE(3);
                        LOAD_CASE(2);
                        LOAD_CASE(1);
                        LOAD_CASE(0);
                    default:
                        break;
                }
#undef LOAD_CASE
                transpose16x16F(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15);
#define SAVE_CASE(i) _mm512_storeu_ps(d + PACK_UNIT * i, r##i);
                SAVE_CASE(0);
                SAVE_CASE(1);
                SAVE_CASE(2);
                SAVE_CASE(3);
                SAVE_CASE(4);
                SAVE_CASE(5);
                SAVE_CASE(6);
                SAVE_CASE(7);
                SAVE_CASE(8);
                SAVE_CASE(9);
                SAVE_CASE(10);
                SAVE_CASE(11);
                SAVE_CASE(12);
                SAVE_CASE(13);
                SAVE_CASE(14);
                SAVE_CASE(15);
#undef SAVE_CASE
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
            for (int v=0; v<PACK_UNIT; ++v) {
                dstPlane[PACK_UNIT * x + v] = srcPlane[x + v * srcAreaOffset];
            }
        }
    }
}
void _AVX512_MNNUnpackCUnit(float* dst, const float* src, size_t area, size_t depth, int* areaOffset) {
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
#define LOAD_CASE(i) auto r##i = _mm512_loadu_ps(s + i * PACK_UNIT)
            LOAD_CASE(0);
            LOAD_CASE(1);
            LOAD_CASE(2);
            LOAD_CASE(3);
            LOAD_CASE(4);
            LOAD_CASE(5);
            LOAD_CASE(6);
            LOAD_CASE(7);
            LOAD_CASE(8);
            LOAD_CASE(9);
            LOAD_CASE(10);
            LOAD_CASE(11);
            LOAD_CASE(12);
            LOAD_CASE(13);
            LOAD_CASE(14);
            LOAD_CASE(15);
#undef LOAD_CASE
            transpose16x16F(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15);

#define SAVE_CASE(i) _mm512_storeu_ps(d + dstAreaOffset * i, r##i);
                SAVE_CASE(0);
                SAVE_CASE(1);
                SAVE_CASE(2);
                SAVE_CASE(3);
                SAVE_CASE(4);
                SAVE_CASE(5);
                SAVE_CASE(6);
                SAVE_CASE(7);
                SAVE_CASE(8);
                SAVE_CASE(9);
                SAVE_CASE(10);
                SAVE_CASE(11);
                SAVE_CASE(12);
                SAVE_CASE(13);
                SAVE_CASE(14);
                SAVE_CASE(15);
#undef SAVE_CASE
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
#define LOAD_CASE(i) auto r##i = _mm512_loadu_ps(s + i * PACK_UNIT)
            LOAD_CASE(0);
            LOAD_CASE(1);
            LOAD_CASE(2);
            LOAD_CASE(3);
            LOAD_CASE(4);
            LOAD_CASE(5);
            LOAD_CASE(6);
            LOAD_CASE(7);
            LOAD_CASE(8);
            LOAD_CASE(9);
            LOAD_CASE(10);
            LOAD_CASE(11);
            LOAD_CASE(12);
            LOAD_CASE(13);
            LOAD_CASE(14);
            LOAD_CASE(15);
#undef LOAD_CASE
            transpose16x16F(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15);

#define SAVE_CASE(i) case (i+1):{_mm512_storeu_ps(d + i * dstAreaOffset, r##i);}

            switch (remain) {
                    SAVE_CASE(14);
                    SAVE_CASE(13);
                    SAVE_CASE(12);
                    SAVE_CASE(11);
                    SAVE_CASE(10);
                    SAVE_CASE(9);
                    SAVE_CASE(8);
                    SAVE_CASE(7);
                    SAVE_CASE(6);
                    SAVE_CASE(5);
                    SAVE_CASE(4);
                    SAVE_CASE(3);
                    SAVE_CASE(2);
                    SAVE_CASE(1);
                    SAVE_CASE(0);
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
void _AVX512_MNNPackCUnitTranspose(float* dst, const float* src, size_t area, size_t depth, int* areaOffset) {
    int c      = (int)depth;
    int cDiv4  = c / PACK_UNIT;
    int cAlign = cDiv4 * PACK_UNIT;
    auto srcAreaOffset = areaOffset[0];
    auto dstAreaOffset = areaOffset[1];
    for (int hi = 0; hi < area; ++hi) {
        const float* srcHeight = src + hi * c;
        float* dstHeight       = dst + hi * PACK_UNIT;
        for (int ci = 0; ci < cDiv4; ++ci) {
            _mm512_storeu_ps(dstHeight + PACK_UNIT * ci * dstAreaOffset, _mm512_loadu_ps(srcHeight + PACK_UNIT * ci));
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
void _AVX512_MNNUnpackCUnitTranspose(float* dst, const float* src, size_t area, size_t depth, int* areaOffset) {
    int c      = (int)depth;
    int cDiv4  = c / PACK_UNIT;
    int cAlign = cDiv4 * PACK_UNIT;
    auto srcAreaOffset = areaOffset[0];
    auto dstAreaOffset = areaOffset[1];
    for (int hi = 0; hi < area; ++hi) {
        const float* srcHeight = src + hi * PACK_UNIT;
        float* dstHeight       = dst + hi * c;
        for (int ci = 0; ci < cDiv4; ++ci) {
            _mm512_storeu_ps(dstHeight + PACK_UNIT * ci, _mm512_loadu_ps(srcHeight + PACK_UNIT * ci * srcAreaOffset));
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

void _AVX512_MNNPackCUnitInt8(int8_t* dst, const int8_t* src, size_t area, size_t depth, int* areaOffset) {
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
void _AVX512_MNNUnpackCUnitInt8(int8_t* dst, const int8_t* src, size_t area, size_t depth, int* areaOffset) {
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

void _AVX512_MNNPackCUnitTransposeInt8(int8_t* dst, const int8_t* src, size_t area, size_t depth, int* areaOffset) {
    int c      = (int)depth;
    int cDiv4  = c / PACK_UNIT;
    int cAlign = cDiv4 * PACK_UNIT;
    auto srcAreaOffset = areaOffset[0];
    auto dstAreaOffset = areaOffset[1];
    for (int hi = 0; hi < area; ++hi) {
        const int8_t* srcHeight = src + hi * c;
        int8_t* dstHeight       = dst + hi * PACK_UNIT;
        for (int ci = 0; ci < cDiv4; ++ci) {
            _mm_storeu_ps((float*)(dstHeight + PACK_UNIT * ci * dstAreaOffset), _mm_loadu_ps((const float*)(srcHeight + PACK_UNIT * ci)));
        }
    }

    if (cAlign == c) {
        return;
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

void _AVX512_MNNUnpackCUnitTransposeInt8(int8_t* dst, const int8_t* src, size_t area, size_t depth, int* areaOffset) {
    int c      = (int)depth;
    int cDiv4  = c / PACK_UNIT;
    int cAlign = cDiv4 * PACK_UNIT;
    auto srcAreaOffset = areaOffset[0];
    auto dstAreaOffset = areaOffset[1];
    for (int hi = 0; hi < area; ++hi) {
        const int8_t* srcHeight = src + hi * PACK_UNIT;
        int8_t* dstHeight       = dst + hi * c;
        for (int ci = 0; ci < cDiv4; ++ci) {
            _mm_storeu_ps((float*)(dstHeight + PACK_UNIT * ci), _mm_loadu_ps((const float*)(srcHeight + PACK_UNIT * ci * srcAreaOffset)));
        }
    }

    if (cAlign == c) {
        return;
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
void _AVX512_ReorderInit(void* functions) {
    auto coreFunction = static_cast<MNN::CoreFunctions*>(functions);
    coreFunction->MNNPackCUnit = _AVX512_MNNPackCUnit;
    coreFunction->MNNPackCUnitTranspose = _AVX512_MNNPackCUnitTranspose;
    coreFunction->MNNUnpackCUnit = _AVX512_MNNUnpackCUnit;
    coreFunction->MNNUnpackCUnitTranspose = _AVX512_MNNUnpackCUnitTranspose;

    coreFunction->MNNUnpackCUnitTransposeInt8 = _AVX512_MNNUnpackCUnitTransposeInt8;
    coreFunction->MNNPackCUnitInt8 = _AVX512_MNNPackCUnitInt8;
    coreFunction->MNNUnpackCUnitInt8 = _AVX512_MNNUnpackCUnitInt8;
    coreFunction->MNNPackCUnitTransposeInt8 = _AVX512_MNNPackCUnitTransposeInt8;
}
