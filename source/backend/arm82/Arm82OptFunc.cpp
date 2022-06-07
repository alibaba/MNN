//
//  Arm82OptFunc.hpp
//  MNN
//
//  Created by MNN on 2019/02/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#if defined(__ANDROID__) || defined(__aarch64__)

#include "Arm82OptFunc.hpp"
#include "Arm82Vec.hpp"
#include "core/Macro.h"
#include "half.hpp"

#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

extern "C" {
void MNNExpFP16(void* dst, const void* src, const FLOAT16* params, size_t blockCount);

void MNNQuantizeFP16_UNIT4(int16_t* dst, const float* src, size_t size, const float* minMax);

}

void Arm82MNNExp(FLOAT16* dst, const FLOAT16* src, size_t dataSize) {
    int blockCount = (int)dataSize / 16;
    static FLOAT16 params[] = {
        (FLOAT16)log(2.0f), (FLOAT16)(1.0f / log(2.0f)), 1.0f, 1.0f, 0.5f, 1.0f / 6.0f, 1.0f / 24.0f, 1.0f / 120.0f};
    if (blockCount > 0) {
        MNNExpFP16(dst, src, params, blockCount);
    }
    int remainSize = dataSize % 16;
    if (remainSize > 0) {
        int16_t srcTemp[16];
        int16_t dstTemp[16];
        ::memcpy(srcTemp, src + blockCount * 16, remainSize * sizeof(int16_t));
        MNNExpFP16(dstTemp, srcTemp, params, 1);
        ::memcpy(dst + blockCount * 16, dstTemp, remainSize * sizeof(int16_t));
    }
}

void Arm82MNNGetMatMulPackMode(int* eP, int *lP, int* hP) {
#ifdef __aarch64__
    *hP = 16;
#else
    *hP = 8;
#endif
    *eP = 12;
    *lP = 1;
}

void MNNQuantizeFP16(const float* src, int16_t* dst, size_t size) {
    int sizeDiv4 = size / 4;
    int remain   = size - sizeDiv4 * 4;
    float minMax[] = {
        -65504.0f,
        65504.0f
    };
    if (sizeDiv4 > 0) {
        MNNQuantizeFP16_UNIT4(dst, src, sizeDiv4, minMax);
        src += sizeDiv4 * 4;
        dst += sizeDiv4 * 4;
    }
    if (remain > 0) {
        float tempSrc[4];
        int16_t tempDst[4];
        ::memcpy(tempSrc, src, remain * sizeof(float));
        MNNQuantizeFP16_UNIT4(tempDst, tempSrc, 1, minMax);
        ::memcpy(dst, tempDst, remain * sizeof(int16_t));
    }
}

void MNNDequantizeFP16(const int16_t* srcint, float* dst, size_t size) {
    auto src = (const FLOAT16*)srcint;
    int sizeDiv4 = size / 4;
    int remain   = size - sizeDiv4 * 4;
    for (int i = 0; i < sizeDiv4; ++i) {
        auto S = vld1_f16(src);
        auto D = vcvt_f32_f16(S);
        vst1q_f32(dst, D);
        dst += 4;
        src += 4;
    }
    if (remain > 0) {
        FLOAT16 tempSrc[4];
        float tempDst[4];
        ::memcpy(tempSrc, src, remain * sizeof(int16_t));
        auto S = vld1_f16(tempSrc);
        auto D = vcvt_f32_f16(S);
        vst1q_f32(tempDst, D);
        ::memcpy(dst, tempDst, remain * sizeof(float));
    }
}

extern "C" {
void MNNPackC8FP16_C8(int16_t* dest, const int16_t* source, size_t area, size_t depth, int32_t* areaOffset);
void MNNUnpackC8FP16_C8(int16_t* dest, const int16_t* source, size_t area, size_t depth, int32_t* areaOffset);
};
void MNNPackC8FP16(int16_t* dest, const int16_t* source, size_t area, size_t depth, int32_t* areaOffset) {
    const int UNIT = 8;
    int srcAreaOffset = areaOffset[0];
    int dstAreaOffset = areaOffset[1];
    int depthC = depth / UNIT;
    int depthR = depth % UNIT;
    if (depthC > 0) {
        MNNPackC8FP16_C8(dest, source, area, depth, areaOffset);
    }
#ifdef MNN_ARM82_REFCODE
    for (int p=0; p<depthC; ++p) {
        auto dst = dest + p * UNIT * dstAreaOffset;
        auto src = source + p * UNIT * srcAreaOffset;
        for (int z = 0; z < UNIT; ++z) {
            auto dstPlane = dst + z;
            auto srcPlane = src + srcAreaOffset * z;
            for (int x = 0; x < area; ++x) {
                dstPlane[x * UNIT] = srcPlane[x];
            }
        }
    }
#endif
    // TODO: Optimize it
    if (depthR > 0) {
        auto dst = dest + depthC * UNIT * dstAreaOffset;
        auto src = source + depthC * UNIT * srcAreaOffset;
        ::memset(dst, 0, area * UNIT * sizeof(int16_t));
        for (int z = 0; z < depthR; ++z) {
            auto srcPlane = z * srcAreaOffset + src;
            auto dstPlane = dst + z;
            for (int x = 0; x < area; ++x) {
                dstPlane[x * UNIT] = srcPlane[x];
            }
        }
    }
}


void MNNUnPackC8FP16(int16_t* dest, const int16_t* source, size_t area, size_t depth, int32_t* areaOffset) {
    const int UNIT = 8;
    int depthC = depth / UNIT;
    int depthR = depth % UNIT;
    int srcAreaOffset = areaOffset[0];
    int dstAreaOffset = areaOffset[1];
    if (depthC > 0) {
        MNNUnpackC8FP16_C8(dest, source, area, depth, areaOffset);
    }
#ifdef MNN_ARM82_REFCODE
    for (int p=0; p<depthC; ++p) {
        auto dst = dest + p * UNIT * dstAreaOffset;
        auto src = source + p * UNIT * srcAreaOffset;
        for (int z = 0; z < UNIT; ++z) {
            auto srcPlane = src + z;
            auto dstPlane = dst + dstAreaOffset * z;
            for (int x = 0; x < area; ++x) {
                dstPlane[x] = srcPlane[x * UNIT];
            }
        }
    }
#endif
    // TODO: Optimize it
    if (depthR > 0) {
        auto dst = dest + depthC * UNIT * dstAreaOffset;
        auto src = source + depthC * UNIT * srcAreaOffset;
        for (int z = 0; z < depthR; ++z) {
            auto srcPlane = src + z;
            auto dstPlane = dst + dstAreaOffset * z;
            for (int x = 0; x < area; ++x) {
                dstPlane[x] = srcPlane[x * UNIT];
            }
        }
    }
}

void MNNNC4HW4TONC8HW8(FLOAT16* dst, const float* source, size_t plane, size_t channel) {
    const int c4 = UP_DIV(channel, 4);
    const int c8 = UP_DIV(channel, 8);
    memset(dst, 0, plane * c8 * 8 * sizeof(FLOAT16));
    auto dest = (float16_t*)dst;
    for (int c = 0; c < c4; ++c) {
        int ci          = c / 2;
        int cj          = c % 2;
        auto dstChannel = dest + ci * 8 * plane + cj * 4;
        auto srcChannle = source + c * plane * 4;

        for (int i = 0; i < plane; ++i) {
            float32x4_t a = vld1q_f32(srcChannle + i * 4);
            vst1_f16(dstChannel + i * 8, vcvt_f16_f32(a));
        }
    }
}

void MNNNC8HW8TONC4HW4(float* dest, const FLOAT16* src, size_t plane, size_t channel) {
    const int c4 = UP_DIV(channel, 4);
    auto source = (float16_t*)src;
    for (int c = 0; c < c4; ++c) {
        int ci          = c / 2;
        int cj          = c % 2;
        auto srcChannel = source + ci * 8 * plane + cj * 4;
        auto dstChannel = dest + c * plane * 4;

        for (int i = 0; i < plane; ++i) {
            float16x4_t a = vld1_f16(srcChannel + i * 8);
            vst1q_f32(dstChannel + i * 4, vcvt_f32_f16(a));
        }
    }
}

void MNNNC8HW8TONHWC(float* dest, const FLOAT16* src, size_t plane, size_t channel) {
    int c      = (int)channel;
    int cDiv8  = c / 8;
    int cAlign = cDiv8 * 8;
#if defined(MNN_USE_NEON) && defined(__aarch64__)
    auto source = (float16_t*)src;
#else
    auto source = src;
#endif
    for (int hi = 0; hi < plane; ++hi) {
        const auto srcHeight = source + hi * 8;
        float* dstHeight     = dest + hi * c;
        for (int ci = 0; ci < cDiv8; ++ci) {
#if defined(MNN_USE_NEON) && defined(__aarch64__)
            float16x8_t a = vld1q_f16(srcHeight + 8 * ci * plane);
            vst1q_f32(dstHeight + 8 * ci, vcvt_high_f32_f16(a));
#else
            half_float::half dataHalf[8];
            memcpy(dataHalf, srcHeight + 8 * ci * plane, 8 * sizeof(FLOAT16));
            for (int i = 0; i < 8; ++i) {
                dstHeight[ci * 8 + i] = float(dataHalf[i]);
            }
#endif
        }
    }
    if (cAlign == c) {
        return;
    }
    int cReamin         = c - cAlign;
    const auto srcAlign = reinterpret_cast<const half_float::half*>(source + plane * cAlign);
    auto dstAlign       = dest + cAlign;
    for (int hi = 0; hi < plane; ++hi) {
        const auto srcHeight = srcAlign + hi * 8;
        float* dstHeight     = dstAlign + hi * c;

        for (int ci = 0; ci < cReamin; ++ci) {
            dstHeight[ci] = float(srcHeight[ci]);
        }
    }
}
#endif
