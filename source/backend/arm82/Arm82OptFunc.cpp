//
//  Arm82OptFunc.hpp
//  MNN
//
//  Created by MNN on 2019/02/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/arm82/Arm82OptFunc.hpp"
#include "core/Macro.h"
#include "half.hpp"
#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif
void MNNQuantizeFP16(FLOAT16* dst, const float* src, int size) {
#ifdef MNN_USE_NEON

    int sizeDiv4 = size / 4;
    int remain   = size - sizeDiv4 * 4;

    if (sizeDiv4 > 0) {
        MNNQuantizeFP16_UNIT4(dst, src, sizeDiv4);
    }

    if (remain > 0) {
        for (int i = sizeDiv4 * 4; i < size; ++i) {
            dst[i] = half_float::half(src[i]);
        }
    }

#else
    for (int i = 0; i < size; ++i) {
        dst[i] = half_float::half(src[i]);
    }
#endif
}

void MNNNC4HW4TONC8HW8(uint16_t* dst, const float* source, size_t plane, size_t channel) {
    const int c4 = UP_DIV(channel, 4);
    const int c8 = UP_DIV(channel, 8);
    memset(dst, 0, plane * c8 * 8 * sizeof(uint16_t));
#if defined(MNN_USE_NEON) && defined(__aarch64__)
    auto dest = (float16_t*)dst;
#else
    auto dest = dst;
#endif
    for (int c = 0; c < c4; ++c) {
        int ci          = c / 2;
        int cj          = c % 2;
        auto dstChannel = dest + ci * 8 * plane + cj * 4;
        auto srcChannle = source + c * plane * 4;

        for (int i = 0; i < plane; ++i) {
#if defined(MNN_USE_NEON) && defined(__aarch64__)
            float32x4_t a = vld1q_f32(srcChannle + i * 4);
            vst1_f16(dstChannel + i * 8, vcvt_f16_f32(a));
#else
            half_float::half dataHalf[4];
            for (int k = 0; k < 4; ++k) {
                dataHalf[k] = srcChannle[i * 4 + k];
                // MNN_PRINT("==> %f\n", float(dataHalf[k]));
            }
            memcpy(dstChannel + i * 8, dataHalf, sizeof(half_float::half) * 4);
#endif
        }
    }
}

void MNNNC8HW8TONC4HW4(float* dest, const uint16_t* src, size_t plane, size_t channel) {
    const int c4 = UP_DIV(channel, 4);
#if defined(MNN_USE_NEON) && defined(__aarch64__)
    auto source = (float16_t*)src;
#else
    auto source = src;
#endif
    for (int c = 0; c < c4; ++c) {
        int ci          = c / 2;
        int cj          = c % 2;
        auto srcChannel = source + ci * 8 * plane + cj * 4;
        auto dstChannel = dest + c * plane * 4;

        for (int i = 0; i < plane; ++i) {
#if defined(MNN_USE_NEON) && defined(__aarch64__)
            float16x4_t a = vld1_f16(srcChannel + i * 8);
            vst1q_f32(dstChannel + i * 4, vcvt_f32_f16(a));
#else
            half_float::half dataHalf[4];
            memcpy(dataHalf, srcChannel + i * 8, sizeof(half_float::half) * 4);
            for (int k = 0; k < 4; ++k) {
                dstChannel[i * 4 + k] = float(dataHalf[k]);
            }
#endif
        }
    }
}

void MNNNC8HW8TONHWC(float* dest, const uint16_t* src, size_t plane, size_t channel) {
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
#ifdef MNN_USE_NEON
            float16x8_t a = vld1q_f16(srcHeight + 8 * ci * plane);
            vst1q_f32(dstHeight + 8 * ci, vcvt_high_f32_f16(a));
#else
            half_float::half dataHalf[8];
            memcpy(dataHalf, srcHeight + 8 * ci * plane, 8 * sizeof(uint16_t));
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

void MNNNCHWTONC8HW8(uint16_t* dest, const float* source, size_t plane, size_t channel) {
    auto halfDest = reinterpret_cast<half_float::half*>(dest);
    MNNPackUNIT<float, half_float::half, 8>(halfDest, source, plane, channel);
}

void MNNNC8HW8TONCHW(float* dest, const uint16_t* source, size_t plane, size_t channel) {
    auto halfSrc = reinterpret_cast<const half_float::half*>(source);
    MNNUnpackUNIT<half_float::half, float, 8>(dest, halfSrc, plane, channel);
}

void MNNNCHWTONC8HW8_NO_TYPE(uint16_t* dest, const uint16_t* source, size_t plane, size_t channel) {
    MNNPackUNIT<uint16_t, uint16_t, 8>(dest, source, plane, channel);
}

void MNNNC8HW8TONCHW_NO_TYPE(uint16_t* dest, const uint16_t* source, size_t plane, size_t channel) {
    MNNUnpackUNIT<uint16_t, uint16_t, 8>(dest, source, plane, channel);
}
