//
//  Arm82Relu.cpp
//  MNN
//
//  Created by MNN on 2020/2/13.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#if defined(__ANDROID__) || defined(__aarch64__)

#include <limits>

#include "Arm82Relu.hpp"
#include "Arm82Backend.hpp"
#include "Arm82OptFunc.hpp"
#include "core/Concurrency.h"
#include "core/Macro.h"
#include "half.hpp"
#include <algorithm>
#include <arm_neon.h>

namespace MNN {

static void _MNNArm82PReluWithChannel(FLOAT16 *dst, const FLOAT16 *src, const FLOAT16 *slope, size_t length) {
#ifdef MNN_USE_NEON
    float16x8_t value_0 = vmovq_n_f16(0);
    float16x8_t slopeV  = vld1q_f16(slope);
#endif

    for (int i = 0; i < length; ++i) {
#ifdef MNN_USE_NEON
        float16x8_t value        = vld1q_f16(src + i * ARMV82_CHANNEL_UNIT);
        float16x8_t mulSlope     = vmulq_f16(value, slopeV);
        uint16x8_t lessThanZero = vcleq_f16(value, value_0);

        vst1q_f16(dst + i * ARMV82_CHANNEL_UNIT, vbslq_f16(lessThanZero, mulSlope, value));
#else

        for (int j = 0; j < ARMV82_CHANNEL_UNIT; ++j) {
            if (src[i * ARMV82_CHANNEL_UNIT + j] < 0) {
                dst[i * ARMV82_CHANNEL_UNIT + j] = src[i * ARMV82_CHANNEL_UNIT + j] * slope[j];
            } else {
                dst[i * ARMV82_CHANNEL_UNIT + j] = src[i * ARMV82_CHANNEL_UNIT + j];
            }
        }

#endif
    }
}

static void _MNNArm82LeakyReluWithChannel(FLOAT16 *dst, const FLOAT16 *src, const FLOAT16 slope, size_t length) {
    float16x8_t value_0 = vmovq_n_f16(0);
    float16x8_t slopeV  = vmovq_n_f16(slope);
    auto lC8 = length / ARMV82_CHANNEL_UNIT;
    auto remain = length % ARMV82_CHANNEL_UNIT;

    for (int i = 0; i < lC8; ++i) {
        float16x8_t value        = vld1q_f16(src);
        float16x8_t mulSlope     = vmulq_f16(value, slopeV);
        uint16x8_t lessThanZero = vcleq_f16(value, value_0);
        vst1q_f16(dst, vbslq_f16(lessThanZero, mulSlope, value));
        src += ARMV82_CHANNEL_UNIT;
        dst += ARMV82_CHANNEL_UNIT;
    }
    if (remain > 0) {
        float16_t tempSrc[ARMV82_CHANNEL_UNIT];
        float16_t tempDst[ARMV82_CHANNEL_UNIT];
        ::memcpy(tempSrc, src, remain * sizeof(int16_t));
        float16x8_t value        = vld1q_f16(tempSrc);
        float16x8_t mulSlope     = vmulq_f16(value, slopeV);
        uint16x8_t lessThanZero = vcleq_f16(value, value_0);
        vst1q_f16(tempDst, vbslq_f16(lessThanZero, mulSlope, value));
        ::memcpy(dst, tempDst, remain * sizeof(int16_t));
    }
}

static void _MNNArm82ReluWithChannel(FLOAT16 *dst, const FLOAT16 *src, size_t length) {
    float16x8_t value_0 = vmovq_n_f16(0);
    auto lC8 = length / ARMV82_CHANNEL_UNIT;
    auto remain = length % ARMV82_CHANNEL_UNIT;
    for (int i = 0; i < lC8; ++i) {
        float16x8_t value        = vld1q_f16(src);
        uint16x8_t lessThanZero = vcleq_f16(value, value_0);

        vst1q_f16(dst, vbslq_f16(lessThanZero, value_0, value));
        dst += ARMV82_CHANNEL_UNIT;
        src += ARMV82_CHANNEL_UNIT;
    }
    if (remain > 0) {
        float16_t tempSrc[ARMV82_CHANNEL_UNIT];
        float16_t tempDst[ARMV82_CHANNEL_UNIT];
        ::memcpy(tempSrc, src, remain * sizeof(int16_t));
        float16x8_t value        = vld1q_f16(tempSrc);
        uint16x8_t lessThanZero = vcleq_f16(value, value_0);
        vst1q_f16(tempDst, vbslq_f16(lessThanZero, value_0, value));
        ::memcpy(dst, tempDst, remain * sizeof(int16_t));
    }
}

void Arm82Relu::reluWithSlopeChannel(float* dstO, const float* srcO, const float* slopeO, size_t sizeQuad, size_t depthQuad) {
    auto dst = (FLOAT16*)dstO;
    auto src = (const FLOAT16*)srcO;
    auto slope = (const FLOAT16*)slopeO;
    for (int z=0; z<depthQuad; ++z) {
        auto dstZ = dst + z * 8 * sizeQuad;
        auto srcZ = src + z * 8 * sizeQuad;
        auto slopeZ = slope + 8 * z;
        _MNNArm82PReluWithChannel(dstZ, srcZ, slopeZ, sizeQuad);
    }
}

} // namespace MNN

#endif
