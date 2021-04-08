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
#include "MNN_generated.h"
#include "Arm82Backend.hpp"
#include "Arm82OptFunc.hpp"
#include "core/Concurrency.h"
#include "core/Macro.h"
#include "half.hpp"
#include <algorithm>
#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

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

Arm82Relu::Arm82Relu(Backend *backend, float slope) : Execution(backend) {
    mSlope = slope;
}

ErrorCode Arm82Relu::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input            = inputs[0];
    auto output           = outputs[0];
    auto size           = ARM82TensorElementSizeHelper(input);
    auto schedule = static_cast<CPUBackend*>(backend())->multiThreadDivide(size);
    
    const auto src = input->host<FLOAT16>();
    auto dst       = output->host<FLOAT16>();

    if (abs(mSlope) < std::numeric_limits<float>::epsilon()) {
        // relu
        MNN_CONCURRENCY_BEGIN(tId, schedule.second) {
            int start = schedule.first * (int)tId;
            int realSize = schedule.first;
            if (tId == schedule.second -1 ) {
                realSize = size - start;
            }

            _MNNArm82ReluWithChannel(dst + start,
                                     src + start, realSize);
        } MNN_CONCURRENCY_END();
    } else {
        // leakyrelu
        FLOAT16 slopeHalf = half_float::half(mSlope);
        MNN_CONCURRENCY_BEGIN(tId, schedule.second) {
            int start = schedule.first * (int)tId;
            int realSize = schedule.first;
            if (tId == schedule.second -1 ) {
                realSize = size - start;
            }

            _MNNArm82LeakyReluWithChannel(dst + start,
                                     src + start, slopeHalf, realSize);
        } MNN_CONCURRENCY_END();
    }

    return NO_ERROR;
}

Arm82PRelu::Arm82PRelu(Backend *backend, const Op *op) : Execution(backend) {
    auto param            = op->main_as_PRelu();
    const int slopeLength = param->slopeCount();
    mSlope.reset(Tensor::createDevice<uint16_t>({slopeLength}));
    auto allocRes = backend->onAcquireBuffer(mSlope.get(), Backend::STATIC);
    if (!allocRes) {
        return;
    }
    auto slopePtr = mSlope->host<int16_t>();
    MNNQuantizeFP16(param->slope()->data(), slopePtr, slopeLength);
}

ErrorCode Arm82PRelu::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    const auto input = inputs[0];
    auto output      = outputs[0];

    const int batch           = input->batch();
    const int channel         = input->channel();
    const int width           = input->width();
    const int height          = input->height();
    const int channelDivUnit  = UP_DIV(channel, ARMV82_CHANNEL_UNIT);
    const int batchAndChannel = batch * channelDivUnit;
    const int plane           = width * height;

    const auto srcPtr   = input->host<FLOAT16>();
    const auto slopePtr = mSlope->host<FLOAT16>();

    auto dstPtr = output->host<FLOAT16>();

    mThreadNumbers = static_cast<Arm82Backend *>(backend())->numberThread();

    MNN_CONCURRENCY_BEGIN(tId, mThreadNumbers)
    for (int b = tId; b < batchAndChannel; ++b) {
        auto curChannel = b % channelDivUnit;
        _MNNArm82PReluWithChannel(dstPtr + b * plane * ARMV82_CHANNEL_UNIT, srcPtr + b * plane * ARMV82_CHANNEL_UNIT,
                                 slopePtr + curChannel * ARMV82_CHANNEL_UNIT, plane);
    }
    MNN_CONCURRENCY_END();

    return NO_ERROR;
}

class Arm82ReluCreator : public Arm82Backend::Arm82Creator {
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        if (op->type() == OpType_ReLU) {
            float slope = 0.0f;
            if (nullptr != op->main_as_Relu()) {
                slope = op->main_as_Relu()->slope();
            }
            return new Arm82Relu(backend, slope);
        }

        auto preluParam = op->main_as_PRelu();
        if (preluParam->slopeCount() == 1) {
            return new Arm82Relu(backend, op->main_as_PRelu()->slope()->data()[0]);
        }
        return new Arm82PRelu(backend, op);
    }
};

REGISTER_ARM82_OP_CREATOR(OpType_ReLU, Arm82ReluCreator);
REGISTER_ARM82_OP_CREATOR(OpType_PReLU, Arm82ReluCreator);

} // namespace MNN

#endif
