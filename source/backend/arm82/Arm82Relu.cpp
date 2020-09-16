//
//  Arm82Relu.cpp
//  MNN
//
//  Created by MNN on 2020/2/13.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef __aarch64__
#include "backend/arm82/Arm82Relu.hpp"
#include "MNN_generated.h"
#include "backend/arm82/Arm82Backend.hpp"
#include "backend/arm82/Arm82OptFunc.hpp"
#include "core/Concurrency.h"
#include "core/Macro.h"
#include "half.hpp"

#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

namespace MNN {

static void _MNNArm82ReluWithChannel(FLOAT16 *dst, const FLOAT16 *src, const FLOAT16 *slope, size_t length) {
#ifdef MNN_USE_NEON
    float16x8_t value_0 = vmovq_n_f16(0);
    float16x8_t slopeV  = vld1q_f16(slope);
#endif

    for (int i = 0; i < length; ++i) {
#ifdef MNN_USE_NEON
        float16x8_t value        = vld1q_f16(src + i * ARMV82_CHANNEL_UNIT);
        float16x8_t mulSlope     = vmulq_f16(value, slopeV);
        float16x8_t lessThanZero = vcleq_f16(value, value_0);

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
#ifdef MNN_USE_NEON
    float16x8_t value_0 = vmovq_n_f16(0);
    float16x8_t slopeV  = vmovq_n_f16(slope);
#endif

    for (int i = 0; i < length; ++i) {
#ifdef MNN_USE_NEON
        float16x8_t value        = vld1q_f16(src + i * ARMV82_CHANNEL_UNIT);
        float16x8_t mulSlope     = vmulq_f16(value, slopeV);
        float16x8_t lessThanZero = vcleq_f16(value, value_0);

        vst1q_f16(dst + i * ARMV82_CHANNEL_UNIT, vbslq_f16(lessThanZero, mulSlope, value));
#else
        for (int j = 0; j < ARMV82_CHANNEL_UNIT; ++j) {
            int index = i * ARMV82_CHANNEL_UNIT + j;
            if (src[index] < 0) {
                dst[index] = src[index] * slope;
            } else {
                dst[index] = src[index];
            }
        }

#endif
    }
}

Arm82Relu::Arm82Relu(Backend *backend, const Op *op) : Execution(backend) {
    mSlope = op->main_as_Relu()->slope();
}

ErrorCode Arm82Relu::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input            = inputs[0];
    auto output           = outputs[0];
    const int dimension = input->dimensions();
    MNN_ASSERT(4 == dimension);
    const int batch           = input->batch();
    const int channel         = input->channel();
    const int width           = input->width();
    const int height          = input->height();
    const int channelDivUnit  = UP_DIV(channel, ARMV82_CHANNEL_UNIT);
    const int batchAndChannel = batch * channelDivUnit;
    const int plane           = width * height;

    const auto src = input->host<FLOAT16>();
    auto dst       = output->host<FLOAT16>();
    FLOAT16 slopeHalf = half_float::half(mSlope);

    mThreadNumbers = static_cast<Arm82Backend *>(backend())->numberThread();
    MNN_CONCURRENCY_BEGIN(tId, mThreadNumbers)
    for (int b = tId; b < batchAndChannel; b += mThreadNumbers) {
        auto curChannel = b % channelDivUnit;
        _MNNArm82LeakyReluWithChannel(dst + b * plane * ARMV82_CHANNEL_UNIT, 
                                      src + b * plane * ARMV82_CHANNEL_UNIT,
                                      slopeHalf, 
                                      plane);
    }
#ifdef MNN_USE_THREAD_POOL
    MNN_CONCURRENCY_ARM82_END();
#else
    MNN_CONCURRENCY_END();
#endif

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
    auto slopePtr = mSlope->host<FLOAT16>();
    MNNQuantizeFP16(slopePtr, param->slope()->data(), slopeLength);
}

ErrorCode Arm82PRelu::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    const auto input = inputs[0];
    auto output      = outputs[0];

    const int dimension = input->dimensions();
    MNN_ASSERT(4 == dimension);
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
        _MNNArm82ReluWithChannel(dstPtr + b * plane * ARMV82_CHANNEL_UNIT, srcPtr + b * plane * ARMV82_CHANNEL_UNIT,
                                 slopePtr + curChannel * ARMV82_CHANNEL_UNIT, plane);
    }
#ifdef MNN_USE_THREAD_POOL
    MNN_CONCURRENCY_ARM82_END();
#else
    MNN_CONCURRENCY_END();
#endif

    return NO_ERROR;
}

class Arm82ReluCreator : public Arm82Backend::Arm82Creator {
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        if (op->type() == OpType_ReLU) {
            return new Arm82Relu(backend, op);
        }

        auto preluParam = op->main_as_PRelu();
        if (preluParam->slopeCount() == 1) {
            // TODO, support Prelu with one slope
            MNN_ERROR("[MNN ERROR]Arm82 not support prelu with one slope NOW");
            return nullptr;
        }
        return new Arm82PRelu(backend, op);
    }
};

REGISTER_ARM82_OP_CREATOR(OpType_ReLU, Arm82ReluCreator);
REGISTER_ARM82_OP_CREATOR(OpType_PReLU, Arm82ReluCreator);

} // namespace MNN

#endif
