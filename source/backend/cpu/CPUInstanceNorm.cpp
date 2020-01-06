//
//  CPUInstanceNorm.hpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUInstanceNorm.hpp"
#include <math.h>
#include "backend/cpu/CPUBackend.hpp"
#include "core/Concurrency.h"
#include <MNN/MNNDefine.h>
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

namespace MNN {

CPUInstanceNorm::CPUInstanceNorm(Backend* backend, const MNN::Op* op) : Execution(backend) {
    auto normParam     = op->main_as_BatchNorm();
    const int channels = normParam->channels();
    mEpsilon           = normParam->epsilon();
    mScale.reset(ALIGN_UP4(channels));
    mScale.clear();
    if (normParam->slopeData() && normParam->slopeData()->data()) {
        ::memcpy(mScale.get(), normParam->slopeData()->data(), channels * sizeof(float));
    }

    mBias.reset(ALIGN_UP4(channels));
    mBias.clear();
    if (normParam->biasData() && normParam->biasData()->data()) {
        ::memcpy(mBias.get(), normParam->biasData()->data(), channels * sizeof(float));
    }
}

ErrorCode CPUInstanceNorm::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    MNN_ASSERT(3 == inputs.size());
    MNN_ASSERT(1 == outputs.size());
    auto input = inputs[0];
    MNN_ASSERT(MNN_DATA_FORMAT_NC4HW4 == TensorUtils::getDescribe(input)->dimensionFormat);
    auto mean                = inputs[1];
    auto variance            = inputs[2];
    auto output              = outputs[0];
    const int batch          = input->batch();
    const int batchStride    = input->stride(0);
    const int channelsDiv4   = UP_DIV(input->channel(), 4);
    const int inImageSize    = input->stride(1);
    const float* scalePtr    = mScale.get();
    const float* biasPtr     = mBias.get();
    const float* meanPtr     = mean->host<float>();
    const float* variancePtr = variance->host<float>();

    for (int b = 0; b < batch; ++b) {
        const float* batchMeanPtr     = meanPtr + b * mean->stride(0);
        const float* batchVariancePtr = variancePtr + b * variance->stride(0);
        const float* batchInputPtr    = input->host<float>() + b * batchStride;
        float* batchOutputPtr         = output->host<float>() + b * batchStride;
        MNN_CONCURRENCY_BEGIN(ic, channelsDiv4);
        const int channelOffset       = (int)ic << 2;
        const float* channelsInputPtr = batchInputPtr + channelOffset * inImageSize;
        float* channelsOutputPtr      = batchOutputPtr + channelOffset * inImageSize;
#ifdef MNN_USE_NEON
        float32x4_t epsilon       = vdupq_n_f32(mEpsilon);
        float32x4_t batchVariance = vld1q_f32(batchVariancePtr + channelOffset);
        float32x4_t meanValue     = vld1q_f32(batchMeanPtr + channelOffset);
        float32x4_t scaleValue    = vld1q_f32(scalePtr + channelOffset);
        float32x4_t biasVaule     = vld1q_f32(biasPtr + channelOffset);
        float32x4_t rsqrt         = vrsqrteq_f32(batchVariance + epsilon);

        float32x4_t gamma = vmulq_f32(scaleValue, rsqrt);
        float32x4_t beta  = vsubq_f32(biasVaule, meanValue * gamma);
        for (int i = 0; i < inImageSize; ++i) {
            float32x4_t value = vld1q_f32(channelsInputPtr + i * 4);
            vst1q_f32(channelsOutputPtr + i * 4, value * gamma + beta);
        }

#else
        float gamma[4];
        float beta[4];
        for (int k = 0; k < 4; ++k) {
            const int index = channelOffset + k;
            gamma[k]        = scalePtr[index] / sqrt(batchVariancePtr[index] + mEpsilon);
            beta[k] = biasPtr[index] - scalePtr[index] * batchMeanPtr[index] / sqrt(batchVariancePtr[index] + mEpsilon);
        }

        for (int i = 0; i < inImageSize; ++i) {
            for (int k = 0; k < 4; ++k) {
                channelsOutputPtr[i * 4 + k] = channelsInputPtr[i * 4 + k] * gamma[k] + beta[k];
            }
        }
#endif
        MNN_CONCURRENCY_END();
    }

    return NO_ERROR;
}

class CPUInstanceNormCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new CPUInstanceNorm(backend, op);
    }
};

REGISTER_CPU_OP_CREATOR(CPUInstanceNormCreator, OpType_InstanceNorm);

} // namespace MNN
