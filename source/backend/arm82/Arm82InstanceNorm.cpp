//
//  Arm82InstanceNorm.hpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#if defined(__ANDROID__) || defined(__aarch64__)

#include "Arm82Backend.hpp"
#include "Arm82OptFunc.hpp"
#include "Arm82InstanceNorm.hpp"
#include "MNN_generated.h"
#include "core/Concurrency.h"
#include <MNN/MNNDefine.h>
#include <cmath>
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

namespace MNN {

Arm82InstanceNorm::Arm82InstanceNorm(Backend* backend, const MNN::Op* op) : Execution(backend) {
    auto normParam     = op->main_as_BatchNorm();
    const int channels = normParam->channels();
    mEpsilon           = normParam->epsilon();
    mScale.reset(ALIGN_UP8(channels));
    mScale.clear();
    if (normParam->slopeData() && normParam->slopeData()->data()) {
        MNNSlowCopy<FLOAT16, float>(mScale.get(), normParam->slopeData()->data(), channels);
    }

    mBias.reset(ALIGN_UP8(channels));
    mBias.clear();
    if (normParam->biasData() && normParam->biasData()->data()) {
        MNNSlowCopy<FLOAT16, float>(mBias.get(), normParam->biasData()->data(), channels);
    }
}

ErrorCode Arm82InstanceNorm::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    MNN_ASSERT(3 == inputs.size());
    MNN_ASSERT(1 == outputs.size());
    
    auto input = inputs[0], mean = inputs[1], variance = inputs[2], output = outputs[0];
    const int batch = input->batch(), imageSize = input->stride(1);
    auto scalePtr = mScale.get(), biasPtr = mBias.get();
    const int threadNum = ((Arm82Backend*)backend())->numberThread();
    const int channelBlock = UP_DIV(input->channel(), 8);

    for (int b = 0; b < batch; ++b) {
        auto inputPtr = input->host<FLOAT16>() + b * ARM82TensorStrideHelper(input, 0);
        auto meanPtr = mean->host<FLOAT16>() + b * ARM82TensorStrideHelper(mean, 0);
        auto variancePtr = variance->host<FLOAT16>() + b * ARM82TensorStrideHelper(variance, 0);
        auto outputPtr = output->host<FLOAT16>() + b * ARM82TensorStrideHelper(output, 0);
        
        MNN_CONCURRENCY_BEGIN(tId, threadNum) {
            const int step = UP_DIV(channelBlock, threadNum) * 8, start = tId * step, end = ALIMIN(start + step, channelBlock);
            for (int c = start; c < end; c += 8) {
                auto inputPtrZ = inputPtr + c * imageSize;
                auto outputPtrZ = outputPtr + c * imageSize;
#ifdef MNN_USE_NEON
                float16x8_t meanVec = vld1q_f16(meanPtr + c), varVec = vld1q_f16(variancePtr + c);
                float16x8_t scaleVec = vld1q_f16(scalePtr + c), biasVec = vld1q_f16(biasPtr + c);
                float16x8_t epsVec = vdupq_n_f16(mEpsilon), rsqrtVec = vrsqrteq_f16(varVec + epsVec);
                
                float16x8_t gamma = vmulq_f16(scaleVec, rsqrtVec);
                float16x8_t beta = vsubq_f16(biasVec, vmulq_f16(meanVec, gamma));
                for (int i = 0; i < imageSize; ++i) {
                    float16x8_t in = vld1q_f16(inputPtr + i * 8);
                    vst1q_f16(outputPtrZ + i * 8, vaddq_f16(vmulq_f16(in, gamma), beta));
                }
#else
                FLOAT16 gamma[8], beta[8];
                for (int k = 0; k < 8; ++k) {
                    int index = c + k;
                    gamma[k] = scalePtr[index] / sqrt(variancePtr[index] + mEpsilon);
                    beta[k] = biasPtr[index] - gamma[k] * meanPtr[index];
                }
                for (int i = 0; i < imageSize; ++i) {
                    for (int k = 0; k < 8; ++k) {
                        outputPtrZ[i * 8 + k] = inputPtrZ[i * 8 + k] * gamma[k] + beta[k];
                    }
                }
#endif
            }
        }
        MNN_CONCURRENCY_END();
    }

    return NO_ERROR;
}

class Arm82InstanceNormCreator : public Arm82Backend::Arm82Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new Arm82InstanceNorm(backend, op);
    }
};

REGISTER_ARM82_OP_CREATOR(OpType_InstanceNorm, Arm82InstanceNormCreator);

} // namespace MNN
#endif
