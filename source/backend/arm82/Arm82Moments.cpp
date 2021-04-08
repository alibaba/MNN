//
//  Arm82Moments.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#if defined(__ANDROID__) || defined(__aarch64__)

#include "Arm82Moments.hpp"
#include "Arm82Backend.hpp"
#include "Arm82Vec.hpp"
#include "core/Concurrency.h"
#include <MNN/MNNDefine.h>
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

using Vec = MNN::Math::Vec<FLOAT16, 8>;
namespace MNN {

Arm82Moments::Arm82Moments(Backend *backend, const MNN::Op *op) : Execution(backend) {
    auto momentsParam = op->main_as_MomentsParam();
    if (momentsParam->dim()) {
        for (int i = 0; i < momentsParam->dim()->size(); ++i) {
            mAxis.push_back(momentsParam->dim()->data()[i]);
        }
    }
    mKeepDims = momentsParam->keepDims();
    MNN_ASSERT(DataType_DT_FLOAT == momentsParam->dType());
}

ErrorCode Arm82Moments::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    return NO_ERROR;
}

void Arm82Moments::calculateMean(const FLOAT16 *src, FLOAT16 *mean, int channelBlock, int planeNumber) {
    const int numberThread = ((Arm82Backend*)backend())->numberThread();
    MNN_CONCURRENCY_BEGIN(tId, numberThread) {
        int step = UP_DIV(channelBlock, numberThread), start = tId * step, end = ALIMIN(start + step, channelBlock);
        for (int z = start; z < end; ++z) {
            const FLOAT16* srcZ = src + z * planeNumber * 8;
            FLOAT16* meanZ = mean + z * 8;
            
            Vec sum(0);
            for (int i = 0; i < planeNumber; ++i) {
                sum = sum + Vec::load(srcZ + i * 8);
            }
            Vec result = sum / (float)planeNumber;
            Vec::save(meanZ, result);
        }
        
    } MNN_CONCURRENCY_END();
}

void Arm82Moments::calculateVariance(const FLOAT16 *src, const FLOAT16 *mean, FLOAT16* var, int channelBlock, int planeNumber) {
    const int numberThread = ((Arm82Backend*)backend())->numberThread();
    MNN_CONCURRENCY_BEGIN(tId, numberThread) {
        int step = UP_DIV(channelBlock, numberThread), start = tId * step, end = ALIMIN(start + step, channelBlock);
        for (int z = start; z < end; ++z) {
            const FLOAT16* srcZ = src + z * planeNumber * 8, *meanZ = mean + z * 8;
            FLOAT16* varZ = var + z * 8;
            
            Vec sum(0), meanVal = Vec::load(meanZ);
            for (int i = 0; i < planeNumber; ++i) {
                Vec diff = Vec::load(srcZ + i * 8) - meanVal;
                sum = sum + diff * diff;
            }
            Vec result = sum / (float)planeNumber;
            Vec::save(varZ, result);
        }
        
    } MNN_CONCURRENCY_END();
}

ErrorCode Arm82Moments::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(1 == inputs.size());
    MNN_ASSERT(2 == outputs.size());
    auto input = inputs[0], mean = outputs[0], variance = outputs[1];

    // the layout of Moments is NC4HW4, now only support for calculating Moments along height and width
    MNN_ASSERT(MNN_DATA_FORMAT_NC4HW4 == TensorUtils::getDescribe(input)->dimensionFormat);
    MNN_ASSERT(mKeepDims);
    MNN_ASSERT(mAxis.size() == 2 && mAxis[0] == 2 && mAxis[1] == 3);

    const int batch = input->batch(), channelBlock = UP_DIV(mean->channel(), 8);
    const int inBatchStride = ARM82TensorStrideHelper(input, 0), outBatchStride = ARM82TensorStrideHelper(mean, 0);
    const int planeNumber = ARM82TensorStrideHelper(input, 1);
    // mean
    for (int b = 0; b < batch; ++b) {
        const FLOAT16* srcPtr = input->host<FLOAT16>() + b * inBatchStride;
        FLOAT16* meanPtr = mean->host<FLOAT16>() + b * outBatchStride;
        calculateMean(srcPtr, meanPtr, channelBlock, planeNumber);
    }
    // variance
    for (int b = 0; b < batch; ++b) {
        const FLOAT16* srcPtr = input->host<FLOAT16>() + b * inBatchStride;
        const FLOAT16* meanPtr = mean->host<FLOAT16>() + b * outBatchStride;
        FLOAT16* variancePtr = variance->host<FLOAT16>() + b * outBatchStride;
        calculateVariance(srcPtr, meanPtr, variancePtr, channelBlock, planeNumber);
    }

    return NO_ERROR;
}

class Arm82MomentsCreator : public Arm82Backend::Arm82Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        return new Arm82Moments(backend, op);
    }
};

REGISTER_ARM82_OP_CREATOR(OpType_Moments, Arm82MomentsCreator);

} // namespace MNN
#endif
