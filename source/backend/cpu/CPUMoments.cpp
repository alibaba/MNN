//
//  CPUMoments.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUMoments.hpp"
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

CPUMoments::CPUMoments(Backend *backend, const MNN::Op *op) : Execution(backend) {
    auto momentsParam = op->main_as_MomentsParam();
    if (momentsParam->dim()) {
        for (int i = 0; i < momentsParam->dim()->size(); ++i) {
            mAxis.push_back(momentsParam->dim()->data()[i]);
        }
    }
    mKeepDims = momentsParam->keepDims();
    MNN_ASSERT(DataType_DT_FLOAT == momentsParam->dType());
}

ErrorCode CPUMoments::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input = inputs[0];
    mMidBuffer.reset(new Tensor(input->dimensions()));
    TensorUtils::copyShape(input, mMidBuffer.get(), true);
    backend()->onAcquireBuffer(mMidBuffer.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mMidBuffer.get(), Backend::DYNAMIC);
    return NO_ERROR;
}
// calculate the Mean of the Image(Height,Width)
void CPUMoments::CalculateMean(const float *src, float *dst, int batch, int channelDiv4, int inImageSize,
                               int inBatchStride, int outBatchStride) {
    for (int b = 0; b < batch; ++b) {
        MNN_CONCURRENCY_BEGIN(oc, channelDiv4);
        const float *channelSrcPtr = src + b * inBatchStride + oc * inImageSize * 4;
        float *channelDstPtr       = dst + b * outBatchStride + oc * 4;
#ifdef MNN_USE_NEON
        float32x4_t sum = vdupq_n_f32(0.0);
        for (int i = 0; i < inImageSize; ++i) {
            float32x4_t value = vld1q_f32(channelSrcPtr + i * 4);
            sum               = vaddq_f32(sum, value);
        }
        float32x4_t lengthReciprocal = vdupq_n_f32(1.0f / inImageSize);
        float32x4_t result           = vmulq_f32(sum, lengthReciprocal);
        vst1q_f32(channelDstPtr, result);
#else
        std::vector<float> sum(4, 0.0f);
        for (int i = 0; i < inImageSize; ++i) {
            for (int k = 0; k < 4; ++k) {
                sum[k] += channelSrcPtr[i * 4 + k];
            }
        }
        for (int j = 0; j < 4; ++j) {
            channelDstPtr[j] = sum[j] / inImageSize;
        }
#endif
        MNN_CONCURRENCY_END();
    }
}

ErrorCode CPUMoments::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(1 == inputs.size());
    MNN_ASSERT(2 == outputs.size());
    auto input    = inputs[0];
    auto mean     = outputs[0];
    auto variance = outputs[1];

    // the layout of Moments is NC4HW4, now only support for calculating Moments along height and width
    MNN_ASSERT(MNN_DATA_FORMAT_NC4HW4 == TensorUtils::getDescribe(input)->dimensionFormat);
    MNN_ASSERT(mKeepDims);
    MNN_ASSERT(mAxis.size() == 2 && mAxis[0] == 2 && mAxis[1] == 3);

    const int batch       = input->batch();
    const int channelDiv4 = UP_DIV(mean->channel(), 4);

    const int inBatchStride  = input->stride(0);
    const int inImagSize     = input->stride(1);
    const int outBatchStride = mean->stride(0);
    const float *src         = input->host<float>();
    float *meanPtr           = mean->host<float>();
    float *variancePtr       = variance->host<float>();
    // mean
    CalculateMean(src, meanPtr, batch, channelDiv4, inImagSize, inBatchStride, outBatchStride);

    float *subMeanSqaure = mMidBuffer->host<float>();
    // variance
    for (int b = 0; b < batch; ++b) {
        MNN_CONCURRENCY_BEGIN(oc, channelDiv4)
        const float *channelMean       = meanPtr + b * outBatchStride + oc * 4;
        const float *channelSrcPtr     = src + b * outBatchStride + oc * inImagSize * 4;
        float *channelSubMeanSqaurePtr = subMeanSqaure + b * outBatchStride + oc * inImagSize * 4;

        for (int i = 0; i < inImagSize; ++i) {
#ifdef MNN_USE_NEON
            float32x4_t value = vld1q_f32(channelSrcPtr + i * 4);
            float32x4_t mean4 = vld1q_f32(channelMean);
            float32x4_t diff  = vsubq_f32(value, mean4);
            vst1q_f32(channelSubMeanSqaurePtr + i * 4, diff * diff);
#else
            for (int k = 0; k < 4; ++k) {
                auto subData                       = channelSrcPtr[i * 4 + k] - channelMean[k];
                channelSubMeanSqaurePtr[i * 4 + k] = powf(subData, 2);
            }
#endif
        }
        MNN_CONCURRENCY_END();
    }
    CalculateMean(subMeanSqaure, variancePtr, batch, channelDiv4, inImagSize, inBatchStride, outBatchStride);

    return NO_ERROR;
}

class CPUMomentsCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        return new CPUMoments(backend, op);
    }
};

REGISTER_CPU_OP_CREATOR(CPUMomentsCreator, OpType_Moments);

} // namespace MNN
