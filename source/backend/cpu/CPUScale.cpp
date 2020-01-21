//
//  CPUScale.cpp
//  MNN
//
//  Created by MNN on 2018/08/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUScale.hpp"
#include "CPUBackend.hpp"
#include "compute/CommonOptFunction.h"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "core/Concurrency.h"

namespace MNN {
CPUScale::CPUScale(const Op* op, Backend* bn) : MNN::Execution(bn) {
    auto scale      = op->main_as_Scale();
    int outputCount = scale->scaleData()->size();
    mScaleBias.reset(
                     Tensor::createDevice<float>(
                                           {2, ALIGN_UP4(outputCount)}
                                           ));
    auto res = bn->onAcquireBuffer(mScaleBias.get(), Backend::STATIC);
    if (!res) {
        MNN_ERROR("Error for alloc buffer for CPUScale\n");
        mScaleBias = nullptr;
        mValid = false;
        return;
    }
    ::memset(mScaleBias->host<float>(), 0, mScaleBias->size());
    ::memcpy(mScaleBias->host<float>(), scale->scaleData()->data(), outputCount * sizeof(float));
    if (nullptr != scale->biasData() && nullptr != scale->biasData()->data()) {
        ::memcpy(mScaleBias->host<float>() + ALIGN_UP4(outputCount), scale->biasData()->data(), outputCount * sizeof(float));
    }
}
CPUScale::~CPUScale() {
    if (nullptr != mScaleBias) {
        backend()->onReleaseBuffer(mScaleBias.get(), Backend::STATIC);
    }
}
ErrorCode CPUScale::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];
    auto scalePtr = mScaleBias->host<float>();
    auto biasPtr = mScaleBias->host<float>() + 1 * mScaleBias->length(1);
    if (TensorUtils::getDescribe(input)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
        auto batch       = input->buffer().dim[0].extent;
        auto depthQuad   = UP_DIV(input->channel(), 4);
        int planeNumber = 1;
        for (int i = 2; i < input->buffer().dimensions; ++i) {
            planeNumber *= input->length(i);
        }
        auto depthStride = planeNumber * 4;
        auto totalDepth = batch * depthQuad;
        int numberThread = ((CPUBackend*)backend())->threadNumber();
        MNN_CONCURRENCY_BEGIN(tId, numberThread) {
            for (int i = tId; i < totalDepth; i+=numberThread) {
                MNNScaleAndAddBias(output->host<float>() + depthStride * i, input->host<float>() + depthStride * i, biasPtr + 4 * i,
                                   scalePtr + 4 * i, planeNumber, 1);
            }
        }
        MNN_CONCURRENCY_END();
        return NO_ERROR;
    }
    MNN_ASSERT(TensorUtils::getDescribe(input)->dimensionFormat == MNN_DATA_FORMAT_NHWC);

    auto channel = input->channel();
    auto outside = input->elementSize() / channel;
    MNNScaleAndAddBiasOutside(output->host<float>(), input->host<float>(), biasPtr, scalePtr, outside, channel);

    return NO_ERROR;
}
class CPUScaleCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new CPUScale(op, backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPUScaleCreator, OpType_Scale);
} // namespace MNN
