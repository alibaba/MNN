//
//  CPUScale.cpp
//  MNN
//
//  Created by MNN on 2018/08/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUScale.hpp"
#include "CPUBackend.hpp"
#include "CommonOptFunction.h"
#include "Macro.h"
#include "TensorUtils.hpp"

namespace MNN {
CPUScale::CPUScale(const Op* op, Backend* bn) : MNN::Execution(bn) {
    auto scale      = op->main_as_Scale();
    int outputCount = scale->scaleData()->size();
    mScale.reset(ALIGN_UP4(outputCount));
    mScale.clear();
    ::memcpy(mScale.get(), scale->scaleData()->data(), outputCount * sizeof(float));

    mBias.reset(ALIGN_UP4(outputCount));
    mBias.clear();
    if (nullptr != scale->biasData() && nullptr != scale->biasData()->data()) {
        ::memcpy(mBias.get(), scale->biasData()->data(), outputCount * sizeof(float));
    }
}
ErrorCode CPUScale::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];
    if (TensorUtils::getDescribe(input)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
        auto batchSize   = input->buffer().dim[0].stride;
        auto batch       = input->buffer().dim[0].extent;
        auto depthQuad   = UP_DIV(input->channel(), 4);
        auto planeNumber = input->width() * input->height();
        for (int i = 0; i < batch; ++i) {
            MNNScaleAndAddBias(output->host<float>() + batchSize * i, input->host<float>() + batchSize * i, mBias.get(),
                               mScale.get(), planeNumber, depthQuad);
        }
        return NO_ERROR;
    }
    MNN_ASSERT(TensorUtils::getDescribe(input)->dimensionFormat == MNN_DATA_FORMAT_NHWC);

    auto channel = input->channel();
    auto outside = input->elementSize() / channel;
    MNNScaleAndAddBiasOutside(output->host<float>(), input->host<float>(), mBias.get(), mScale.get(), outside, channel);

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
