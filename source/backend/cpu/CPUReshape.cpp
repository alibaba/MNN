//
//  CPUReshape.cpp
//  MNN
//
//  Created by MNN on 2018/07/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUReshape.hpp"
#include "CPUBackend.hpp"
#include "CommonOptFunction.h"
#include "Macro.h"
#include "TensorUtils.hpp"

namespace MNN {

CPUReshape::CPUReshape(Backend *b, MNN_DATA_FORMAT midFormat) : MNN::Execution(b), mStorage(2) {
    mMidFormat = midFormat;
}

ErrorCode CPUReshape::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(1 == inputs.size() || 2 == inputs.size());
    MNN_ASSERT(1 == outputs.size());

    auto input    = inputs[0];
    auto output   = outputs[0];
    int totalSize = 1;

    mWrapTensorForInput.buffer().type  = inputs[0]->buffer().type;
    mWrapTensorForOutput.buffer().type = inputs[0]->buffer().type;

    if (TensorUtils::getDescribe(input)->dimensionFormat != MNN_DATA_FORMAT_NC4HW4) {
        return NO_ERROR;
    }
    for (int i = 0; i < input->buffer().dimensions; ++i) {
        totalSize *= input->buffer().dim[i].extent;
    }
    TensorUtils::getDescribe(&mStorage)->dimensionFormat = MNN_DATA_FORMAT_NCHW;
    mStorage.buffer().dim[0].extent = 1;
    mStorage.buffer().dim[1].extent = totalSize;
    mStorage.buffer().dimensions    = 2;
    mStorage.buffer().type          = input->getType();
    backend()->onAcquireBuffer(&mStorage, Backend::DYNAMIC);
    backend()->onReleaseBuffer(&mStorage, Backend::DYNAMIC);
    mWrapTensorForInput.buffer().host = mStorage.buffer().host;
    mWrapTensorForOutput.buffer().host = mStorage.buffer().host;
    if (MNN_DATA_FORMAT_NHWC == mMidFormat) {
        TensorUtils::getDescribe(&mWrapTensorForInput)->dimensionFormat  = MNN_DATA_FORMAT_NHWC;
        TensorUtils::getDescribe(&mWrapTensorForOutput)->dimensionFormat = MNN_DATA_FORMAT_NHWC;
        mWrapTensorForInput.buffer().dimensions = 4;
        mWrapTensorForOutput.buffer().dimensions = 4;
        mWrapTensorForInput.setLength(0, input->batch());
        mWrapTensorForInput.setLength(1, input->height());
        mWrapTensorForInput.setLength(2, input->width());
        mWrapTensorForInput.setLength(3, input->channel());
        mWrapTensorForOutput.setLength(0, output->batch());
        mWrapTensorForOutput.setLength(1, output->height());
        mWrapTensorForOutput.setLength(2, output->width());
        mWrapTensorForOutput.setLength(3, output->channel());
    } else {
        TensorUtils::getDescribe(&mWrapTensorForInput)->dimensionFormat  = MNN_DATA_FORMAT_NCHW;
        TensorUtils::getDescribe(&mWrapTensorForOutput)->dimensionFormat = MNN_DATA_FORMAT_NCHW;
        TensorUtils::copyShape(inputs[0], &mWrapTensorForInput);
        TensorUtils::copyShape(outputs[0], &mWrapTensorForOutput);
    }
    TensorUtils::setLinearLayout(&mWrapTensorForInput);
    TensorUtils::setLinearLayout(&mWrapTensorForOutput);
    return NO_ERROR;
}

ErrorCode CPUReshape::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(1 == inputs.size() || 2 == inputs.size());
    MNN_ASSERT(1 == outputs.size());
    if (TensorUtils::getDescribe(inputs[0])->dimensionFormat != MNN_DATA_FORMAT_NC4HW4) {
        ::memcpy(outputs[0]->host<float>(), inputs[0]->host<float>(), inputs[0]->size());
        return NO_ERROR;
    }

    auto input  = inputs[0];
    auto output = outputs[0];

    backend()->onCopyBuffer(input, &mWrapTensorForInput);
    backend()->onCopyBuffer(&mWrapTensorForOutput, output);

    return NO_ERROR;
}

class CPUReshapeCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        return new CPUReshape(backend, op->main_as_Reshape()->dimType());
    }
};

REGISTER_CPU_OP_CREATOR(CPUReshapeCreator, OpType_Reshape);

} // namespace MNN
