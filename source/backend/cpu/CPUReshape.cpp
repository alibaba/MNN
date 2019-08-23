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

CPUReshape::CPUReshape(Backend *b) : MNN::Execution(b), mStorage(2) {
    // nothing to do
}

ErrorCode CPUReshape::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(1 == inputs.size() || 2 == inputs.size());
    MNN_ASSERT(1 == outputs.size());

    auto input    = inputs[0];
    int totalSize = 1;

    mWrapTensorForInput.buffer().type  = inputs[0]->buffer().type;
    mWrapTensorForOutput.buffer().type = inputs[0]->buffer().type;

    if (TensorUtils::getDescribe(input)->dimensionFormat != MNN_DATA_FORMAT_NC4HW4) {
        return NO_ERROR;
    }
    TensorUtils::getDescribe(&mWrapTensorForInput)->dimensionFormat  = MNN_DATA_FORMAT_NCHW;
    TensorUtils::getDescribe(&mWrapTensorForOutput)->dimensionFormat = MNN_DATA_FORMAT_NCHW;

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

    TensorUtils::copyShape(inputs[0], &mWrapTensorForInput);

    mWrapTensorForInput.buffer().host = mStorage.buffer().host;
    TensorUtils::setLinearLayout(&mWrapTensorForInput);

    TensorUtils::copyShape(outputs[0], &mWrapTensorForOutput);
    mWrapTensorForOutput.buffer().host = mStorage.buffer().host;
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
        return new CPUReshape(backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPUReshapeCreator, OpType_Reshape);

} // namespace MNN
