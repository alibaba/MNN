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

CPUReshape::CPUReshape(Backend *b, MNN_DATA_FORMAT dimType) : MNN::Execution(b), mStorage(2), mDimType(dimType) {
    // nothing to do
}

ErrorCode CPUReshape::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(1 == inputs.size() || 2 == inputs.size());
    MNN_ASSERT(1 == outputs.size());

    auto input    = inputs[0];
    int totalSize = 1;

    mWrapTensorForInput.buffer().type  = inputs[0]->buffer().type;
    mWrapTensorForOutput.buffer().type = inputs[0]->buffer().type;

    if (TensorUtils::getDescribe(input)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
        TensorUtils::getDescribe(&mWrapTensorForInput)->dimensionFormat  = MNN_DATA_FORMAT_NCHW;
        TensorUtils::getDescribe(&mWrapTensorForOutput)->dimensionFormat = MNN_DATA_FORMAT_NCHW;
    } else {
        TensorUtils::getDescribe(&mWrapTensorForInput)->dimensionFormat  = MNN_DATA_FORMAT_NHWC;
        TensorUtils::getDescribe(&mWrapTensorForOutput)->dimensionFormat = MNN_DATA_FORMAT_NHWC;
    }

    for (int i = 0; i < input->buffer().dimensions; ++i) {
        totalSize *= input->buffer().dim[i].extent;
    }

    mStorage.buffer().dim[0].extent = 1;
    mStorage.buffer().dim[1].extent = totalSize;
    mStorage.buffer().dim[1].flags  = 0;
    mStorage.buffer().dimensions    = 2;
    mStorage.buffer().type          = input->getType();
    backend()->onAcquireBuffer(&mStorage, Backend::DYNAMIC);
    backend()->onReleaseBuffer(&mStorage, Backend::DYNAMIC);

    TensorUtils::copyShape(inputs[0], &mWrapTensorForInput);
    if (TensorUtils::getDescribe(input)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4 &&
        mDimType == MNN_DATA_FORMAT_NHWC) {
        TensorUtils::getDescribe(&mWrapTensorForInput)->dimensionFormat = MNN_DATA_FORMAT_NHWC;
        if (mWrapTensorForInput.buffer().dimensions == 4) {
            int channels                               = mWrapTensorForInput.buffer().dim[1].extent;
            mWrapTensorForInput.buffer().dim[1].extent = mWrapTensorForInput.buffer().dim[2].extent;
            mWrapTensorForInput.buffer().dim[2].extent = mWrapTensorForInput.buffer().dim[3].extent;
            mWrapTensorForInput.buffer().dim[3].extent = channels;
        }
    }

    if (input->buffer().dimensions > 1) {
        mWrapTensorForInput.buffer().dim[1].flags = 0;
    }
    mWrapTensorForInput.buffer().host = mStorage.buffer().host;
    TensorUtils::setLinearLayout(&mWrapTensorForInput);

    TensorUtils::copyShape(outputs[0], &mWrapTensorForOutput);
    if (TensorUtils::getDescribe(input)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4 &&
        mDimType == MNN_DATA_FORMAT_NHWC) {
        TensorUtils::getDescribe(&mWrapTensorForOutput)->dimensionFormat = MNN_DATA_FORMAT_NHWC;
        if (mWrapTensorForOutput.buffer().dimensions == 4) {
            int channels                                = mWrapTensorForOutput.buffer().dim[1].extent;
            mWrapTensorForOutput.buffer().dim[1].extent = mWrapTensorForOutput.buffer().dim[2].extent;
            mWrapTensorForOutput.buffer().dim[2].extent = mWrapTensorForOutput.buffer().dim[3].extent;
            mWrapTensorForOutput.buffer().dim[3].extent = channels;
        }
    }
    if (outputs[0]->buffer().dimensions > 1) {
        mWrapTensorForOutput.buffer().dim[1].flags = 0;
    }
    mWrapTensorForOutput.buffer().host = mStorage.buffer().host;
    TensorUtils::setLinearLayout(&mWrapTensorForOutput);

    return NO_ERROR;
}

ErrorCode CPUReshape::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(1 == inputs.size() || 2 == inputs.size());
    MNN_ASSERT(1 == outputs.size());

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
