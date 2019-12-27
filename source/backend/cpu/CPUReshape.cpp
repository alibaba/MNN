//
//  CPUReshape.cpp
//  MNN
//
//  Created by MNN on 2018/07/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUReshape.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {

CPUReshape::CPUReshape(Backend *b, MNN_DATA_FORMAT midFormat) : MNN::Execution(b), mStorage(2) {
    mMidFormat = midFormat;
}

ErrorCode CPUReshape::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(1 == inputs.size() || 2 == inputs.size());
    MNN_ASSERT(1 == outputs.size());

    auto input    = inputs[0];
    auto output   = outputs[0];

    if (TensorUtils::getDescribe(input)->dimensionFormat != MNN_DATA_FORMAT_NC4HW4) {
        return NO_ERROR;
    }

    int totalSize = 1;
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

    auto convertTensorMeta = [&](const Tensor* tensor, Tensor* wrapTensor) {
        wrapTensor->buffer().host       = mStorage.buffer().host;
        wrapTensor->buffer().dimensions = tensor->dimensions();
        wrapTensor->buffer().type       = tensor->buffer().type;
        TensorUtils::getDescribe(wrapTensor)->dimensionFormat = mMidFormat;
        auto tensorFormat      = TensorUtils::getDescribe(tensor)->dimensionFormat;
        bool originCaffeFormat = (tensorFormat == MNN_DATA_FORMAT_NCHW || tensorFormat == MNN_DATA_FORMAT_NC4HW4);
        bool wrapCaffeFormat   = (mMidFormat == MNN_DATA_FORMAT_NCHW || mMidFormat == MNN_DATA_FORMAT_NC4HW4);
        bool originTfFormat    = (tensorFormat == MNN_DATA_FORMAT_NHWC || tensorFormat == MNN_DATA_FORMAT_NHWC4);
        bool wrapTfFormat      = (mMidFormat == MNN_DATA_FORMAT_NHWC || mMidFormat == MNN_DATA_FORMAT_NHWC4);
        if ((originCaffeFormat && wrapCaffeFormat) || (originTfFormat && wrapTfFormat)) {
            TensorUtils::copyShape(tensor, wrapTensor);
        } else if (originCaffeFormat && wrapTfFormat) {
            for (int i = 1; i < wrapTensor->dimensions() - 1; ++i) {
                wrapTensor->setLength(i, tensor->length(i + 1));
            }
            wrapTensor->setLength(0, tensor->length(0));
            wrapTensor->setLength(wrapTensor->dimensions() - 1, tensor->length(1));
        } else if (originTfFormat && wrapCaffeFormat) {
            for (int i = 2; i < wrapTensor->dimensions(); ++i) {
                wrapTensor->setLength(i, tensor->length(i - 1));
            }
            wrapTensor->setLength(0, tensor->length(0));
            wrapTensor->setLength(1, tensor->length(tensor->dimensions() - 1));
        } else {
            // will not reach here
            MNN_ASSERT(false);
        }
        TensorUtils::setLinearLayout(wrapTensor);
    };

    convertTensorMeta(input, &mWrapTensorForInput);
    convertTensorMeta(output, &mWrapTensorForOutput);

    return NO_ERROR;
}

ErrorCode CPUReshape::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(1 == inputs.size() || 2 == inputs.size());
    MNN_ASSERT(1 == outputs.size());
    if (TensorUtils::getDescribe(inputs[0])->dimensionFormat != MNN_DATA_FORMAT_NC4HW4) {
        auto outputPtr = outputs[0]->host<uint8_t>();
        auto inputPtr = inputs[0]->host<uint8_t>();
        auto totalSize = inputs[0]->size();
        ::memcpy(outputPtr, inputPtr, totalSize);
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
