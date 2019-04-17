//
//  CPUCrop.cpp
//  MNN
//
//  Created by MNN on 2018/10/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUCrop.hpp"
#include "CPUBackend.hpp"
#include "Macro.h"

namespace MNN {

CPUCrop::CPUCrop(Backend* b, const MNN::Op* op) : Execution(b) {
    auto cropParam = op->main_as_Crop();
    mAxis          = cropParam->axis();
    int offsetSize = cropParam->offset()->size();

    mOffsets.resize(offsetSize);
    for (int i = 0; i < offsetSize; ++i) {
        mOffsets[i] = cropParam->offset()->data()[i];
    }
}

void CPUCrop::cropCopy(const Tensor* inputTensor, Tensor* outputTensor, const std::vector<int>& offsets) {
    const int outImgSize = outputTensor->buffer().dim[0].stride;
    const int outHW      = outputTensor->buffer().dim[1].stride * 4;
    const int inImgSize  = inputTensor->buffer().dim[0].stride;
    const int inHW       = inputTensor->buffer().dim[1].stride * 4;

    const float* inData = inputTensor->host<float>();
    float* outData      = outputTensor->host<float>();

    const int outChannels = UP_DIV(outputTensor->channel(), 4);
    const int outWidth    = outputTensor->width() * 4;
    const int inWidth     = inputTensor->width() * 4;

    for (int b = 0; b < outputTensor->batch(); ++b) {
        for (int c = 0; c < outChannels; ++c) {
            for (int h = 0; h < outputTensor->height(); ++h) {
                float* outPtr      = outData + b * outImgSize + c * outHW + h * outWidth;
                const float* inPtr = inData + (b + offsets[0]) * inImgSize + (c + offsets[1]) * inHW +
                                     (h + offsets[2]) * inWidth + offsets[3] * 4;
                ::memcpy(outPtr, inPtr, sizeof(float) * outWidth);
            }
        }
    }
}

ErrorCode CPUCrop::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    const auto input0  = inputs[0];
    const auto input1  = inputs[1];
    const int inputDim = input0->buffer().dimensions;
    std::vector<int> offsets(inputDim, 0);
    MNN_ASSERT(2 <= mAxis);
    for (int i = 0; i < inputDim; ++i) {
        int cropOffset = 0;
        if (i >= mAxis) {
            if (mOffsets.size() == 1) {
                cropOffset = mOffsets[0];
            } else if (mOffsets.size() > 1) {
                cropOffset = mOffsets[i - mAxis];
            }
            MNN_ASSERT(input0->buffer().dim[i].extent - cropOffset >= input1->buffer().dim[i].extent);
        }
        offsets[i] = cropOffset;
    }
    CPUCrop::cropCopy(input0, outputs[0], offsets);

    return NO_ERROR;
}

class CPUCropCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new CPUCrop(backend, op);
    }
};

REGISTER_CPU_OP_CREATOR(CPUCropCreator, OpType_Crop);

} // namespace MNN
