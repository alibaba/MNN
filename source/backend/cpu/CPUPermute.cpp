//
//  CPUPermute.cpp
//  MNN
//
//  Created by MNN on 2018/07/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUPermute.hpp"
#include "CPUBackend.hpp"
#include "CommonOptFunction.h"
#include "Macro.h"
#include "TensorUtils.hpp"

namespace MNN {

CPUPermute::CPUPermute(Backend *b, const MNN::Op *op) : MNN::Execution(b) {
    auto shape = op->main_as_Permute()->dims();
    for (int i = 0; i < shape->size(); ++i) {
        mDims.push_back(shape->data()[i]);
    }
}

ErrorCode CPUPermute::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    TensorUtils::copyShape(inputs[0], &mStorage);
    mStorage.buffer().dim[1].flags  = 0;
    mStorage.buffer().dim[0].extent = 1;
    TensorUtils::setLinearLayout(&mStorage);

    backend()->onAcquireBuffer(&mStorage, Backend::DYNAMIC);
    backend()->onReleaseBuffer(&mStorage, Backend::DYNAMIC);

    return NO_ERROR;
}

ErrorCode CPUPermute::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(1 == inputs.size());
    MNN_ASSERT(1 == outputs.size());

    auto &input  = inputs[0]->buffer();
    auto &output = outputs[0]->buffer();

    // Currently don't support batch reshape, but support multi batch
    MNN_ASSERT(output.dim[0].extent == input.dim[0].extent);
    MNN_ASSERT(output.dimensions == input.dimensions);

    MNN_ASSERT(output.dimensions == 4);

    int areaInput  = 1;
    int areaOutput = 1;
    for (int i = 2; i < input.dimensions; ++i) {
        areaInput *= input.dim[i].extent;
        areaOutput *= output.dim[i].extent;
    }
    int inputBatchSize  = ALIGN_UP4(input.dim[1].extent) * areaInput;
    int outputBatchSize = ALIGN_UP4(output.dim[1].extent) * areaOutput;

    auto originInput  = (const float *)input.host;
    auto originOutput = (float *)output.host;
    auto storgeData   = mStorage.host<float>();
    for (int b = 0; b < input.dim[0].extent; ++b) {
        auto inputCurrent  = originInput + inputBatchSize * b;
        auto outputCurrent = originOutput + outputBatchSize * b;
        if (1 == areaInput) {
            ::memcpy(outputCurrent, inputCurrent, input.dim[1].extent * sizeof(float));
        } else {
            MNNUnpackC4(outputCurrent, inputCurrent, areaInput, input.dim[1].extent);
        }

        int dimIndexes[4];
        const int width         = input.dim[3].extent;
        const int height        = input.dim[2].extent;
        const int inputRealArea = width * height;

        const int outputWidth    = output.dim[3].extent;
        const int outputHeight   = output.dim[2].extent;
        const int outputChannel  = output.dim[1].extent;
        const int outputRealArea = outputWidth * outputHeight;

        for (int iz = 0; iz < outputChannel; ++iz) {
            dimIndexes[mDims[1]] = iz;
            for (int iy = 0; iy < outputHeight; ++iy) {
                dimIndexes[mDims[2]] = iy;
                for (int ix = 0; ix < outputWidth; ++ix) {
                    dimIndexes[mDims[3]] = ix;
                    int inputIndex       = dimIndexes[1] * inputRealArea + dimIndexes[2] * width + dimIndexes[3];
                    int outputIndex      = iz * outputRealArea + iy * outputWidth + ix;

                    storgeData[outputIndex] = outputCurrent[inputIndex];
                }
            }
        }

        if (1 == areaOutput) {
            ::memcpy(outputCurrent, storgeData, output.dim[1].extent * sizeof(float));
        } else {
            MNNPackC4(outputCurrent, storgeData, areaOutput, output.dim[1].extent);
        }
    }

    return NO_ERROR;
}

class CPUPermuteCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        return new CPUPermute(backend, op);
    }
};

REGISTER_CPU_OP_CREATOR(CPUPermuteCreator, OpType_Permute);
} // namespace MNN
