//
//  CPUQuantizedConcat.cpp
//  MNN
//
//  Created by MNN on 2018/12/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUQuantizedConcat.hpp"
#include "CPUBackend.hpp"
#include "CPUFixedPoint.hpp"
#include "CPUQuantizationUtils.hpp"
#include "Macro.h"
#include "OptimizedComputer.hpp"

namespace MNN {

CPUQuantizedConcat::CPUQuantizedConcat(Backend *backend, const Op *op) : Execution(backend) {
    auto quantizedConcatParam = op->main_as_QuantizedConcat();
    mAxis                     = quantizedConcatParam->axis();
    for (int i = 0; i < quantizedConcatParam->inputZeroPoint()->size(); i++) {
        mInputZeroPoint.push_back(quantizedConcatParam->inputZeroPoint()->data()[i]);
        mInputScale.push_back(quantizedConcatParam->inputScale()->data()[i]);
    }
    mOutputZeroPoint = quantizedConcatParam->outputQuantizedParam()->zeroPoint();
    mOutputScale     = quantizedConcatParam->outputQuantizedParam()->scale();
}

ErrorCode CPUQuantizedConcat::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    if (mAxis < 0) {
        mAxis += outputs[0]->buffer().dimensions;
    }
    return NO_ERROR;
}

ErrorCode CPUQuantizedConcat::onExecute(const std::vector<MNN::Tensor *> &inputs,
                                        const std::vector<MNN::Tensor *> &outputs) {
    int inputsCount = (int)inputs.size();
    MNN_ASSERT(inputsCount > 1);
    int concatSize = 0;
    int concatDim  = mAxis;

    for (int i = 0; i < inputsCount; i++) {
        for (int j = 0; j < 4; j++) {
            if (j != concatDim) {
                MNN_ASSERT(inputs[i]->buffer().dim[j].extent == outputs[0]->buffer().dim[j].extent);
            }
        }
        concatSize += inputs[i]->buffer().dim[concatDim].extent;
    }
    MNN_ASSERT(concatSize == outputs[0]->buffer().dim[concatDim].extent);

    int outerSize = 1;
    for (int i = concatDim - 1; i >= 0; i--) {
        outerSize *= outputs[0]->buffer().dim[i].extent;
    }

    const float inverseOutputScale = 1.f / mOutputScale;
    uint8_t *outputPtr             = outputs[0]->host<uint8_t>();

    for (int k = 0; k < outerSize; k++) {
        for (int i = 0; i < inputsCount; ++i) {
            const int copySize      = inputs[i]->buffer().dim[concatDim].extent * inputs[i]->stride(concatDim);
            const uint8_t *inputPtr = inputs[i]->host<uint8_t>() + k * copySize;
            if (mInputZeroPoint[i] == mOutputZeroPoint && mInputScale[i] == mOutputScale) {
                memcpy(outputPtr, inputPtr, copySize);
            } else {
                const float scale = mInputScale[i] * inverseOutputScale;
                const float bias  = -mInputZeroPoint[i] * scale;
                for (int j = 0; j < copySize; ++j) {
                    const int32_t value = static_cast<int32_t>(round(inputPtr[j] * scale + bias)) + mOutputZeroPoint;
                    outputPtr[j]        = static_cast<uint8_t>(std::max(std::min(255, value), 0));
                }
            }
            outputPtr += copySize;
        }
    }

    return NO_ERROR;
}

class CPUQuantizedConcatCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const {
        return new CPUQuantizedConcat(backend, op);
    }
};
REGISTER_CPU_OP_CREATOR(CPUQuantizedConcatCreator, OpType_QuantizedConcat);
} // namespace MNN
