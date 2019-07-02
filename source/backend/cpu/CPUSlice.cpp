//
//  CPUSlice.cpp
//  MNN
//
//  Created by MNN on 2018/07/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUSlice.hpp"
#include "CPUBackend.hpp"
#include "CommonOptFunction.h"
#include "Macro.h"

using namespace std;

namespace MNN {

static void _sliceInAxis(const Tensor* inputTensor, const vector<Tensor*>& outputTensors, int axis) {
    int outsideSize = 1;
    for (int i = 0; i < axis; ++i) {
        if (i == 1) {
            outsideSize *= UP_DIV(inputTensor->length(i), 4);
        } else {
            outsideSize *= inputTensor->length(i);
        }
    }

    int inputStride = inputTensor->getType().bytes();
    int axisStride  = inputTensor->stride(axis) * inputTensor->getType().bytes();
    if (axis > 0) {
        inputStride *= inputTensor->stride(axis - 1) * 4;
        axisStride *= 4;
    }

    int currentPos = 0;
    for (int b = 0; b < outputTensors.size(); ++b) {
        auto srcCurrent     = inputTensor->host<char>() + currentPos * axisStride;
        int length          = outputTensors[b]->length(axis);
        auto dstCurrent     = outputTensors[b]->host<char>();
        int dstOutputStride = outputTensors[b]->getType().bytes();
        if (axis > 0) {
            dstOutputStride *= outputTensors[b]->stride(axis - 1) * 4;
        }
        for (int o = 0; o < outsideSize; ++o) {
            auto src = srcCurrent + o * inputStride;
            auto dst = dstCurrent + o * dstOutputStride;
            ::memcpy(dst, src, length * axisStride);
        }
        currentPos += length;
    }
}

static void _sliceInAxisTf(const Tensor* inputTensor, const vector<Tensor*>& outputTensors, int axis) {
    int outsideSize = 1;
    for (int i = 0; i < axis; ++i) {
        outsideSize *= inputTensor->length(i);
    }

    int inputStride = inputTensor->getType().bytes();
    int axisStride  = inputTensor->stride(axis) * inputTensor->getType().bytes();
    if (axis > 0) {
        inputStride *= inputTensor->stride(axis - 1);
    }

    int currentPos = 0;
    for (int b = 0; b < outputTensors.size(); ++b) {
        auto srcCurrent     = inputTensor->host<char>() + currentPos * axisStride;
        int length          = outputTensors[b]->length(axis);
        auto dstCurrent     = outputTensors[b]->host<char>();
        int dstOutputStride = outputTensors[b]->getType().bytes();
        if (axis > 0) {
            dstOutputStride *= outputTensors[b]->stride(axis - 1);
        }
        for (int o = 0; o < outsideSize; ++o) {
            auto src = srcCurrent + o * inputStride;
            auto dst = dstCurrent + o * dstOutputStride;
            ::memcpy(dst, src, length * axisStride);
        }
        currentPos += length;
    }
}

static int _sliceChannel(const Tensor* inputTensor, const vector<Tensor*>& outputTensors,
                         const Tensor* tempInputTensor) {
    MNN_ASSERT(inputTensor->getType().bytes() == sizeof(float));
    auto inputDim        = inputTensor->buffer().dim;
    int height           = std::max(inputDim[2].extent, 1);
    int width            = std::max(inputDim[3].extent, 1);
    int inputPlaneStride = 4 * height * width;
    float* inputOrigin   = (float*)inputTensor->buffer().host;
    for (int batchIndex = 0; batchIndex < inputTensor->batch(); ++batchIndex) {
        if (nullptr != tempInputTensor) {
            float* tempinput = tempInputTensor->host<float>();
            MNN_ASSERT(nullptr != tempinput);
            MNNUnpackC4(tempinput, inputTensor->host<float>() + batchIndex * inputTensor->stride(0), width * height,
                        inputTensor->channel());
            float* currentinput = tempinput;
            for (int b = 0; b < outputTensors.size(); b++) {
                auto outputTensor = outputTensors[b];
                int size          = outputTensor->width() * outputTensor->height() * outputTensor->channel();
                MNNPackC4(outputTensor->host<float>() + batchIndex * outputTensor->stride(0), currentinput,
                          width * height, outputTensor->channel());
                currentinput += size;
            }
            return 0;
        }
        int currentPositionZ = 0;
        for (size_t b = 0; b < outputTensors.size(); b++) {
            auto& outputTensor  = outputTensors[b]->buffer();
            float* outputOrigin = (float*)outputTensor.host + batchIndex * outputTensor.dim[0].stride;
            int outputZ         = UP_DIV(outputTensor.dim[1].extent, 4);
            float* dst = inputOrigin + inputPlaneStride * currentPositionZ + batchIndex * inputTensor->stride(0);
            float* src = outputOrigin;

            memcpy(src, dst, inputPlaneStride * outputZ * sizeof(float));
            currentPositionZ += outputZ;
        }
    }

    return 0;
}

CPUSlice::CPUSlice(Backend* b, int axis) : MNN::Execution(b) {
    mAxis      = axis;
}

ErrorCode CPUSlice::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    MNN_ASSERT(1 == inputs.size());
    MNN_ASSERT(outputs.size() >= 2);
    auto input              = inputs[0];
    const auto tensorFormat = input->getDimensionType();
    mTempInput.reset();
    if (Tensor::CAFFE == tensorFormat) {
        // TODO Support other flag
        MNN_ASSERT(inputs[0]->buffer().dim[1].flags == MNN::Tensor::REORDER_4);
        if (mAxis == 1) {
            bool useSlowMethod = false;
            // Last one need not be 4 aligned
            for (size_t b = 0; b < outputs.size() - 1; b++) {
                auto& outputTensor = outputs[b]->buffer();
                if (outputTensor.dim[1].extent % 4 != 0) {
                    useSlowMethod = true;
                }
            }
            if (useSlowMethod) {
                mTempInput.reset(Tensor::createDevice<float>(input->shape()));
                mTempInput->setLength(0, 1);
                bool success = backend()->onAcquireBuffer(mTempInput.get(), Backend::DYNAMIC);
                if (!success) {
                    return OUT_OF_MEMORY;
                }
                backend()->onReleaseBuffer(mTempInput.get(), Backend::DYNAMIC);
            }
        }
    }
    return NO_ERROR;
}

ErrorCode CPUSlice::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input = inputs[0];
    const auto tensorFormat = input->getDimensionType();
    if (Tensor::CAFFE == tensorFormat) {
        if (mAxis == 1) {
            _sliceChannel(inputs[0], outputs, mTempInput.get());
            return NO_ERROR;
        }
        _sliceInAxis(inputs[0], outputs, mAxis);
    } else {
        _sliceInAxisTf(inputs[0], outputs, mAxis);
    }

    return NO_ERROR;
}

class CPUSliceCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const {
        auto slice = op->main_as_Slice();
        if (nullptr == slice || inputs.empty()) {
            return nullptr;
        }
        auto axis = slice->axis();
        if (axis < 0) {
            axis = axis + inputs[0]->dimensions();
        }
        return new CPUSlice(backend, axis);
    }
};

REGISTER_CPU_OP_CREATOR(CPUSliceCreator, OpType_Slice);
} // namespace MNN
