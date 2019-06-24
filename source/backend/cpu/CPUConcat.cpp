//
//  CPUConcat.cpp
//  MNN
//
//  Created by MNN on 2018/07/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUConcat.hpp"
#include "AutoStorage.h"
#include "CPUBackend.hpp"
#include "CommonOptFunction.h"
#include "Macro.h"

using namespace std;

namespace MNN {

static int _concatWidth(const Tensor* outputTensor, const vector<Tensor*>& inputTensors) {
    auto outputDim              = outputTensor->buffer().dim;
    const int depthQuad         = UP_DIV(outputDim[1].extent, 4);
    const int height            = outputDim[2].extent;
    const int width             = outputDim[3].extent;
    const int outputPlaneStride = 4 * height * width;
    const int outputLineStride  = 4 * width;

    int batchSize = outputDim[0].extent;

    for (int batchIndex = 0; batchIndex < batchSize; ++batchIndex) {
        int currentPositionW = 0;
        float* outputOrigin  = reinterpret_cast<float*>(outputTensor->buffer().host) + outputDim[0].stride * batchIndex;

        for (size_t b = 0; b < inputTensors.size(); b++) {
            auto& inputTensor    = inputTensors[b]->buffer();
            float* inputOrigin   = reinterpret_cast<float*>(inputTensor.host) + inputTensor.dim[0].stride * batchIndex;
            int inputPlaneStride = inputTensor.dim[3].extent * inputTensor.dim[2].extent * 4;
            int inputLineStride  = inputTensor.dim[3].extent * 4;
            int inputW           = inputTensor.dim[3].extent;
            for (int z = 0; z < depthQuad; ++z) {
                float* dstZ = outputOrigin + outputPlaneStride * z;
                float* srcZ = inputOrigin + inputPlaneStride * z;
                for (int y = 0; y < height; ++y) {
                    float* dstY = dstZ + outputLineStride * y + currentPositionW * 4;
                    float* srcY = srcZ + inputLineStride * y;
                    memcpy(dstY, srcY, 4 * inputW * sizeof(float));
                }
            }
            currentPositionW += inputW;
        }
    }
    return 0;
}

static int _concatHeight(const Tensor* outputTensor, const vector<Tensor*>& inputTensors) {
    auto outputDim              = outputTensor->buffer().dim;
    const int batchSize         = outputDim[0].extent;
    const int depthQuad         = UP_DIV(outputDim[1].extent, 4);
    const int height            = outputDim[2].extent;
    const int width             = outputDim[3].extent;
    const int outputPlaneStride = 4 * height * width;
    const int outputLineStride  = 4 * width;
    for (int batchIndex = 0; batchIndex < batchSize; ++batchIndex) {
        float* outputOrigin  = reinterpret_cast<float*>(outputTensor->buffer().host) + outputDim[0].stride * batchIndex;
        int currentPositionH = 0;
        for (size_t b = 0; b < inputTensors.size(); b++) {
            auto& inputTensor    = inputTensors[b]->buffer();
            float* inputOrigin   = reinterpret_cast<float*>(inputTensor.host) + inputTensor.dim[0].stride * batchIndex;
            int inputPlaneStride = inputTensor.dim[2].extent * inputTensor.dim[3].extent * 4;
            int inputH           = inputTensor.dim[2].extent;
            for (int z = 0; z < depthQuad; ++z) {
                float* dstZ = outputOrigin + outputPlaneStride * z;
                float* srcZ = inputOrigin + inputPlaneStride * z;

                memcpy(dstZ + currentPositionH * outputLineStride, srcZ, inputPlaneStride * sizeof(float));
            }
            currentPositionH += inputH;
        }
    }
    return 0;
}

static int _concatBatch(const Tensor* outputTensor, const vector<Tensor*>& inputTensors) {
    auto outputDim      = outputTensor->buffer().dim;
    const int batchSize = outputDim[0].extent;
    for (int batchIndex = 0; batchIndex < batchSize; ++batchIndex) {
        float* outputOrigin = reinterpret_cast<float*>(outputTensor->buffer().host) + outputDim[0].stride * batchIndex;
        for (size_t b = 0; b < inputTensors.size(); b++) {
            auto& inputTensor  = inputTensors[b]->buffer();
            float* inputOrigin = reinterpret_cast<float*>(inputTensor.host) + inputTensor.dim[0].stride * batchIndex;
            ::memcpy(outputOrigin, inputOrigin, inputTensor.dim[0].stride * sizeof(float));
        }
    }
    return 0;
}

static int _concatChannel(const Tensor* outputTensor, const vector<Tensor*>& inputTensors, bool useSlowMethod,
                          const Tensor* tempOutputTensor) {
    auto outputDim        = outputTensor->buffer().dim;
    const int height      = outputDim[2].extent;
    const int width       = outputDim[3].extent;
    int outputPlaneStride = 4 * height * width;
    float* outputOrigin   = reinterpret_cast<float*>(outputTensor->buffer().host);
    int batchSize         = outputDim[0].extent;

    if (useSlowMethod) {
        auto tempOutput = tempOutputTensor->host<float>();
        MNN_ASSERT(nullptr != tempOutput);
        for (int batchIndex = 0; batchIndex < batchSize; ++batchIndex) {
            float* currentOutput = tempOutput;
            for (int b = 0; b < inputTensors.size(); b++) {
                auto inputTensor = inputTensors[b];

                int size = inputTensor->width() * inputTensor->height() * inputTensor->channel();
                MNNUnpackC4(currentOutput, inputTensor->host<float>() + inputTensor->stride(0) * batchIndex,
                            inputTensor->width() * inputTensor->height(), inputTensor->channel());
                currentOutput += size;
            }
            MNNPackC4(outputTensor->host<float>() + batchIndex * outputTensor->stride(0), tempOutput,
                      outputTensor->width() * outputTensor->height(), outputTensor->channel());
        }
        return 0;
    }
    for (int batchIndex = 0; batchIndex < batchSize; ++batchIndex) {
        int currentPositionZ = 0;
        for (size_t b = 0; b < inputTensors.size(); b++) {
            auto& inputTensor  = inputTensors[b]->buffer();
            float* inputOrigin = reinterpret_cast<float*>(inputTensor.host) + inputTensor.dim[0].stride * batchIndex;
            int inputZ         = UP_DIV(inputTensor.dim[1].extent, 4);
            float* dst         = outputOrigin + outputPlaneStride * currentPositionZ + outputDim[0].stride * batchIndex;
            float* src         = inputOrigin;

            memcpy(dst, src, outputPlaneStride * inputZ * sizeof(float));
            currentPositionZ += inputZ;
        }
    }

    return 0;
}

static int _concatTf(const Tensor* outputTensor, const vector<Tensor*>& inputTensors, int axis) {
    auto& ob        = outputTensor->buffer();
    int outsideSize = 1;
    for (int i = 0; i < axis; ++i) {
        outsideSize *= ob.dim[i].extent;
    }
    int insideStride = ob.type.bytes();
    for (int i = axis + 1; i < ob.dimensions; ++i) {
        insideStride *= ob.dim[i].extent;
    }
    int outsideStride = insideStride * ob.dim[axis].extent;

    int sumAxis           = 0;
    uint8_t* outputOrigin = reinterpret_cast<uint8_t*>(outputTensor->buffer().host);
    for (size_t b = 0; b < inputTensors.size(); b++) {
        auto& inputTensor = inputTensors[b]->buffer();
        if (0 == inputTensor.dimensions) {
            continue;
        }
        uint8_t* inputOrigin = reinterpret_cast<uint8_t*>(inputTensor.host);
        int inputPlaneStride = inputTensor.dim[axis].extent * insideStride;

        for (int z = 0; z < outsideSize; ++z) {
            uint8_t* dstZ = outputOrigin + outsideStride * z + sumAxis * insideStride;
            uint8_t* srcZ = inputOrigin + inputPlaneStride * z;

            memcpy(dstZ, srcZ, inputPlaneStride);
        }
        sumAxis += inputTensor.dim[axis].extent;
    }
    return 0;
}

ErrorCode CPUConcat::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    MNN_ASSERT(outputs.size() == 1);
    MNN_ASSERT(inputs.size() >= 2);
    auto output    = outputs[0];
    mUseSlowMethod = false;
    mTempOutput.reset();
    if (output->buffer().dimensions > 1 && output->buffer().dim[1].flags == Tensor::REORDER_4) {
        if (1 == mAxis) {
            // The last tensor needn't be aligned
            for (size_t b = 0; b < inputs.size() - 1; b++) {
                if (inputs[b]->length(1) % 4 != 0) {
                    mUseSlowMethod = true;
                    break;
                }
            }
            if (mUseSlowMethod) {
                mTempOutput.reset(Tensor::createDevice<float>(output->shape()));
                mTempOutput->setLength(0, 1);
                bool success = backend()->onAcquireBuffer(mTempOutput.get(), Backend::DYNAMIC);
                if (false == success) {
                    return OUT_OF_MEMORY;
                }
                backend()->onReleaseBuffer(mTempOutput.get(), Backend::DYNAMIC);
            }
        }
    }

    return NO_ERROR;
}

ErrorCode CPUConcat::onExecute(const vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    MNN_ASSERT(1 == outputs.size());
    MNN_ASSERT(inputs.size() >= 2);
    auto input = inputs[0];
    if (input->buffer().dimensions > 1 && input->buffer().dim[1].flags == Tensor::REORDER_4) {
        switch (mAxis) {
            case 0:
                _concatBatch(outputs[0], inputs);
                break;
            case 1:
                _concatChannel(outputs[0], inputs, mUseSlowMethod, mTempOutput.get());
                break;
            case 2:
                _concatHeight(outputs[0], inputs);
                break;
            case 3:
                _concatWidth(outputs[0], inputs);
                break;

            default:
                break;
        }
    } else {
        int axis = mAxis;
        // tf concat
        _concatTf(outputs[0], inputs, axis);
    }

    return NO_ERROR;
}

class CPUConcatCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const {
        auto axis = op->main_as_Axis();
        if (nullptr != axis) {
            if (axis->axis() < 0) {
                return new CPUConcat(backend, outputs[0]->dimensions() + axis->axis());
            }
            return new CPUConcat(backend, axis->axis());
        }
        return new CPUConcat(backend, 0);
    }
};

REGISTER_CPU_OP_CREATOR(CPUConcatCreator, OpType_Concat);
} // namespace MNN
