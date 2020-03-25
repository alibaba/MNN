//
//  Arm82Concat.cpp
//  MNN
//
//  Created by MNN on 2018/07/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/arm82/Arm82Concat.hpp"
#include "backend/arm82/Arm82Backend.hpp"
#include "backend/arm82/Arm82OptFunc.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
using namespace std;

namespace MNN {

static int _concatWidth(const Tensor* outputTensor, const vector<Tensor*>& inputTensors) {
    auto outputDim              = outputTensor->buffer().dim;
    const int depthQuad         = UP_DIV(outputDim[1].extent, ARMV82_CHANNEL_UNIT);
    const int height            = outputDim[2].extent;
    const int width             = outputDim[3].extent;
    const int outputPlaneStride = ARMV82_CHANNEL_UNIT * height * width;
    const int outputLineStride  = ARMV82_CHANNEL_UNIT * width;
    const int outputBatchStride = depthQuad * outputPlaneStride;
    int batchSize               = outputDim[0].extent;

    for (int batchIndex = 0; batchIndex < batchSize; ++batchIndex) {
        int currentPositionW = 0;
        FLOAT16* outputOrigin =
            reinterpret_cast<FLOAT16*>(outputTensor->buffer().host) + outputBatchStride * batchIndex;

        for (size_t b = 0; b < inputTensors.size(); b++) {
            auto& inputTensor    = inputTensors[b]->buffer();
            const int stride     = ARM82TensorBatchStrideHelper(inputTensors[b]);
            FLOAT16* inputOrigin = reinterpret_cast<FLOAT16*>(inputTensor.host) + stride * batchIndex;
            int inputPlaneStride = inputTensor.dim[3].extent * inputTensor.dim[2].extent * ARMV82_CHANNEL_UNIT;
            int inputLineStride  = inputTensor.dim[3].extent * ARMV82_CHANNEL_UNIT;
            int inputW           = inputTensor.dim[3].extent;
            for (int z = 0; z < depthQuad; ++z) {
                FLOAT16* dstZ = outputOrigin + outputPlaneStride * z;
                FLOAT16* srcZ = inputOrigin + inputPlaneStride * z;
                for (int y = 0; y < height; ++y) {
                    FLOAT16* dstY = dstZ + outputLineStride * y + currentPositionW * ARMV82_CHANNEL_UNIT;
                    FLOAT16* srcY = srcZ + inputLineStride * y;
                    memcpy(dstY, srcY, ARMV82_CHANNEL_UNIT * inputW * sizeof(FLOAT16));
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
    const int depthQuad         = UP_DIV(outputDim[1].extent, ARMV82_CHANNEL_UNIT);
    const int height            = outputDim[2].extent;
    const int width             = outputDim[3].extent;
    const int outputPlaneStride = ARMV82_CHANNEL_UNIT * height * width;
    const int outputLineStride  = ARMV82_CHANNEL_UNIT * width;
    const int outputBatchStride = depthQuad * outputPlaneStride;
    for (int batchIndex = 0; batchIndex < batchSize; ++batchIndex) {
        FLOAT16* outputOrigin =
            reinterpret_cast<FLOAT16*>(outputTensor->buffer().host) + outputBatchStride * batchIndex;
        int currentPositionH = 0;
        for (size_t b = 0; b < inputTensors.size(); b++) {
            auto& inputTensor    = inputTensors[b]->buffer();
            const int stride     = ARM82TensorBatchStrideHelper(inputTensors[b]);
            FLOAT16* inputOrigin = reinterpret_cast<FLOAT16*>(inputTensor.host) + stride * batchIndex;
            int inputPlaneStride = inputTensor.dim[2].extent * inputTensor.dim[3].extent * ARMV82_CHANNEL_UNIT;
            int inputH           = inputTensor.dim[2].extent;
            for (int z = 0; z < depthQuad; ++z) {
                FLOAT16* dstZ = outputOrigin + outputPlaneStride * z;
                FLOAT16* srcZ = inputOrigin + inputPlaneStride * z;

                memcpy(dstZ + currentPositionH * outputLineStride, srcZ, inputPlaneStride * sizeof(FLOAT16));
            }
            currentPositionH += inputH;
        }
    }
    return 0;
}

static int _concatBatch(const Tensor* outputTensor, const vector<Tensor*>& inputTensors) {
    auto outputDim              = outputTensor->buffer().dim;
    const int batchSize         = outputDim[0].extent;
    const int outputBatchStride = ARM82TensorBatchStrideHelper(outputTensor);
    for (int batchIndex = 0; batchIndex < batchSize; ++batchIndex) {
        FLOAT16* outputOrigin =
            reinterpret_cast<FLOAT16*>(outputTensor->buffer().host) + outputBatchStride * batchIndex;
        for (size_t b = 0; b < inputTensors.size(); b++) {
            auto& inputTensor    = inputTensors[b]->buffer();
            const int stride     = ARM82TensorBatchStrideHelper(inputTensors[b]);
            FLOAT16* inputOrigin = reinterpret_cast<FLOAT16*>(inputTensor.host) + stride * batchIndex;
            ::memcpy(outputOrigin, inputOrigin, stride * sizeof(FLOAT16));
        }
    }
    return 0;
}

static int _concatChannel(const Tensor* outputTensor, const vector<Tensor*>& inputTensors, bool useSlowMethod,
                          const Tensor* tempOutputTensor) {
    auto outputDim      = outputTensor->buffer().dim;
    int batchSize       = outputDim[0].extent;
    const int outStride = ARM82TensorBatchStrideHelper(outputTensor);
    if (useSlowMethod) {
        auto tempOutput = tempOutputTensor->host<uint16_t>();
        MNN_ASSERT(nullptr != tempOutput);
        for (int batchIndex = 0; batchIndex < batchSize; ++batchIndex) {
            uint16_t* currentOutput = tempOutput;
            for (int b = 0; b < inputTensors.size(); b++) {
                auto inputTensor = inputTensors[b];
                const int stride = ARM82TensorBatchStrideHelper(inputTensor);

                int size = inputTensor->width() * inputTensor->height() * inputTensor->channel();
                MNNNC8HW8TONCHW_NO_TYPE(currentOutput, inputTensor->host<uint16_t>() + stride * batchIndex,
                                        inputTensor->width() * inputTensor->height(), inputTensor->channel());
                currentOutput += size;
            }

            MNNNCHWTONC8HW8_NO_TYPE(outputTensor->host<uint16_t>() + batchIndex * outStride, tempOutput,
                                    outputTensor->width() * outputTensor->height(), outputTensor->channel());
        }
        return 0;
    }

    FLOAT16* outputOrigin = reinterpret_cast<FLOAT16*>(outputTensor->buffer().host);
    for (int batchIndex = 0; batchIndex < batchSize; ++batchIndex) {
        int currentPositionZ = 0;
        for (size_t b = 0; b < inputTensors.size(); b++) {
            auto& inputTensor    = inputTensors[b]->buffer();
            const int stride     = ARM82TensorBatchStrideHelper(inputTensors[b]);
            FLOAT16* inputOrigin = reinterpret_cast<FLOAT16*>(inputTensor.host) + stride * batchIndex;
            int inputZ           = UP_DIV(inputTensor.dim[1].extent, ARMV82_CHANNEL_UNIT);
            FLOAT16* dst =
                outputOrigin + outputDim[1].stride * currentPositionZ * ARMV82_CHANNEL_UNIT + outStride * batchIndex;
            FLOAT16* src = inputOrigin;

            memcpy(dst, src, outputDim[1].stride * ARMV82_CHANNEL_UNIT * inputZ * sizeof(FLOAT16));
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

ErrorCode Arm82Concat::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    MNN_ASSERT(outputs.size() == 1);
    MNN_ASSERT(inputs.size() >= 2);
    auto output    = outputs[0];
    mUseSlowMethod = false;
    mTempOutput.reset();
    if (output->buffer().dimensions > 1 &&
        TensorUtils::getDescribe(output)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
        if (1 == mAxis) {
            // The last tensor needn't be aligned
            for (size_t b = 0; b < inputs.size() - 1; b++) {
                if (inputs[b]->length(1) % ARMV82_CHANNEL_UNIT != 0) {
                    mUseSlowMethod = true;
                    break;
                }
            }
            if (mUseSlowMethod) {
                mTempOutput.reset(Tensor::createDevice<uint16_t>(output->shape()));
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

ErrorCode Arm82Concat::onExecute(const vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    MNN_ASSERT(1 == outputs.size());
    MNN_ASSERT(inputs.size() >= 2);
    auto input = inputs[0];
    if (input->buffer().dimensions > 1 && TensorUtils::getDescribe(input)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
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

class Arm82ConcatCreator : public Arm82Backend::Arm82Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const {
        auto axis = op->main_as_Axis();
        if (nullptr != axis) {
            if (axis->axis() < 0) {
                return new Arm82Concat(backend, outputs[0]->dimensions() + axis->axis());
            }
            return new Arm82Concat(backend, axis->axis());
        }
        return new Arm82Concat(backend, 0);
    }
};

REGISTER_ARM82_OP_CREATOR(OpType_Concat, Arm82ConcatCreator);
} // namespace MNN
