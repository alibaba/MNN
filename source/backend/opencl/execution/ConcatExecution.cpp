//
//  ConcatExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "execution/ConcatExecution.hpp"

namespace MNN {
namespace OpenCL {
ErrorCode ConcatImageExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto runtime = ((OpenCLBackend *)backend())->getOpenCLRuntime();
    mUnits.resize(inputs.size());
    auto output        = outputs[0];
    int outputWH[]     = {output->width(), output->height()};
    int outputOffset[] = {0, 0, 0, 0};
    int inputOffset[]  = {0, 0, 0, 0};
    for (int i = 0; i < mUnits.size(); ++i) {
        auto input    = inputs[i];
        auto &unit    = mUnits[i];
        int inputWH[] = {input->width(), input->height()};
        int region[]  = {input->batch(), UP_DIV(input->channel(), 4), input->height(), input->width()};
        int wh[]      = {input->width(), input->height()};

        unit.kernel = runtime->buildKernel("blit", "blit", std::set<std::string>{});
        unit.kernel.setArg(0, openCLImage(input));
        unit.kernel.setArg(1, openCLImage(output));
        unit.kernel.setArg(2, inputOffset);
        unit.kernel.setArg(3, outputOffset);
        unit.kernel.setArg(4, region);
        unit.kernel.setArg(5, inputWH);
        unit.kernel.setArg(6, outputWH);
        unit.kernel.setArg(7, wh);
        unit.localWorkSize  = {16, 16};
        unit.globalWorkSize = {(uint32_t)UP_DIV(region[1] * region[3], 16) * 16,
                               (uint32_t)UP_DIV(region[0] * region[2], 16) * 16};

        outputOffset[mAxis] += region[mAxis];
    }
    return NO_ERROR;
}

ErrorCode ConcatBufferExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto runtime = ((OpenCLBackend *)backend())->getOpenCLRuntime();
    mTempInputs.resize(inputs.size());

    // Alloc Temp buffer
    auto bufferPool     = ((OpenCLBackend *)backend())->getBufferPool();
    auto bufferUnitSize = runtime->isSupportedFP16() ? sizeof(int16_t) : sizeof(float);
    mTempOutput         = bufferPool->alloc(outputs[0]->elementSize() * bufferUnitSize);
    for (int i = 0; i < inputs.size(); ++i) {
        auto bufferSize = inputs[i]->elementSize() * bufferUnitSize;
        auto buffer     = bufferPool->alloc(bufferSize);
        mTempInputs[i]  = buffer;
        bufferPool->recycle(buffer);
    }
    bufferPool->recycle(mTempOutput);
    int inputSize = inputs.size();

    // Create Unit
    mUnits.resize(2 * inputSize + 1);
    auto output        = outputs[0];
    auto outputShape   = tensorShapeFormat(output);
    int outputOffset[] = {0, 0, 0, 0};
    int inputOffset[]  = {0, 0, 0, 0};

    // Use nhwc stride
    int outputStride[] = {outputShape[2] * outputShape[1] * outputShape[3], 1, outputShape[2] * outputShape[3],
                          outputShape[3]};
    for (int i = 0; i < inputSize; ++i) {
        auto tempBuffer    = mTempInputs[i];
        auto input         = inputs[i];
        auto inputShape    = tensorShapeFormat(input);
        int inputWH[]      = {inputShape[2], inputShape[1]};
        int region[]       = {inputShape[0], UP_DIV(inputShape[3], 4), inputShape[1], inputShape[2]};
        int regionBuffer[] = {inputShape[0], inputShape[3], inputShape[1], inputShape[2]};
        int inputStride[]  = {inputShape[2] * inputShape[1] * inputShape[3], 1, inputShape[2] * inputShape[3],
                             inputShape[3]};

        // Image to buffer
        {
            Unit &unit          = mUnits[2 * i + 0];
            unit.kernel         = runtime->buildKernel("blitBuffer", "blitImageToBuffer", {});
            unit.localWorkSize  = {16, 16};
            unit.globalWorkSize = {(uint32_t)UP_DIV(region[3] * region[1], 16) * 16,
                                   (uint32_t)UP_DIV(region[2] * region[0], 16) * 16};
            unit.kernel.setArg(0, openCLImage(input));
            unit.kernel.setArg(1, *tempBuffer);
            unit.kernel.setArg(2, inputOffset);
            unit.kernel.setArg(3, inputOffset);
            unit.kernel.setArg(4, region);
            unit.kernel.setArg(5, inputWH);
            unit.kernel.setArg(6, inputStride);
            unit.kernel.setArg(7, 4 * sizeof(int), inputShape.data());
        }

        // Blit buffer to buffer
        {
            Unit &unit          = mUnits[2 * i + 1];
            unit.kernel         = runtime->buildKernel("blitBuffer", "blitBuffer", {});
            unit.localWorkSize  = {16, 16};
            unit.globalWorkSize = {(uint32_t)UP_DIV(regionBuffer[3] * regionBuffer[1], 16) * 16,
                                   (uint32_t)UP_DIV(regionBuffer[2] * regionBuffer[0], 16) * 16};
            unit.kernel.setArg(0, *tempBuffer);
            unit.kernel.setArg(1, *mTempOutput);
            unit.kernel.setArg(2, inputOffset);
            unit.kernel.setArg(3, outputOffset);
            unit.kernel.setArg(4, regionBuffer);
            unit.kernel.setArg(5, inputStride);
            unit.kernel.setArg(6, outputStride);
            unit.kernel.setArg(7, inputWH);
        }
        outputOffset[mAxis] += regionBuffer[mAxis];
    }
    {
        int wh[]     = {outputShape[2], outputShape[1]};
        int region[] = {outputShape[0], UP_DIV(outputShape[3], 4), outputShape[1], outputShape[2]};

        Unit &unit          = mUnits[2 * inputSize];
        unit.kernel         = runtime->buildKernel("blitBuffer", "blitBufferToImage", {});
        unit.localWorkSize  = {16, 16};
        unit.globalWorkSize = {(uint32_t)UP_DIV(region[3] * region[1], 16) * 16,
                               (uint32_t)UP_DIV(region[2] * region[0], 16) * 16};
        unit.kernel.setArg(0, *mTempOutput);
        unit.kernel.setArg(1, openCLImage(output));
        unit.kernel.setArg(2, inputOffset);
        unit.kernel.setArg(3, inputOffset);
        unit.kernel.setArg(4, region);
        unit.kernel.setArg(5, outputStride);
        unit.kernel.setArg(6, wh);
        unit.kernel.setArg(7, wh);
    }

    return NO_ERROR;
}

class ConcatCreator : public OpenCLBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        auto axis = op->main_as_Axis()->axis();
        if (-1 == axis) {
            axis = inputs[0]->dimensions() - 1;
        }
        if (outputs[0]->getDimensionType() == Tensor::TENSORFLOW) {
            if(outputs[0]->dimensions() == 3){
                int index[] = {2, 3, 1};
                return new ConcatBufferExecution(inputs, index[axis], backend);
            }
            if(outputs[0]->dimensions() == 4){
                int index[] = {0, 2, 3, 1};
                return new ConcatBufferExecution(inputs, index[axis], backend);
            }
            return nullptr;
        }
        
        if (1 == axis) {
            for (int i = 0; i < inputs.size() - 1; ++i) {
                if (inputs[i]->channel() % 4 != 0) {
                    return new ConcatBufferExecution(inputs, axis, backend);
                }
            }
        }
        return new ConcatImageExecution(inputs, axis, backend);
    }
};

OpenCLCreatorRegister<ConcatCreator> __concat_op(OpType_Concat);

} // namespace OpenCL
} // namespace MNN
