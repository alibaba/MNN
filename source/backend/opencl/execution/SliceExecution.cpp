//
//  SliceExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "SliceExecution.hpp"
#include <Macro.h>
#include "TensorUtils.hpp"

namespace MNN {
namespace OpenCL {

ErrorCode SliceExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto inputShape = tensorShapeFormat(inputs[0]);
    mUnits.resize(outputs.size());
    auto runTime       = ((OpenCLBackend *)backend())->getOpenCLRuntime();
    int inputWH[]      = {inputShape[2], inputShape[1]};
    int inputOffset[]  = {0, 0, 0, 0};
    int outputOffset[] = {0, 0, 0, 0};
    for (int i = 0; i < outputs.size(); ++i) {
        auto output      = outputs[i];
        auto outputShape = tensorShapeFormat(output);
        int region[]     = {outputShape[0], UP_DIV(outputShape[3], 4), outputShape[2], outputShape[1]};
        int outputWH[]   = {outputShape[2], outputShape[1]};

        auto &unit          = mUnits[i];
        unit.kernel         = runTime->buildKernel("blit", "blit", {});
        unit.localWorkSize  = {16, 16};
        unit.globalWorkSize = {(uint32_t)UP_DIV(region[1] * region[3], 16) * 16,
                               (uint32_t)UP_DIV(region[0] * region[2], 16) * 16};
        unit.kernel.setArg(0, openCLImage(inputs[0]));
        unit.kernel.setArg(1, openCLImage(output));
        unit.kernel.setArg(2, inputOffset);
        unit.kernel.setArg(3, outputOffset);
        unit.kernel.setArg(4, region);
        unit.kernel.setArg(5, inputWH);
        unit.kernel.setArg(6, outputWH);
        unit.kernel.setArg(7, outputWH);

        inputOffset[mAxis] += region[mAxis];
    }

    return NO_ERROR;
}

ErrorCode SliceBufferExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto runTime        = ((OpenCLBackend *)backend())->getOpenCLRuntime();
    auto bufferPool     = ((OpenCLBackend *)backend())->getBufferPool();
    auto bufferUnitSize = runTime->isSupportedFP16() ? sizeof(int16_t) : sizeof(float);
    auto inputShape     = tensorShapeFormat(inputs[0]);

    // The last output may read from channel-1 -> channel+3, alloc enough memory for avoid of memory overflow
    auto bufferSize = UP_DIV(inputShape[3] + 3, 4) * 4 * inputShape[0] * inputShape[1] * inputShape[2];

    mTempInput = bufferPool->alloc(bufferSize * bufferUnitSize);
    bufferPool->recycle(mTempInput);

    mUnits.resize(1 + outputs.size());
    int outputOffset[] = {0, 0, 0, 0};
    int inputOffset[]  = {0, 0, 0, 0};

    // NCHW's stride, use the stride of nhwc
    int inputStride[] = {inputShape[1] * inputShape[2] * inputShape[3], 1, inputShape[2] * inputShape[3],
                         inputShape[3]};
    int inputWH[]     = {inputShape[2], inputShape[1]};
    {
        int region[] = {inputShape[0], UP_DIV(inputShape[3], 4), inputShape[1], inputShape[2]};
        uint32_t gw0 = region[1] * region[3];
        uint32_t gw1 = region[0] * region[2];
        auto &unit   = mUnits[0];
        unit.kernel  = runTime->buildKernel("blitBuffer", "blitImageToBuffer", {});
        unit.kernel.setArg(0, openCLImage(inputs[0]));
        unit.kernel.setArg(1, *mTempInput);
        unit.kernel.setArg(2, inputOffset);
        unit.kernel.setArg(3, inputOffset);
        unit.kernel.setArg(4, region);
        unit.kernel.setArg(5, inputWH);
        unit.kernel.setArg(6, inputStride);
        unit.kernel.setArg(7, 4 * sizeof(int), inputShape.data());
        unit.localWorkSize  = {16, 16};
        unit.globalWorkSize = {UP_DIV(gw0, 16) * 16, UP_DIV(gw1, 16) * 16};
    }
    for (int i = 0; i < outputs.size(); ++i) {
        auto outputShape   = tensorShapeFormat(outputs[i]);
        auto &unit         = mUnits[i + 1];
        int regionBuffer[] = {outputShape[0], outputShape[3], outputShape[1], outputShape[2]};
        int region[]       = {outputShape[0], UP_DIV(outputShape[3], 4), outputShape[1], outputShape[2]};
        int outputWH[]     = {outputShape[2], outputShape[1]};
        uint32_t gw0       = region[1] * region[3];
        uint32_t gw1       = region[0] * region[2];
        unit.kernel        = runTime->buildKernel("blitBuffer", "blitBufferToImage", {});
        unit.kernel.setArg(0, *mTempInput);
        unit.kernel.setArg(1, openCLImage(outputs[i]));
        unit.kernel.setArg(2, inputOffset);
        unit.kernel.setArg(3, outputOffset);
        unit.kernel.setArg(4, region);
        unit.kernel.setArg(5, inputStride);
        unit.kernel.setArg(6, outputWH);
        unit.kernel.setArg(7, outputWH);
        unit.localWorkSize  = {16, 16};
        unit.globalWorkSize = {UP_DIV(gw0, 16) * 16, UP_DIV(gw1, 16) * 16};
        inputOffset[mAxis] += regionBuffer[mAxis];
    }
    return NO_ERROR;
}

class SliceCreator : public OpenCLBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        auto slice = op->main_as_Slice();
        auto axis  = slice->axis();
        if (0 > axis) {
            axis = inputs[0]->dimensions() + axis;
        }
        auto type = inputs[0]->getDimensionType();
        if (1 == axis) {
            for (int i = 0; i < outputs.size() - 1; ++i) {
                int channel = outputs[i]->channel();
                if (channel % 4 != 0) {
                    return new SliceBufferExecution(inputs, axis, backend);
                }
            }
        }
        return new SliceExecution(inputs, axis, backend);
    }
};

OpenCLCreatorRegister<SliceCreator> __slice_op(OpType_Slice);

} // namespace OpenCL
} // namespace MNN
