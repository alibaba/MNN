//
//  PermuteExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "PermuteExecution.hpp"
#include <Macro.h>
#include "TensorUtils.hpp"
#include "core/OpenCLBackend.hpp"

namespace MNN {
namespace OpenCL {

PermuteExecution::PermuteExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend)
    : CommonExecution(backend) {
    auto shape = op->main_as_Permute()->dims();
    // FIXME, support less than 4
    MNN_ASSERT(shape->size() == 4);
    mDims.resize(4);
    for (int i = 0; i < shape->size(); ++i) {
        auto dim   = shape->data()[i];
        mDims[dim] = i;
    }
}

ErrorCode PermuteExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    Tensor *input  = inputs[0];
    Tensor *output = outputs[0];
    // FIXME, support nhwc format
    MNN_ASSERT(input->getDimensionType() != Tensor::TENSORFLOW);

    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);

    auto runTime        = ((OpenCLBackend *)backend())->getOpenCLRuntime();
    auto bufferPool     = ((OpenCLBackend *)backend())->getBufferPool();
    auto bufferUnitSize = runTime->isSupportedFP16() ? sizeof(int16_t) : sizeof(float);
    auto bufferSize0    = UP_DIV(outputShape[3], 4) * 4 * outputShape[0] * outputShape[1] * outputShape[2];
    auto bufferSize1    = UP_DIV(inputShape[3], 4) * 4 * inputShape[0] * inputShape[1] * inputShape[2];
    mTempInput          = bufferPool->alloc(std::max(bufferSize0, bufferSize1) * bufferUnitSize);
    bufferPool->recycle(mTempInput);

    mUnits.resize(2);
    int offset[] = {0, 0, 0, 0};

    // NCHW's stride, use the stride of nhwc
    int outputStride[] = {outputShape[1] * outputShape[2] * outputShape[3], 1, outputShape[2] * outputShape[3],
                          outputShape[3]};
    int permuteInputStride[4];
    for (int i = 0; i < mDims.size(); ++i) {
        permuteInputStride[i] = outputStride[mDims[i]];
    }
    int inputWH[]  = {inputShape[2], inputShape[1]};
    int outputWH[] = {outputShape[2], outputShape[1]};
    {
        int region[] = {inputShape[0], UP_DIV(inputShape[3], 4), inputShape[1], inputShape[2]};
        uint32_t gw0 = region[1] * region[3];
        uint32_t gw1 = region[0] * region[2];
        auto &unit   = mUnits[0];
        unit.kernel  = runTime->buildKernel("blitBuffer", "blitImageToBuffer", {});
        unit.kernel.setArg(0, openCLImage(inputs[0]));
        unit.kernel.setArg(1, *mTempInput);
        unit.kernel.setArg(2, offset);
        unit.kernel.setArg(3, offset);
        unit.kernel.setArg(4, region);
        unit.kernel.setArg(5, inputWH);
        unit.kernel.setArg(6, 4 * sizeof(int), permuteInputStride);
        unit.kernel.setArg(7, 4 * sizeof(int), inputShape.data());
        unit.localWorkSize  = {16, 16};
        unit.globalWorkSize = {UP_DIV(gw0, 16) * 16, UP_DIV(gw1, 16) * 16};
    }
    {
        int region[] = {outputShape[0], UP_DIV(outputShape[3], 4), outputShape[1], outputShape[2]};

        auto &unit          = mUnits[1];
        unit.kernel         = runTime->buildKernel("blitBuffer", "blitBufferToImage", {});
        unit.localWorkSize  = {16, 16};
        unit.globalWorkSize = {(uint32_t)UP_DIV(region[3] * region[1], 16) * 16,
                               (uint32_t)UP_DIV(region[2] * region[0], 16) * 16};
        unit.kernel.setArg(0, *mTempInput);
        unit.kernel.setArg(1, openCLImage(output));
        unit.kernel.setArg(2, offset);
        unit.kernel.setArg(3, offset);
        unit.kernel.setArg(4, region);
        unit.kernel.setArg(5, outputStride);
        unit.kernel.setArg(6, outputWH);
        unit.kernel.setArg(7, outputWH);
    }
    return NO_ERROR;
}

OpenCLCreatorRegister<TypedCreator<PermuteExecution>> __permute_op(OpType_Permute);

} // namespace OpenCL
} // namespace MNN
