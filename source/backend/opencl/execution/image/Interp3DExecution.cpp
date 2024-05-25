//
//  Interp3DExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/image/Interp3DExecution.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace OpenCL {

Interp3DExecution::Interp3DExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend)
    : CommonExecution(backend, op) {
    mUnits.resize(1);
    auto &unit = mUnits[0];
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    auto runtime   = mOpenCLBackend->getOpenCLRuntime();
    auto interp3DParam = op->main_as_Interp();
    mCordTransform[0] = interp3DParam->widthScale();
    mCordTransform[1] = interp3DParam->widthOffset();
    mCordTransform[2] = interp3DParam->heightScale();
    mCordTransform[3] = interp3DParam->heightOffset();
    mCordTransform[4] = interp3DParam->depthScale();
    mCordTransform[5] = interp3DParam->depthOffset();

    std::set<std::string> buildOptions;
    std::string kernelName = "interp3D";
    if (op->main_as_Interp()->resizeType() == 1) {
        unit.kernel                = runtime->buildKernel("nearest", kernelName, buildOptions);
    } else {
        MNN_ERROR("Resize types other than nearest are not supported in Interp3D opencl! Using nearest instead\n");
        unit.kernel                = runtime->buildKernel("nearest", kernelName, buildOptions);
    }

    mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));
}

ErrorCode Interp3DExecution::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    Tensor *input  = inputs[0];
    Tensor *output = outputs[0];
    auto &unit = mUnits[0];

    std::vector<int> inputImageShape  = tensorShapeFormat(input); // {C/4 * H * W, N * D} for 5-D Tensor
    std::vector<int> outputImageShape = tensorShapeFormat(output);

    auto inputShape = input->shape();
    auto outputShape = output->shape();
    const int inputBatch    = inputShape[0];
    const int inputChannels = inputShape[1];
    const int inputDepth    = inputShape[2];
    const int inputHeight   = inputShape[3];
    const int inputWidth    = inputShape[4];

    const int channelBlocks = UP_DIV(inputChannels, 4);

    const int outputDepth    = outputShape[2];
    const int outputHeight   = outputShape[3];
    const int outputWidth    = outputShape[4];

    mGWS = {static_cast<uint32_t>(channelBlocks),
            static_cast<uint32_t>(outputHeight * outputWidth),
            static_cast<uint32_t>(outputDepth * inputBatch)};

    MNN_ASSERT(outputDepth > 0 && outputHeight > 0 && outputWidth > 0);

    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= unit.kernel->get().setArg(idx++, mGWS[0]);
    ret |= unit.kernel->get().setArg(idx++, mGWS[1]);
    ret |= unit.kernel->get().setArg(idx++, mGWS[2]);
    ret |= unit.kernel->get().setArg(idx++, openCLImage(input));
    ret |= unit.kernel->get().setArg(idx++, openCLImage(output));
    ret |= unit.kernel->get().setArg(idx++, mCordTransform[4]);
    ret |= unit.kernel->get().setArg(idx++, mCordTransform[2]);
    ret |= unit.kernel->get().setArg(idx++, mCordTransform[0]);
    ret |= unit.kernel->get().setArg(idx++, mCordTransform[5]);
    ret |= unit.kernel->get().setArg(idx++, mCordTransform[3]);
    ret |= unit.kernel->get().setArg(idx++, mCordTransform[1]);
    ret |= unit.kernel->get().setArg(idx++, static_cast<int32_t>(inputDepth));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int32_t>(inputHeight));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int32_t>(inputWidth));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int32_t>(outputDepth));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int32_t>(outputHeight));
    MNN_CHECK_CL_SUCCESS(ret, "setArg Intep3DExecution");

    std::string name = "interp3D";
    mLWS = localWS3DDefault(mGWS, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), name, unit.kernel).first;
    mOpenCLBackend->recordKernel3d(unit.kernel, mGWS, mLWS);
    unit.globalWorkSize = {mGWS[0], mGWS[1], mGWS[2]};
    unit.localWorkSize = {mLWS[0], mLWS[1], mLWS[2]};
    return NO_ERROR;

}

using Interp3DCreator = TypedCreator<Interp3DExecution>;
REGISTER_OPENCL_OP_CREATOR(Interp3DCreator, OpType_Interp3D, IMAGE);

} // namespace OpenCL
} // namespace MNN
