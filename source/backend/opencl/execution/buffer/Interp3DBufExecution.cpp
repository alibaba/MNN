//
//  Interp3DBufExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED

#include "backend/opencl/execution/buffer/Interp3DBufExecution.hpp"

namespace MNN {
namespace OpenCL {

Interp3DBufExecution::Interp3DBufExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) : CommonExecution(backend, op) {
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
    if (op->main_as_Interp()->resizeType() == 1) {
        mKernelName = "nearest3D_buf";
        unit.kernel                = runtime->buildKernel("interp_buf", mKernelName, buildOptions);
    } else {
        MNN_ERROR("Resize type other than nearest is not supported in Interp3DBuf, change to nearest!");
        mKernelName = "nearest3D_buf";
        unit.kernel                = runtime->buildKernel("interp_buf", mKernelName, buildOptions);
    }

    mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));
}

ErrorCode Interp3DBufExecution::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto &unit = mUnits[0];
    Tensor *input  = inputs[0];
    Tensor *output = outputs[0];
    auto runtime = ((OpenCLBackend *)backend())->getOpenCLRuntime();

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
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(input));
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(output));
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
    ret |= unit.kernel->get().setArg(idx++, static_cast<int32_t>(outputWidth));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int32_t>(channelBlocks));
    MNN_CHECK_CL_SUCCESS(ret, "setArg Interp3DBufExecution");

    mLWS = localWS3DDefault(mGWS, mMaxWorkGroupSize, runtime, mKernelName, unit.kernel).first;
    mOpenCLBackend->recordKernel3d(unit.kernel, mGWS, mLWS);
    unit.globalWorkSize = {mGWS[0], mGWS[1], mGWS[2]};
    unit.localWorkSize = {mLWS[0], mLWS[1], mLWS[2]};
    return NO_ERROR;

}

class Interp3DBufCreator : public OpenCLBackend::Creator {
public:
    virtual ~Interp3DBufCreator() = default;
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        for (int i = 0; i < inputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(inputs[i], false);
        }
        for (int i = 0; i < outputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(outputs[i], false);
        }
        return new Interp3DBufExecution(inputs, op, backend);
        ;
    }
};

REGISTER_OPENCL_OP_CREATOR(Interp3DBufCreator, OpType_Interp3D, BUFFER);

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
