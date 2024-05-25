//
//  InterpBufExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED

#include "backend/opencl/execution/buffer/InterpBufExecution.hpp"

namespace MNN {
namespace OpenCL {

InterpBufExecution::InterpBufExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) : CommonExecution(backend, op) {
    mUnits.resize(1);
    auto &unit = mUnits[0];
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    auto runtime   = mOpenCLBackend->getOpenCLRuntime();
    auto interpParam = op->main_as_Interp();
    mCordTransform[0] = interpParam->widthScale();
    mCordTransform[1] = interpParam->widthOffset();
    mCordTransform[2] = interpParam->heightScale();
    mCordTransform[3] = interpParam->heightOffset();

    std::set<std::string> buildOptions;
    if (op->main_as_Interp()->resizeType() == 1) {
        mKernelName = "nearest_buf";
        unit.kernel             = runtime->buildKernel("interp_buf", mKernelName, buildOptions);
    } else if(op->main_as_Interp()->resizeType() == 4) {
        mKernelName = "nearest_buf";
        buildOptions.emplace("-DUSE_ROUND");
        unit.kernel             = runtime->buildKernel("interp_buf", mKernelName, buildOptions);
    }else {
        mKernelName = "bilinear_buf";
        unit.kernel             = runtime->buildKernel("interp_buf", mKernelName, buildOptions);
    }

    mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));
}

ErrorCode InterpBufExecution::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto &unit = mUnits[0];
    Tensor *input  = inputs[0];
    Tensor *output = outputs[0];
    auto runtime = ((OpenCLBackend *)backend())->getOpenCLRuntime();

    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);

    const int inputBatch    = input->batch();
    const int inputHeight   = input->height();
    const int inputWidth    = input->width();
    const int inputChannels = input->channel();

    const int channelBlocks = UP_DIV(inputChannels, 4);

    const int outputHeight = output->height();
    const int outputWidth  = output->width();

    mGWS = {static_cast<uint32_t>(channelBlocks),
            static_cast<uint32_t>(outputWidth),
            static_cast<uint32_t>(outputHeight * inputBatch)};

    MNN_ASSERT(outputHeight > 0 && outputWidth > 0);

    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= unit.kernel->get().setArg(idx++, mGWS[0]);
    ret |= unit.kernel->get().setArg(idx++, mGWS[1]);
    ret |= unit.kernel->get().setArg(idx++, mGWS[2]);
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(input));
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(output));
    ret |= unit.kernel->get().setArg(idx++, mCordTransform[2]);
    ret |= unit.kernel->get().setArg(idx++, mCordTransform[0]);
    ret |= unit.kernel->get().setArg(idx++, mCordTransform[3]);
    ret |= unit.kernel->get().setArg(idx++, mCordTransform[1]);
    ret |= unit.kernel->get().setArg(idx++, static_cast<int32_t>(inputHeight));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int32_t>(inputWidth));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int32_t>(outputHeight));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int32_t>(outputWidth));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int32_t>(channelBlocks));
    MNN_CHECK_CL_SUCCESS(ret, "setArg InterpBufExecution");

    mLWS = localWS3DDefault(mGWS, mMaxWorkGroupSize, runtime, mKernelName, unit.kernel).first;
    mOpenCLBackend->recordKernel3d(unit.kernel, mGWS, mLWS);
    unit.globalWorkSize = {mGWS[0], mGWS[1], mGWS[2]};
    unit.localWorkSize = {mLWS[0], mLWS[1], mLWS[2]};
    return NO_ERROR;

}

class InterpBufCreator : public OpenCLBackend::Creator {
public:
    virtual ~InterpBufCreator() = default;
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, const MNN::Op *op, Backend *backend) const override {
        for (int i = 0; i < inputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(inputs[i], false);
        }
        for (int i = 0; i < outputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(outputs[i], false);
        }
        if(op->main_as_Interp()->resizeType() == 3) {
            MNN_PRINT("openCL buffer not support interp type:%d, fallback to cpu\n", op->main_as_Interp()->resizeType());
            return nullptr;
        }
        return new InterpBufExecution(inputs, op, backend);
    }
};
    
REGISTER_OPENCL_OP_CREATOR(InterpBufCreator, OpType_Interp, BUFFER);

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
