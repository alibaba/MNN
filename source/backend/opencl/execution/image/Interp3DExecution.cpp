//
//  InterpExecution.cpp
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
    : Execution(backend) {
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
        mKernel                = runtime->buildKernel("nearest", kernelName, buildOptions);
    } else {
        MNN_ERROR("Resize types other than nearest are not supported in Interp3D opencl! Using nearest instead\n");
        mKernel                = runtime->buildKernel("nearest", kernelName, buildOptions);
    }

    mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel));
}

ErrorCode Interp3DExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
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
    mKernel.setArg(idx++, mGWS[0]);
    mKernel.setArg(idx++, mGWS[1]);
    mKernel.setArg(idx++, mGWS[2]);
    mKernel.setArg(idx++, openCLImage(input));
    mKernel.setArg(idx++, openCLImage(output));
    mKernel.setArg(idx++, mCordTransform[4]);
    mKernel.setArg(idx++, mCordTransform[2]);
    mKernel.setArg(idx++, mCordTransform[0]);
    mKernel.setArg(idx++, mCordTransform[5]);
    mKernel.setArg(idx++, mCordTransform[3]);
    mKernel.setArg(idx++, mCordTransform[1]);
    mKernel.setArg(idx++, static_cast<int32_t>(inputDepth));
    mKernel.setArg(idx++, static_cast<int32_t>(inputHeight));
    mKernel.setArg(idx++, static_cast<int32_t>(inputWidth));
    mKernel.setArg(idx++, static_cast<int32_t>(outputDepth));
    mKernel.setArg(idx++, static_cast<int32_t>(outputHeight));
    
    std::string name = "interp3D";
    mLWS = localWS3DDefault(mGWS, mMaxWorkGroupSize, runtime, name, mKernel).first;
    return NO_ERROR;

}

ErrorCode Interp3DExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start Interp3DExecution onExecute... \n");
#endif

#ifdef ENABLE_OPENCL_TIME_PROFILER
    cl::Event event;
    run3DKernelDefault(mKernel, mGWS, mLWS,
                       mOpenCLBackend->getOpenCLRuntime(), &event);
    
    int costTime = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
    MNN_PRINT("kernel cost:%d    us Interp3D\n",costTime);
#else
    run3DKernelDefault(mKernel, mGWS, mLWS, mOpenCLBackend->getOpenCLRuntime());
#endif

#ifdef LOG_VERBOSE
    MNN_PRINT("end Interp3DExecution onExecute... \n");
#endif

    return NO_ERROR;
}

OpenCLCreatorRegister<TypedCreator<Interp3DExecution>> __Interp3D_op_(OpType_Interp3D, IMAGE);

} // namespace OpenCL
} // namespace MNN
