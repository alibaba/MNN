//
//  InterpExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/InterpExecution.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace OpenCL {

InterpExecution::InterpExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend)
    : Execution(backend) {
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    auto runtime   = mOpenCLBackend->getOpenCLRuntime();
    auto interpParam = op->main_as_Interp();
    mCordTransform[0] = interpParam->widthScale();
    mCordTransform[1] = interpParam->widthOffset();
    mCordTransform[2] = interpParam->heightScale();
    mCordTransform[3] = interpParam->heightOffset();

    std::set<std::string> buildOptions;
    std::string kernelName = "interp";
    if (op->main_as_Interp()->resizeType() == 1) {
        mKernel                = runtime->buildKernel("nearest", kernelName, buildOptions);
    } else {
        mKernel                = runtime->buildKernel("interp", kernelName, buildOptions);
    }

    mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel));
}

std::vector<uint32_t> InterpExecution::interpLocalWS(const std::vector<uint32_t> &gws,
                                                     const uint32_t maxWorkGroupSize) {
    std::vector<uint32_t> lws(4, 0);
    GpuType gpuType             = mOpenCLBackend->getOpenCLRuntime()->getGpuType();
    uint32_t deviceComputeUnits = mOpenCLBackend->getOpenCLRuntime()->deviceComputeUnits();
    if (gpuType == GpuType::ADRENO) {
        lws[0] = deviceComputeUnits * 4;
        lws[1] = 4;
        lws[2] = 1;
    } else {
        lws[0] = deviceComputeUnits * 2;
        lws[1] = 4;
        lws[2] = 1;
    }
    for (int i = 0; i < gws.size(); ++i) {
        while (gws[i] % lws[i] != 0) {
            --lws[i];
        }
    }
    return lws;
}

ErrorCode InterpExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
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
    mKernel.setArg(idx++, mGWS[0]);
    mKernel.setArg(idx++, mGWS[1]);
    mKernel.setArg(idx++, mGWS[2]);
    mKernel.setArg(idx++, openCLImage(input));
    mKernel.setArg(idx++, openCLImage(output));
    mKernel.setArg(idx++, mCordTransform[2]);
    mKernel.setArg(idx++, mCordTransform[0]);
    mKernel.setArg(idx++, mCordTransform[3]);
    mKernel.setArg(idx++, mCordTransform[1]);
    mKernel.setArg(idx++, static_cast<int32_t>(inputHeight));
    mKernel.setArg(idx++, static_cast<int32_t>(inputWidth));
    mKernel.setArg(idx++, static_cast<int32_t>(outputHeight));
    
    std::string name = "interp";
    mLWS = localWS3DDefault(mGWS, mMaxWorkGroupSize, runtime, name, mKernel);
    return NO_ERROR;

}

ErrorCode InterpExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start InterpExecution onExecute... \n");
#endif

#ifdef ENABLE_OPENCL_TIME_PROFILER
    cl::Event event;
    run3DKernelDefault(mKernel, mGWS, mLWS,
                       mOpenCLBackend->getOpenCLRuntime(), &event);
    
    int costTime = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
    MNN_PRINT("kernel cost:%d    us Interp\n",costTime);
#else
    run3DKernelDefault(mKernel, mGWS, mLWS, mOpenCLBackend->getOpenCLRuntime());
#endif

#ifdef LOG_VERBOSE
    MNN_PRINT("end InterpExecution onExecute... \n");
#endif

    return NO_ERROR;
}

OpenCLCreatorRegister<TypedCreator<InterpExecution>> __Interp_op_(OpType_Interp);

} // namespace OpenCL
} // namespace MNN
