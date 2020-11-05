//
//  ResizeExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/ResizeExecution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace OpenCL {

ResizeExecution::ResizeExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend)
    : Execution(backend) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start ResizeExecution init !\n");
#endif
    mOpenCLBackend          = static_cast<OpenCLBackend *>(backend);
    const auto *scaleParams = op->main_as_Resize();
    mXScale                 = scaleParams->xScale();
    mYScale                 = scaleParams->yScale();
#ifdef LOG_VERBOSE
    MNN_PRINT("end ResizeExecution init !\n");
#endif
}

ErrorCode ResizeExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start ResizeExecution onResize !\n");
#endif
    auto runtime = mOpenCLBackend->getOpenCLRuntime();

    if (mKernel.get() == nullptr) {
        mKernel           = runtime->buildKernel("interp", "interp", {});
        mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel));
    }
    
    Tensor *input                = inputs[0];
    Tensor *output               = outputs[0];
    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);

    const float x_scaling_ = 1.0 / mXScale;
    const float y_scaling_ = 1.0 / mYScale;

    const int batch    = outputShape.at(0);
    const int height   = outputShape.at(1);
    const int width    = outputShape.at(2);
    const int channels = outputShape.at(3);

    const int channelBlocks = UP_DIV(channels, 4);
    const int inputHeight   = input->height();
    const int inputWidth    = input->width();

    const std::vector<uint32_t> gws = {static_cast<uint32_t>(channelBlocks),
                                       static_cast<uint32_t>(width),
                                       static_cast<uint32_t>(height * batch)};

    uint32_t idx = 0;

    mKernel.setArg(idx++, gws[0]);
    mKernel.setArg(idx++, gws[1]);
    mKernel.setArg(idx++, gws[2]);
    mKernel.setArg(idx++, openCLImage(input));
    mKernel.setArg(idx++, openCLImage(output));
    mKernel.setArg(idx++, y_scaling_);
    mKernel.setArg(idx++, x_scaling_);
    mKernel.setArg(idx++, static_cast<int32_t>(inputHeight));
    mKernel.setArg(idx++, static_cast<int32_t>(inputWidth));
    mKernel.setArg(idx++, static_cast<int32_t>(height));

    std::string name = "Interp";
    mLWS = localWS3DDefault(gws, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), name, mKernel);

    for (size_t i = 0; i < mLWS.size(); ++i) {
        if (mLWS[i] != 0) {
            mGWS[i] = ROUND_UP(gws[i], std::max((uint32_t)1, mLWS[i]));
        }
    }

#ifdef LOG_VERBOSE
    MNN_PRINT("end ResizeExecution onResize !\n");
#endif
    return NO_ERROR;
}

ErrorCode ResizeExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start ResizeExecution onExecute !\n");
#endif

    auto runtime = mOpenCLBackend->getOpenCLRuntime();

#ifdef ENABLE_OPENCL_TIME_PROFILER
    cl::Event event;
    auto error = runtime->commandQueue().enqueueNDRangeKernel(
        mKernel, cl::NullRange,
        cl::NDRange(mGWS[0], mGWS[1], mGWS[2]),
        cl::NDRange(mLWS[0], mLWS[1], mLWS[2]), nullptr, &event);
    
    int costTime = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
    MNN_PRINT("kernel cost:%d    us Resize\n",costTime);
#else
    auto error = runtime->commandQueue().enqueueNDRangeKernel(
        mKernel, cl::NullRange,
        cl::NDRange(mGWS[0], mGWS[1], mGWS[2]),
        cl::NDRange(mLWS[0], mLWS[1], mLWS[2]), nullptr, nullptr);
#endif
    
    MNN_CHECK_CL_SUCCESS(error);
#ifdef LOG_VERBOSE
    MNN_PRINT("end ResizeExecution onExecute !\n");
#endif
    return NO_ERROR;
}

OpenCLCreatorRegister<TypedCreator<ResizeExecution>> __resize_op(OpType_Resize);

} // namespace OpenCL
} // namespace MNN
