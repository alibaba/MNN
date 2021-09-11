//
//  GridSampleExecution.cpp
//  MNN
//
//  Created by MNN on 2021/08/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/image/GridSampleExecution.hpp"
#include "core/TensorUtils.hpp"
#include "backend/cpu/CPUTensorConvert.hpp"

namespace MNN {
namespace OpenCL {
GridSampleExecution::GridSampleExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend)
    : Execution(backend) {
    mPaddingMode = op->main_as_GridSample()->paddingMode();
    if (op->main_as_GridSample()->alignCorners()) {
        mAlignCorners = 1;
    }
    else {
        mAlignCorners = 0;
    }

    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    auto gridSampleParam = op->main_as_GridSample();

    std::set<std::string> buildOptions;
    if (op->main_as_GridSample()->mode() == 0) {
        mKernelName = "bilinear";
        mKernel = runtime->buildKernel("grid_sample", mKernelName, buildOptions);
    }
    else {
        mKernelName = "nearest";
        mKernel = runtime->buildKernel("grid_sample", mKernelName, buildOptions);

    }

    mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel));
}

ErrorCode GridSampleExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto inputTensor = inputs[0];
    auto gridTensor = inputs[1];
    auto outputTensor = outputs[0];
    auto runtime = ((OpenCLBackend *)backend())->getOpenCLRuntime();

    const int batches = inputTensor->buffer().dim[0].extent;
    const int channels = inputTensor->buffer().dim[1].extent;
    const int inH = inputTensor->buffer().dim[2].extent;
    const int inW = inputTensor->buffer().dim[3].extent;
    const int channelC4 = UP_DIV(channels, 4);

    const int outH = outputTensor->buffer().dim[2].extent;
    const int outW = outputTensor->buffer().dim[3].extent;

    mGlobalWorkSize = {
        static_cast<uint32_t>(channelC4),
        static_cast<uint32_t>(outW),
        static_cast<uint32_t>(outH * batches)
    };

    MNN_ASSERT(outW > 0 && outH > 0);

    uint32_t idx = 0;
    mKernel.setArg(idx++, mGlobalWorkSize[0]);
    mKernel.setArg(idx++, mGlobalWorkSize[1]);
    mKernel.setArg(idx++, mGlobalWorkSize[2]);
    mKernel.setArg(idx++, openCLImage(inputTensor));
    mKernel.setArg(idx++, openCLImage(gridTensor));
    mKernel.setArg(idx++, openCLImage(outputTensor));
    mKernel.setArg(idx++, static_cast<uint32_t>(inH));
    mKernel.setArg(idx++, static_cast<uint32_t>(inW));
    mKernel.setArg(idx++, static_cast<uint32_t>(outH));
    mKernel.setArg(idx++, static_cast<uint32_t>(outW));
    mKernel.setArg(idx++, mPaddingMode);
    mKernel.setArg(idx++, mAlignCorners);

    mLocalWorkSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, runtime, mKernelName, mKernel).first;

    return NO_ERROR;
}

ErrorCode GridSampleExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef ENABLE_OPENCL_TIME_PROFILER
    cl::Event event;
    run3DKernelDefault(mKernel, mGlobalWorkSize, mLocalWorkSize,
        mOpenCLBackend->getOpenCLRuntime(), &event);

    int costTime = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
    MNN_PRINT("kernel cost:%d    us GridSample\n", costTime);
#else
    run3DKernelDefault(mKernel, mGlobalWorkSize, mLocalWorkSize, mOpenCLBackend->getOpenCLRuntime());
#endif
    return NO_ERROR;
}

OpenCLCreatorRegister<TypedCreator<GridSampleExecution>> __GridSample_op_(OpType_GridSample, IMAGE);

} // namespace OpenCL
} // namespace MNN