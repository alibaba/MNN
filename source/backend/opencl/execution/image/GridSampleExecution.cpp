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
    startRecord(mOpenCLBackend->getOpenCLRuntime(), mRecording);
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
    cl_int ret = CL_SUCCESS;
    ret |= mKernel.setArg(idx++, mGlobalWorkSize[0]);
    ret |= mKernel.setArg(idx++, mGlobalWorkSize[1]);
    ret |= mKernel.setArg(idx++, mGlobalWorkSize[2]);
    ret |= mKernel.setArg(idx++, openCLImage(inputTensor));
    ret |= mKernel.setArg(idx++, openCLImage(gridTensor));
    ret |= mKernel.setArg(idx++, openCLImage(outputTensor));
    ret |= mKernel.setArg(idx++, static_cast<uint32_t>(inH));
    ret |= mKernel.setArg(idx++, static_cast<uint32_t>(inW));
    ret |= mKernel.setArg(idx++, static_cast<uint32_t>(outH));
    ret |= mKernel.setArg(idx++, static_cast<uint32_t>(outW));
    ret |= mKernel.setArg(idx++, mPaddingMode);
    ret |= mKernel.setArg(idx++, mAlignCorners);
    MNN_CHECK_CL_SUCCESS(ret, "setArg GridSampleExecution");

    mLocalWorkSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, runtime, mKernelName, mKernel).first;
    recordKernel3d(mKernel, mGlobalWorkSize, mLocalWorkSize, mOpenCLBackend->getOpenCLRuntime());
    endRecord(mOpenCLBackend->getOpenCLRuntime(), mRecording);
    return NO_ERROR;
}

ErrorCode GridSampleExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef ENABLE_OPENCL_TIME_PROFILER
    cl::Event event;
    run3DKernelDefault(mKernel, mGlobalWorkSize, mLocalWorkSize,
        mOpenCLBackend->getOpenCLRuntime(), &event);
    
    mOpenCLBackend->getOpenCLRuntime()->pushEvent({"GridSample", event});
#else
    if(mOpenCLBackend->getOpenCLRuntime()->isUseRecordQueue()){
        if(mOpenCLBackend->getOpenCLRuntime()->isDevideOpRecord())
            mOpenCLBackend->getOpenCLRuntime()->getRecordings()->emplace_back(mRecording);
        return NO_ERROR;
    }
    run3DKernelDefault(mKernel, mGlobalWorkSize, mLocalWorkSize, mOpenCLBackend->getOpenCLRuntime());
#endif
    return NO_ERROR;
}

OpenCLCreatorRegister<TypedCreator<GridSampleExecution>> __GridSample_op_(OpType_GridSample, IMAGE);

} // namespace OpenCL
} // namespace MNN
