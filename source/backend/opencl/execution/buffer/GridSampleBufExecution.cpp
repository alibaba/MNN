//
//  GridSampleBufExecution.cpp
//  MNN
//
//  Created by MNN on 2021/08/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED

#include "backend/opencl/execution/buffer/GridSampleBufExecution.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace OpenCL {
GridSampleBufExecution::GridSampleBufExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend)
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
        mKernelName = "bilinear_buf";
        mKernel = runtime->buildKernel("grid_sample_buf", mKernelName, buildOptions);
    }
    else {
        mKernelName = "nearest_buf";
        mKernel = runtime->buildKernel("grid_sample_buf", mKernelName, buildOptions);
    }
    mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel));
}

ErrorCode GridSampleBufExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
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
    ret |= mKernel.setArg(idx++, openCLBuffer(inputTensor));
    ret |= mKernel.setArg(idx++, openCLBuffer(gridTensor));
    ret |= mKernel.setArg(idx++, openCLBuffer(outputTensor));
    ret |= mKernel.setArg(idx++, static_cast<uint32_t>(inH));
    ret |= mKernel.setArg(idx++, static_cast<uint32_t>(inW));
    ret |= mKernel.setArg(idx++, static_cast<uint32_t>(outH));
    ret |= mKernel.setArg(idx++, static_cast<uint32_t>(outW));
    ret |= mKernel.setArg(idx++, static_cast<uint32_t>(channelC4));
    ret |= mKernel.setArg(idx++, mPaddingMode);
    ret |= mKernel.setArg(idx++, mAlignCorners);
    MNN_CHECK_CL_SUCCESS(ret, "setArg GridSampleBufExecution");

    mLocalWorkSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, runtime, mKernelName, mKernel).first;

    return NO_ERROR;
}

ErrorCode GridSampleBufExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef ENABLE_OPENCL_TIME_PROFILER
    cl::Event event;
    run3DKernelDefault(mKernel, mGlobalWorkSize, mLocalWorkSize,
        mOpenCLBackend->getOpenCLRuntime(), &event);
    
    mOpenCLBackend->getOpenCLRuntime()->pushEvent({"GridSample", event});
#else
    run3DKernelDefault(mKernel, mGlobalWorkSize, mLocalWorkSize, mOpenCLBackend->getOpenCLRuntime());
#endif
    return NO_ERROR;
}

class GridSampleBufCreator :public OpenCLBackend::Creator {
public:
    virtual ~GridSampleBufCreator() = default;
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
        const MNN::Op *op, Backend *backend) const override {
        for (int i = 0; i < inputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(inputs[i], false);
        }
        for (int i = 0; i < outputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(outputs[i], false);
        }
        if (op->main_as_GridSample()->mode() != 0 && op->main_as_GridSample()->mode() != 1) {
            MNN_PRINT("openCL buffer not support interpolate type: %d, fallback to cpu\n", op->main_as_GridSample()->mode());
            return nullptr;
        }
        return new GridSampleBufExecution(inputs, op, backend);
    }
};

OpenCLCreatorRegister<GridSampleBufCreator> __GridSampleBuf_op_(OpType_GridSample, BUFFER);

} // namespace OpenCL
} // namespace MNN

#endif // MNN_OPENCL_BUFFER_CLOSED
