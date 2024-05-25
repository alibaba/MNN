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
    : CommonExecution(backend, op) {
    mUnits.resize(1);
    auto &unit = mUnits[0];
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
        unit.kernel = runtime->buildKernel("grid_sample", mKernelName, buildOptions);
    }
    else {
        mKernelName = "nearest";
        unit.kernel = runtime->buildKernel("grid_sample", mKernelName, buildOptions);

    }

    mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));
}

ErrorCode GridSampleExecution::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto &unit = mUnits[0];
    auto inputTensor = inputs[0];
    auto gridTensor = inputs[1];
    auto outputTensor = outputs[0];

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
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[0]);
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[1]);
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[2]);
    ret |= unit.kernel->get().setArg(idx++, openCLImage(inputTensor));
    ret |= unit.kernel->get().setArg(idx++, openCLImage(gridTensor));
    ret |= unit.kernel->get().setArg(idx++, openCLImage(outputTensor));
    ret |= unit.kernel->get().setArg(idx++, static_cast<uint32_t>(inH));
    ret |= unit.kernel->get().setArg(idx++, static_cast<uint32_t>(inW));
    ret |= unit.kernel->get().setArg(idx++, static_cast<uint32_t>(outH));
    ret |= unit.kernel->get().setArg(idx++, static_cast<uint32_t>(outW));
    ret |= unit.kernel->get().setArg(idx++, mPaddingMode);
    ret |= unit.kernel->get().setArg(idx++, mAlignCorners);
    MNN_CHECK_CL_SUCCESS(ret, "setArg GridSampleExecution");

    mLocalWorkSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), mKernelName, unit.kernel).first;
    mOpenCLBackend->recordKernel3d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
    unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
    unit.localWorkSize = {mLocalWorkSize[0], mLocalWorkSize[1], mLocalWorkSize[2]};
    return NO_ERROR;
}

using GridSampleCreator = TypedCreator<GridSampleExecution>;
REGISTER_OPENCL_OP_CREATOR(GridSampleCreator, OpType_GridSample, IMAGE);

} // namespace OpenCL
} // namespace MNN
