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
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    mMode = op->main_as_GridSample()->mode();
    mPaddingMode = op->main_as_GridSample()->paddingMode();
    if (op->main_as_GridSample()->alignCorners()) {
        mAlignCorners = 1;
    }else {
        mAlignCorners = 0;
    }
}

ErrorCode GridSampleExecution::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    auto inputTensor = inputs[0];
    auto gridTensor = inputs[1];
    auto outputTensor = outputs[0];
    if(outputs[0]->dimensions() > 4){
        mUnits.resize(1);
        const int batches = inputTensor->buffer().dim[0].extent;
        const int channels = inputTensor->buffer().dim[1].extent;
        const int inD = inputTensor->buffer().dim[2].extent;
        const int inH = inputTensor->buffer().dim[3].extent;
        const int inW = inputTensor->buffer().dim[4].extent;
        const int channelC4 = UP_DIV(channels, 4);
        const int outD = outputTensor->buffer().dim[2].extent;
        const int outH = outputTensor->buffer().dim[3].extent;
        const int outW = outputTensor->buffer().dim[4].extent;
        std::vector<int> outputShape = tensorShapeFormat(gridTensor);
        auto &unit = mUnits[0];
        std::set<std::string> buildOptions;
        if (mMode == 0) {
            mKernelName = "bilinear5d";
            unit.kernel = runtime->buildKernel("grid_sample", mKernelName, buildOptions);
        }
        else {
            mKernelName = "nearest5d";
            unit.kernel = runtime->buildKernel("grid_sample", mKernelName, buildOptions);
            
        }
        mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));
        mGlobalWorkSize = {
            static_cast<uint32_t>(channelC4 * outD),
            static_cast<uint32_t>(outW),
            static_cast<uint32_t>(outH * batches)
        };
        MNN_ASSERT(outW > 0 && outH > 0);
        
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[0]);
        ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[1]);
        ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[2]);
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(inputTensor));
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(gridTensor));
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(outputTensor));
        ret |= unit.kernel->get().setArg(idx++, static_cast<uint32_t>(inH));
        ret |= unit.kernel->get().setArg(idx++, static_cast<uint32_t>(inW));
        ret |= unit.kernel->get().setArg(idx++, static_cast<uint32_t>(inD));
        ret |= unit.kernel->get().setArg(idx++, static_cast<uint32_t>(outH));
        ret |= unit.kernel->get().setArg(idx++, static_cast<uint32_t>(outW));
        ret |= unit.kernel->get().setArg(idx++, static_cast<uint32_t>(outD));
        ret |= unit.kernel->get().setArg(idx++, static_cast<uint32_t>(batches));
        ret |= unit.kernel->get().setArg(idx++, mPaddingMode);
        ret |= unit.kernel->get().setArg(idx++, mAlignCorners);
        MNN_CHECK_CL_SUCCESS(ret, "setArg GridSampleExecution");
        
        mLocalWorkSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, runtime, mKernelName, unit.kernel).first;
        
        mOpenCLBackend->recordKernel3d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
        unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
        unit.localWorkSize = {mLocalWorkSize[0], mLocalWorkSize[1], mLocalWorkSize[2]};
    }else{
        mUnits.resize(1);
        auto &unit = mUnits[0];
        const int batches = inputTensor->buffer().dim[0].extent;
        const int channels = inputTensor->buffer().dim[1].extent;
        const int inH = inputTensor->buffer().dim[2].extent;
        const int inW = inputTensor->buffer().dim[3].extent;
        const int channelC4 = UP_DIV(channels, 4);
        
        const int outH = outputTensor->buffer().dim[2].extent;
        const int outW = outputTensor->buffer().dim[3].extent;
        
        std::set<std::string> buildOptions;
        if (mMode == 0) {
            mKernelName = "bilinear";
            unit.kernel = runtime->buildKernel("grid_sample", mKernelName, buildOptions);
        }
        else {
            mKernelName = "nearest";
            unit.kernel = runtime->buildKernel("grid_sample", mKernelName, buildOptions);
            
        }
        
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
    }
    return NO_ERROR;
}

using GridSampleCreator = TypedCreator<GridSampleExecution>;
REGISTER_OPENCL_OP_CREATOR(GridSampleCreator, OpType_GridSample, IMAGE);

} // namespace OpenCL
} // namespace MNN
