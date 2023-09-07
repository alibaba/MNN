//
//  RangeBufExecution.cpp
//  MNN
//
//  Created by MNN on 2023/08/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED
#include "backend/opencl/execution/buffer/RangeBufExecution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"

namespace MNN {
namespace OpenCL {

RangeBufExecution::RangeBufExecution(const std::string &compute, Backend* backend) : Execution(backend) {
    mBuildOptions.emplace(compute);
    // Do nothing
}
ErrorCode RangeBufExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto openCLBackend = static_cast<OpenCLBackend*>(backend());
    auto runtime       = openCLBackend->getOpenCLRuntime();
    mKernel = runtime->buildKernel("range_buf", "range_buf", mBuildOptions);
    mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel));

    std::vector<int> outputShape = tensorShapeFormat(outputs[0]);

    int batch        = outputShape.at(0);
    int outputHeight = outputShape.at(1);
    int outputWidth  = outputShape.at(2);
    int channels     = outputShape.at(3);
    int channelBlocks = (channels + 3) / 4;

    mGlobalWorkSize = {
        static_cast<uint32_t>(outputWidth),
        static_cast<uint32_t>(outputHeight),
        static_cast<uint32_t>(batch * channelBlocks)
    };

    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= mKernel.setArg(idx++, mGlobalWorkSize[0]);
    ret |= mKernel.setArg(idx++, mGlobalWorkSize[1]);
    ret |= mKernel.setArg(idx++, mGlobalWorkSize[2]);
    ret |= mKernel.setArg(idx++, openCLBuffer(inputs[0]));
    ret |= mKernel.setArg(idx++, openCLBuffer(inputs[2]));
    ret |= mKernel.setArg(idx++, openCLBuffer(outputs[0]));
    ret |= mKernel.setArg(idx++, outputWidth);
    ret |= mKernel.setArg(idx++, outputHeight);
    ret |= mKernel.setArg(idx++, channels);
    ret |= mKernel.setArg(idx++, channelBlocks);
    MNN_CHECK_CL_SUCCESS(ret, "setArg RangeBufExecution");

    std::string kernelName = "range_buf";
    mLocalSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, openCLBackend->getOpenCLRuntime(), kernelName, mKernel).first;
    return NO_ERROR;
}

ErrorCode RangeBufExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start RangeBufExecution onExecute...");
#endif
    auto mOpenCLBackend = static_cast<OpenCLBackend*>(backend());
    
#ifdef ENABLE_OPENCL_TIME_PROFILER
    cl::Event event;
    run3DKernelDefault(mKernel, mGlobalWorkSize, mLocalSize,
                       mOpenCLBackend->getOpenCLRuntime(), &event);
    
    mOpenCLBackend->getOpenCLRuntime()->pushEvent({"Range", event});
#else
    run3DKernelDefault(mKernel, mGlobalWorkSize, mLocalSize,
                       mOpenCLBackend->getOpenCLRuntime());
#endif

#ifdef LOG_VERBOSE
    MNN_PRINT("end RangeBufExecution onExecute...");
#endif
    return NO_ERROR;
}

class RangeBufCreator : public OpenCLBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        for (int i = 0; i < inputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(inputs[i], false);
        }
        for (int i = 0; i < outputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(outputs[i], false);
        }
        auto code = inputs[0]->getType().code;
        switch (code) {
            case halide_type_int:
                return new RangeBufExecution("-DUSE_INT", backend);
            case halide_type_float:
                return new RangeBufExecution("-DUSE_FLOAT", backend);
            default:
                return nullptr;
        }
    }
};

OpenCLCreatorRegister<RangeBufCreator> __RangeBuf__(OpType_Range, BUFFER);
} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
