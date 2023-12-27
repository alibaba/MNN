//
//  RangeBufExecution.cpp
//  MNN
//
//  Created by MNN on 2023/12/1.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/image/RangeExecution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"

namespace MNN {
namespace OpenCL {

RangeExecution::RangeExecution(const std::string &compute, Backend* backend) : Execution(backend) {
    mBuildOptions.emplace(compute);
    // Do nothing
}
ErrorCode RangeExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto openCLBackend = static_cast<OpenCLBackend*>(backend());
    auto runtime       = openCLBackend->getOpenCLRuntime();
    openCLBackend->startRecord(mRecording);
    mKernel = runtime->buildKernel("range", "range", mBuildOptions);
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
    ret |= mKernel.setArg(idx++, openCLImage(inputs[0]));
    ret |= mKernel.setArg(idx++, openCLImage(inputs[2]));
    ret |= mKernel.setArg(idx++, openCLImage(outputs[0]));
    ret |= mKernel.setArg(idx++, outputWidth);
    ret |= mKernel.setArg(idx++, outputHeight);
    ret |= mKernel.setArg(idx++, channels);
    ret |= mKernel.setArg(idx++, channelBlocks);
    MNN_CHECK_CL_SUCCESS(ret, "setArg RangeExecution");

    std::string kernelName = "range";
    mLocalSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, openCLBackend->getOpenCLRuntime(), kernelName, mKernel).first;
    openCLBackend->recordKernel3d(mKernel, mGlobalWorkSize, mLocalSize);
    openCLBackend->endRecord(mRecording);
    return NO_ERROR;
}

ErrorCode RangeExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
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
    if(mOpenCLBackend->isUseRecordQueue()){
        if(mOpenCLBackend->isDevideOpRecord())
            mOpenCLBackend->addRecord(mRecording);
#ifdef LOG_VERBOSE
        MNN_PRINT("End RangeExecution onExecute... \n");
#endif
        return NO_ERROR;
    }
    run3DKernelDefault(mKernel, mGlobalWorkSize, mLocalSize,
                       mOpenCLBackend->getOpenCLRuntime());
#endif

#ifdef LOG_VERBOSE
    MNN_PRINT("end RangeBufExecution onExecute...");
#endif
    return NO_ERROR;
}

class RangeCreator : public OpenCLBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        auto code = inputs[0]->getType().code;
        switch (code) {
            case halide_type_int:
                return new RangeExecution("-DUSE_INT", backend);
            case halide_type_float:
                return new RangeExecution("-DUSE_FLOAT", backend);
            default:
                return nullptr;
        }
    }
};

REGISTER_OPENCL_OP_CREATOR(RangeCreator, OpType_Range, IMAGE);
} // namespace OpenCL
} // namespace MNN
