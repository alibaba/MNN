//
//  FuseExecution.cpp
//  MNN
//
//  Created by MNN on 2022/11/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/image/FuseExecution.hpp"
#include "core/Macro.h"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"

namespace MNN {
namespace OpenCL {

FuseExecution::FuseExecution(const std::vector<Tensor *> &inputs, Backend *backend, const Op* op)
    : Execution(backend) {
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    buildFuseKernel(op);
}

bool FuseExecution::buildFuseKernel(const Op* op) {
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    if (mKernel.get() == nullptr) {
        std::set<std::string> buildOptions;
        std::string kernelName;
        auto extra = op->main_as_Extra();
        auto source = reinterpret_cast<const char*>(extra->info()->data());
        auto name = extra->type()->c_str();
        mKernelName = extra->type()->str();
        mKernel = runtime->buildKernelFromSource(source, name, buildOptions);
        mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel));
    }
    return true;
}

ErrorCode FuseExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    startRecord(mOpenCLBackend->getOpenCLRuntime(), mRecording);
    Tensor *input  = inputs[0];
    Tensor *output = outputs[0];

    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);

    const int outputBatch    = outputShape.at(0);
    const int outputHeight   = outputShape.at(1);
    const int outputWidth    = outputShape.at(2);
    const int outputChannels = outputShape.at(3);

    const int channelBlocks  = UP_DIV(outputChannels, 4);
    const int remainChannels = channelBlocks * 4 - outputChannels;
    mGlobalWorkSize = {
        static_cast<uint32_t>(channelBlocks),
        static_cast<uint32_t>(outputWidth),
        static_cast<uint32_t>(outputHeight * outputBatch)
    };
    
    uint32_t idx    = 0;
    cl_int ret = CL_SUCCESS;
    for (auto input : inputs) {
        ret |= mKernel.setArg(idx++, openCLImage(input));
    }
    for (auto output : outputs) {
        ret |= mKernel.setArg(idx++, openCLImage(output));
    }
    ret |= mKernel.setArg(idx++, mGlobalWorkSize[0]);
    ret |= mKernel.setArg(idx++, mGlobalWorkSize[1]);
    ret |= mKernel.setArg(idx++, mGlobalWorkSize[2]);
    MNN_CHECK_CL_SUCCESS(ret, "setArg FuseExecution");

    mLocalWorkSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), mKernelName, mKernel).first;
    recordKernel3d(mKernel, mGlobalWorkSize, mLocalWorkSize, mOpenCLBackend->getOpenCLRuntime());
    endRecord(mOpenCLBackend->getOpenCLRuntime(), mRecording);
    return NO_ERROR;
}

ErrorCode FuseExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start FuseExecution onExecute !\n");
#endif
#ifdef ENABLE_OPENCL_TIME_PROFILER
    cl::Event event;
    run3DKernelDefault(mKernel, mGlobalWorkSize, mLocalWorkSize,
                       mOpenCLBackend->getOpenCLRuntime(), &event);
    
    mOpenCLBackend->getOpenCLRuntime()->pushEvent({"Fuse", event});
#else
    if(mOpenCLBackend->getOpenCLRuntime()->isUseRecordQueue()){
        if(mOpenCLBackend->getOpenCLRuntime()->isDevideOpRecord())
            mOpenCLBackend->getOpenCLRuntime()->getRecordings()->emplace_back(mRecording);
#ifdef LOG_VERBOSE
        MNN_PRINT("end SoftmaxExecution onExecute !\n");
#endif
        return NO_ERROR;
    }
    run3DKernelDefault(mKernel, mGlobalWorkSize, mLocalWorkSize, mOpenCLBackend->getOpenCLRuntime());
#endif

#ifdef LOG_VERBOSE
    MNN_PRINT("end SoftmaxExecution onExecute !\n");
#endif

    return NO_ERROR;
}

class FuseCreator : public OpenCLBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        return new FuseExecution(inputs, backend, op);
    }
};
OpenCLCreatorRegister<FuseCreator> __Fuse_op(OpType_Extra, IMAGE);

} // namespace OpenCL
} // namespace MNN
