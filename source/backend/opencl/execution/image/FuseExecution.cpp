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
    : CommonExecution(backend, op) {
    mUnits.resize(1);
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    std::set<std::string> buildOptions;
    std::string kernelName;
    auto extra = op->main_as_Extra();
    auto source = reinterpret_cast<const char*>(extra->info()->data());
    auto name = extra->type()->c_str();
    mKernelName = extra->type()->str();
    mUnits[0].kernel = runtime->buildKernelFromSource(source, name, buildOptions);
    mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mUnits[0].kernel));
}

ErrorCode FuseExecution::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    Tensor *input  = inputs[0];
    Tensor *output = outputs[0];

    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);
    auto &unit  = mUnits[0];

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
        ret |= unit.kernel->get().setArg(idx++, openCLImage(input));
    }
    for (auto output : outputs) {
        ret |= unit.kernel->get().setArg(idx++, openCLImage(output));
    }
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[0]);
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[1]);
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[2]);
    MNN_CHECK_CL_SUCCESS(ret, "setArg FuseExecution");

    mLocalWorkSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), mKernelName, unit.kernel).first;
    mOpenCLBackend->recordKernel3d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
    unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
    unit.localWorkSize = {mLocalWorkSize[0], mLocalWorkSize[1], mLocalWorkSize[2]};
    return NO_ERROR;
}

class FuseCreator : public OpenCLBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        return new FuseExecution(inputs, backend, op);
    }
};
REGISTER_OPENCL_OP_CREATOR(FuseCreator, OpType_Extra, IMAGE);

} // namespace OpenCL
} // namespace MNN
