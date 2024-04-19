//
//  SelectBufExecution.cpp
//  MNN
//
//  Created by MNN on 2023/08/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED
#include "backend/opencl/execution/buffer/SelectBufExecution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"

namespace MNN {
namespace OpenCL {

SelectBufExecution::SelectBufExecution(Backend* backend, const MNN::Op* Op) : CommonExecution(backend, Op) {
    // Do nothing
}
ErrorCode SelectBufExecution::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    mUnits.resize(1);
    auto &unit = mUnits[0];
    auto inSize1 = inputs[1]->elementSize();
    auto inSize2 = inputs[2]->elementSize();
    auto openCLBackend = static_cast<OpenCLBackend*>(backend());
    auto runtime       = openCLBackend->getOpenCLRuntime();
    std::set<std::string> buildOptions = mBuildOptions;
    if(inSize1 == 1)
        buildOptions.emplace("-DINSIZE1_EUQAL_1");
    if(inSize2 == 1)
        buildOptions.emplace("-DINSIZE2_EUQAL_1");
    unit.kernel = runtime->buildKernel("select_buf", "select_buf", buildOptions);
    mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));

    std::vector<int> outputShape = tensorShapeFormat(outputs[0]);

    int batch        = outputShape.at(0);
    int outputHeight = outputShape.at(1);
    int outputWidth  = outputShape.at(2);
    int channels     = outputShape.at(3);
    int channelBlocks = (channels + 3) / 4;
    int outSize = batch * channelBlocks * outputWidth * outputHeight * 4;

    mGlobalWorkSize = {
        static_cast<uint32_t>(outSize),
        1
    };

    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[0]);
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[1]);
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(inputs[0]));
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(inputs[1]));
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(inputs[2]));
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(outputs[0]));
    MNN_CHECK_CL_SUCCESS(ret, "setArg SelectBufExecution");

    std::string kernelName = "select_buf";
    mLocalSize = localWS2DDefault(mGlobalWorkSize, mMaxWorkGroupSize, openCLBackend->getOpenCLRuntime(), kernelName, unit.kernel).first;
    openCLBackend->recordKernel2d(unit.kernel, mGlobalWorkSize, mLocalSize);
    unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1]};
    unit.localWorkSize = {mLocalSize[0], mLocalSize[1]};
    return NO_ERROR;
}

class SelectBufCreator : public OpenCLBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        for (int i = 0; i < inputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(inputs[i], false);
        }
        for (int i = 0; i < outputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(outputs[i], false);
        }
        return new SelectBufExecution(backend, op);
    }
};

REGISTER_OPENCL_OP_CREATOR(SelectBufCreator, OpType_Select, BUFFER);

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
