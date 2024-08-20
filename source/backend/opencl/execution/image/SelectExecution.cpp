//
//  SelectExecution.cpp
//  MNN
//
//  Created by MNN on 2023/12/1.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/image/SelectExecution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"

namespace MNN {
namespace OpenCL {

SelectExecution::SelectExecution(const MNN::Op *op, Backend* backend) : CommonExecution(backend, op) {
    // Do nothing
}
ErrorCode SelectExecution::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
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
    unit.kernel = runtime->buildKernel("select", "select_img", buildOptions);
    mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));

    std::vector<int> outputShape = tensorShapeFormat(outputs[0]);

    int batch        = outputShape.at(0);
    int outputHeight = outputShape.at(1);
    int outputWidth  = outputShape.at(2);
    int channels     = outputShape.at(3);
    int channelBlocks = (channels + 3) / 4;

    mGlobalWorkSize = {
        static_cast<uint32_t>(channelBlocks * outputWidth),
        static_cast<uint32_t>(batch * outputHeight)
    };

    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[0]);
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[1]);
    ret |= unit.kernel->get().setArg(idx++, openCLImage(inputs[0]));
    ret |= unit.kernel->get().setArg(idx++, openCLImage(inputs[1]));
    ret |= unit.kernel->get().setArg(idx++, openCLImage(inputs[2]));
    ret |= unit.kernel->get().setArg(idx++, openCLImage(outputs[0]));
    MNN_CHECK_CL_SUCCESS(ret, "setArg SelectExecution");

    std::string kernelName = "select_img";
    mLocalSize = localWS2DDefault(mGlobalWorkSize, mMaxWorkGroupSize, openCLBackend->getOpenCLRuntime(), kernelName, unit.kernel).first;
    openCLBackend->recordKernel2d(unit.kernel, mGlobalWorkSize, mLocalSize);
    unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1]};
    unit.localWorkSize = {mLocalSize[0], mLocalSize[1]};
    return NO_ERROR;
}

class SelectCreator : public OpenCLBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new SelectExecution(op, backend);
    }
};

REGISTER_OPENCL_OP_CREATOR(SelectCreator, OpType_Select, IMAGE);
} // namespace OpenCL
} // namespace MNN
