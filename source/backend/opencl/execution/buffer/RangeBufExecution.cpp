//
//  RangeBufExecution.cpp
//  MNN
//
//  Created by MNN on 2023/08/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED
#include "backend/opencl/execution/buffer/RangeBufExecution.hpp"

namespace MNN {
namespace OpenCL {

RangeBufExecution::RangeBufExecution(const std::string &compute, const MNN::Op *Op, Backend* backend) : CommonExecution(backend, Op) {
    mBuildOptions.emplace(compute);
    // Do nothing
}
ErrorCode RangeBufExecution::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    mUnits.resize(1);
    auto &unit = mUnits[0];
    auto openCLBackend = static_cast<OpenCLBackend*>(backend());
    auto runtime       = openCLBackend->getOpenCLRuntime();    
    std::vector<int> outputShape = tensorShapeFormat(outputs[0]);
    int totalSize = outputShape[0] * outputShape[1] * outputShape[2] * outputShape[3];
    mGlobalWorkSize = {
        static_cast<uint32_t>(UP_DIV(totalSize, 4)),
        static_cast<uint32_t>(1)
    };
    std::set<std::string> buildOption = mBuildOptions;
    if((totalSize % 4) != 0){
        buildOption.emplace("-DPACK_LEAVE");
    }
    unit.kernel = runtime->buildKernel("range_buf", "range_buf", buildOption, inputs[0], outputs[0]);
    mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));

    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[0]);
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[1]);
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(inputs[0]));
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(inputs[2]));
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(outputs[0]));
    ret |= unit.kernel->get().setArg(idx++, totalSize);
    MNN_CHECK_CL_SUCCESS(ret, "setArg RangeBufExecution");

    std::string kernelName = "range_buf";
    mLocalSize = localWS2DDefault(mGlobalWorkSize, mMaxWorkGroupSize, openCLBackend->getOpenCLRuntime(), kernelName, unit.kernel).first;
    openCLBackend->recordKernel2d(unit.kernel, mGlobalWorkSize, mLocalSize);
    unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1]};
    unit.localWorkSize = {mLocalSize[0], mLocalSize[1]};
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
                return new RangeBufExecution("-DUSE_INT", op, backend);
            case halide_type_float:
                return new RangeBufExecution("-DUSE_FLOAT", op, backend);
            default:
                return nullptr;
        }
    }
};

REGISTER_OPENCL_OP_CREATOR(RangeBufCreator, OpType_Range, BUFFER);

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
