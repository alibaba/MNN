//
//  ReductionBufExecution.cpp
//  MNN
//
//  Created by MNN on 2019/10/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED

#include "backend/opencl/execution/buffer/ReductionBufExecution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace OpenCL {

ReductionBufExecution::ReductionBufExecution(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, const MNN::Op* op, Backend* backend) : CommonExecution(backend, op) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start ReductionBufExecution init !\n");
#endif
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    mAxis = op->main_as_ReductionParam()->dim()->data()[0];
    switch (op->main_as_ReductionParam()->operation()) {
        case ReductionType_MEAN:
            mBuildOptions.emplace("-DOPERATE(a,b)=(a+b)");
            mBuildOptions.emplace("-DGET_AVG");
            mBuildOptions.emplace("-DVALUE=0");
            break;
        case ReductionType_MAXIMUM:
            mBuildOptions.emplace("-DOPERATE(a,b)=max(a,b)");
            mBuildOptions.emplace("-DVALUE=-FLT_MAX");
            break;
        case ReductionType_MINIMUM:
            mBuildOptions.emplace("-DOPERATE(a,b)=min(a,b)");
            mBuildOptions.emplace("-DVALUE=FLT_MAX");
            break;
        case ReductionType_PROD:
            mBuildOptions.emplace("-DOPERATE(a,b)=(a*b)");
            mBuildOptions.emplace("-DVALUE=1");
            break;
        case ReductionType_SUM:
            mBuildOptions.emplace("-DOPERATE(a,b)=(a+b)");
            mBuildOptions.emplace("-DVALUE=0");
            break;
        default:
            MNN_ASSERT(false);
            break;
    }
    auto kernel = mOpenCLBackend->getOpenCLRuntime()->buildKernel("reduction_buf", "reduct_buf", {"-DOPERATE(a,b)=(a+b)","-DVALUE=0","-DLOCAL_SIZE=512"}, inputs[0], outputs[0]);
    mMaxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(kernel));
#ifdef LOG_VERBOSE
    MNN_PRINT("end ReductionBufExecution init !\n");
#endif
}

int ReductionBufExecution::getLocalSize(int size, int maxGroupSize){
    int local_size = 1;
    while(local_size * 2 <= maxGroupSize && local_size * 2 <= size){
        local_size *= 2;
    }
    return local_size;
}

ErrorCode ReductionBufExecution::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mUnits.resize(1);
    auto &unit = mUnits[0];
    auto openCLBackend = static_cast<OpenCLBackend*>(backend());
    auto runtime       = openCLBackend->getOpenCLRuntime();
    auto MaxLocalSize = std::min(runtime->getMaxWorkItemSizes()[0], mMaxWorkGroupSize);
    auto input = inputs[0];
    auto output = outputs[0];
    if(mAxis < 0){
        mAxis = input->dimensions() + mAxis;
    }
    int inside = 1;
    int outside = 1;
    for(int i = 0; i < mAxis; ++i){
        outside *= input->length(i);
    }
    for(int i = mAxis + 1; i < input->dimensions(); ++i){
        inside *= input->length(i);
    }
    int dim = input->length(mAxis);
    
    int localSize = getLocalSize(dim, MaxLocalSize);
    if(localSize < 4){
        localSize = 1;
    }
    
    std::set<std::string> buildOptions = mBuildOptions;
    buildOptions.emplace("-DREDUCT_LOCAL_SIZE=" + std::to_string(localSize));
    std::string kernelName;
    if(inside % 4 == 0){
        unit.kernel = runtime->buildKernel("reduction_buf", "reduct_v4_buf", buildOptions, input, output);
        mGlobalWorkSize = {static_cast<uint32_t>(localSize), static_cast<uint32_t>(UP_DIV(inside, 4)), static_cast<uint32_t>(outside)};
    }else {
        unit.kernel = runtime->buildKernel("reduction_buf", "reduct_buf", buildOptions, input, output);
        mGlobalWorkSize = {static_cast<uint32_t>(localSize), static_cast<uint32_t>(inside), static_cast<uint32_t>(outside)};
    }
    mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));
    mLocalWorkSize = {(uint32_t)(localSize), 1, 1};

    mUnits.resize(1);
    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[0]);
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[1]);
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[2]);
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(input));
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(output));
    ret |= unit.kernel->get().setArg(idx++, inside);
    ret |= unit.kernel->get().setArg(idx++, outside);
    ret |= unit.kernel->get().setArg(idx++, dim);
    MNN_CHECK_CL_SUCCESS(ret, "setArg ReductionBufExecution");

    if(localSize == 1){
        mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));
        std::string kernelName = "reduct_buf";
        mLocalWorkSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, openCLBackend->getOpenCLRuntime(), kernelName, unit.kernel).first;
    }
    openCLBackend->recordKernel3d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
    unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
    unit.localWorkSize = {mLocalWorkSize[0], mLocalWorkSize[1], mLocalWorkSize[2]};
    return NO_ERROR;
}

class ReductionBufCreator : public OpenCLBackend::Creator {
public:
    virtual ~ReductionBufCreator() = default;
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                 const MNN::Op *op, Backend *backend) const override {
        for (int i = 0; i < inputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(inputs[i], false);
        }
        for (int i = 0; i < outputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(outputs[i], false);
        }
        
        auto openCLBackend = static_cast<OpenCLBackend *>(backend);
        auto reduct = op->main_as_ReductionParam();
        if (nullptr == reduct->dim()) {
            return NULL;
        }
        if(reduct->dim()->size() != 1) {
            return NULL;
        }
        switch (op->main_as_ReductionParam()->operation()) {
            case ReductionType_MEAN:
                break;
            case ReductionType_MAXIMUM:
                break;
            case ReductionType_MINIMUM:
                break;
            case ReductionType_PROD:
                break;
            case ReductionType_SUM:
                break;
            default:
                return NULL;
                break;
        }
        return new ReductionBufExecution(inputs, outputs, op, backend);
    }
};

REGISTER_OPENCL_OP_CREATOR(ReductionBufCreator, OpType_Reduction, BUFFER);

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
