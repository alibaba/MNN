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
            mReductType = 0;
            break;
        case ReductionType_MAXIMUM:
            mReductType = 1;
            break;
        case ReductionType_MINIMUM:
            mReductType = 2;
            break;
        case ReductionType_PROD:
            mReductType = 3;
            break;
        case ReductionType_SUM:
            mReductType = 4;
            break;
        default:
            MNN_ASSERT(false);
            break;
    }
    auto kernel = mOpenCLBackend->getOpenCLRuntime()->buildKernel("reduction_buf", "reduct_width_buf", {"-DOPERATE(a,b)=(a+b)","-DVALUE=0","-DLOCAL_SIZE=512"}, inputs[0], outputs[0]);
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
    int local_size = 0;
    
    if(dim >= 16){
        mUseLocal = true;
    }

    std::vector<int> inputShape = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);

    int batch = inputShape.at(0);
    int inputHeight = inputShape.at(1);
    int inputWidth  = inputShape.at(2);
    int inputChannels = inputShape.at(3);
    int inputChannelBlocks = (inputChannels + 3) / 4;
    int outputBatch = outputShape.at(0);
    int outputHeight = outputShape.at(1);
    int outputWidth  = outputShape.at(2);
    int outputChannels = outputShape.at(3);
    int outputChannelBlocks = (outputChannels + 3) / 4;

    std::set<std::string> buildOption;
    switch (mReductType) {
        case 0:
            buildOption.emplace("-DOPERATE(a,b)=(a+b)");
            buildOption.emplace("-DGET_AVG");
            buildOption.emplace("-DVALUE=0");
            break;
        case 1:
            buildOption.emplace("-DOPERATE(a,b)=max(a,b)");
            buildOption.emplace("-DVALUE=-FLT_MAX");
            break;
        case 2:
            buildOption.emplace("-DOPERATE(a,b)=min(a,b)");
            buildOption.emplace("-DVALUE=FLT_MAX");
            break;
        case 3:
            buildOption.emplace("-DOPERATE(a,b)=(a*b)");
            buildOption.emplace("-DVALUE=1");
            break;
        case 4:
            buildOption.emplace("-DOPERATE(a,b)=(a+b)");
            buildOption.emplace("-DVALUE=0");
            break;
        default:
            MNN_ASSERT(false);
            break;
    }
    
    mGlobalWorkSize = {
        static_cast<uint32_t>(outputWidth),
        static_cast<uint32_t>(outputHeight),
        static_cast<uint32_t>(outputBatch * outputChannelBlocks)
    };
    
    if(mUseLocal){
        if(batch * inputHeight * inputChannels == outside && 1 == inside && dim == inputWidth){
            local_size = getLocalSize(inputWidth, MaxLocalSize);
            buildOption.emplace("-DLOCAL_SIZE=" + std::to_string(local_size));
            unit.kernel = runtime->buildKernel("reduction_buf", "reduct_width_buf", buildOption, input, output);
        }else if(batch * inputChannels == outside && inputWidth == inside && dim == inputHeight){
            local_size = getLocalSize(inputHeight, MaxLocalSize);
            buildOption.emplace("-DLOCAL_SIZE=" + std::to_string(local_size));
            unit.kernel = runtime->buildKernel("reduction_buf", "reduct_height_buf", buildOption, input, output);
        }else if(batch == outside && inputWidth * inputHeight == inside && dim == inputChannels){
            local_size = getLocalSize(inputChannelBlocks - 1, MaxLocalSize);
            buildOption.emplace("-DLOCAL_SIZE=" + std::to_string(local_size));
            if(output->buffer().dimensions == 1){
                unit.kernel = runtime->buildKernel("reduction_buf", "reduct_channel_dim1_buf", buildOption, input, output);
            }else{
                unit.kernel = runtime->buildKernel("reduction_buf", "reduct_channel_buf", buildOption, input, output);
            }
            mGlobalWorkSize[2] = static_cast<uint32_t>(outputBatch * outputChannels);
        }else if(1 == outside && inputWidth * inputHeight * inputChannels == inside && dim == batch){
            local_size = getLocalSize(batch, MaxLocalSize);
            buildOption.emplace("-DLOCAL_SIZE=" + std::to_string(local_size));
            unit.kernel = runtime->buildKernel("reduction_buf", "reduct_batch_buf", buildOption, input, output);
        }
        mGlobalWorkSize[0] *= local_size;
    }else{
        buildOption.emplace("-DLOCAL_SIZE=0");
        if(batch * inputHeight * inputChannels == outside && 1 == inside && dim == inputWidth){
            unit.kernel = runtime->buildKernel("reduction_buf", "reduct_width_buf", buildOption, input, output);
        }else if(batch * inputChannels == outside && inputWidth == inside && dim == inputHeight){
            unit.kernel = runtime->buildKernel("reduction_buf", "reduct_height_buf", buildOption, input, output);
        }else if(batch == outside && inputWidth * inputHeight == inside && dim == inputChannels){
            if(output->buffer().dimensions == 1){
                unit.kernel = runtime->buildKernel("reduction_buf", "reduct_channel_dim1_buf", buildOption, input, output);
            }else{
                unit.kernel = runtime->buildKernel("reduction_buf", "reduct_channel_buf", buildOption, input, output);
            }
            mGlobalWorkSize[2] = static_cast<uint32_t>(outputBatch * outputChannels);
        }else if(1 == outside && inputWidth * inputHeight * inputChannels == inside && dim == batch){
            unit.kernel = runtime->buildKernel("reduction_buf", "reduct_batch_buf", buildOption, input, output);
        }
    }
    //printf("reduce axis:%d , %d %d %d %d, useLocal:%d\n", mAxis[0], inputShape[0], inputShape[1], inputShape[2], inputShape[3], mUseLocal);

    mUnits.resize(1);
    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[0]);
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[1]);
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[2]);
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(input));
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(output));
    ret |= unit.kernel->get().setArg(idx++, inputWidth);
    ret |= unit.kernel->get().setArg(idx++, inputHeight);
    ret |= unit.kernel->get().setArg(idx++, inputChannels);
    ret |= unit.kernel->get().setArg(idx++, batch);
    ret |= unit.kernel->get().setArg(idx++, inputChannelBlocks);
    ret |= unit.kernel->get().setArg(idx++, outputWidth);
    ret |= unit.kernel->get().setArg(idx++, outputHeight);
    ret |= unit.kernel->get().setArg(idx++, outputChannels);
    ret |= unit.kernel->get().setArg(idx++, outputChannelBlocks);
    MNN_CHECK_CL_SUCCESS(ret, "setArg ReductionBufExecution");

    if(mUseLocal){
        mLocalWorkSize = {static_cast<uint32_t>(local_size), 1, 1};
    }else{
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
