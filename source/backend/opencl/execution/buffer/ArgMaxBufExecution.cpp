//
//  ArgMaxBufExecution.cpp
//  MNN
//
//  Created by MNN on 2023/08/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED
#include "backend/opencl/execution/buffer/ArgMaxBufExecution.hpp"

namespace MNN {
namespace OpenCL {

ArgMaxBufExecution::ArgMaxBufExecution(const std::string &compute, const MNN::Op* op, Backend* backend, const int axis) : CommonExecution(backend, op) {
    mBuildOptions.emplace(compute);
    mAxis = axis;
    // Do nothing
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    std::set<std::string> buildOptions = mBuildOptions;
    buildOptions.emplace("-DARGMAX_LOCAL_SIZE=512");
    auto kernel = mOpenCLBackend->getOpenCLRuntime()->buildKernel("argmax_buf", "argmax_channel_buf", buildOptions);
    mMaxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(kernel));
}

int ArgMaxBufExecution::getLocalSize(int size, int maxGroupSize){
    int local_size = 1;
    while(local_size * 2 <= maxGroupSize && local_size * 2 <= size){
        local_size *= 2;
    }
    return local_size;
}

ErrorCode ArgMaxBufExecution::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    mUnits.resize(1);
    auto &unit = mUnits[0];
    auto runtime       = mOpenCLBackend->getOpenCLRuntime();
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
    
    int localSize = getLocalSize(dim, MaxLocalSize);
    if(localSize < 4){
        localSize = 1;
    }
    std::set<std::string> buildOptions = mBuildOptions;
    buildOptions.emplace("-DARGMAX_LOCAL_SIZE=" + std::to_string(localSize));
    std::string kernelName;
    if(batch * inputHeight * inputChannels == outside && 1 == inside && dim == inputWidth){
        kernelName = "argmax_width_buf";
        unit.kernel = runtime->buildKernel("argmax_buf", kernelName, buildOptions);
        mGlobalWorkSize = {static_cast<uint32_t>(localSize), static_cast<uint32_t>(outputHeight), static_cast<uint32_t>(outputBatch * outputChannelBlocks)};
    }else if(batch * inputChannels == outside && inputWidth == inside && dim == inputHeight){
        kernelName = "argmax_height_buf";
        unit.kernel = runtime->buildKernel("argmax_buf", kernelName, buildOptions);
        mGlobalWorkSize = {static_cast<uint32_t>(localSize), static_cast<uint32_t>(outputWidth), static_cast<uint32_t>(outputBatch * outputChannelBlocks)};
    }else if(batch == outside && inputWidth * inputHeight == inside && dim == inputChannels){
        if(output->buffer().dimensions == 1){
            buildOptions.emplace("-DARGMAX_CHANNEL_DIM1");
        }
        kernelName = "argmax_channel_buf";
        unit.kernel = runtime->buildKernel("argmax_buf", kernelName, buildOptions);
        mGlobalWorkSize = {static_cast<uint32_t>(localSize), static_cast<uint32_t>(outputWidth * outputHeight), static_cast<uint32_t>(outputBatch * outputChannels)};
    }else if(1 == outside && inputWidth * inputHeight * inputChannels == inside && dim == batch){
        kernelName = "argmax_batch_buf";
        unit.kernel = runtime->buildKernel("argmax_buf", kernelName, buildOptions);
        mGlobalWorkSize = {static_cast<uint32_t>(localSize), static_cast<uint32_t>(outputWidth * outputHeight), static_cast<uint32_t>(outputChannelBlocks)};
    }
    mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));
    mLocalSize = {(uint32_t)(localSize), 1, 1};

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
    MNN_CHECK_CL_SUCCESS(ret, "setArg ArgMaxBufExecution");

    if(localSize == 1){
        mLocalSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), kernelName, unit.kernel).first;
    }
    mOpenCLBackend->recordKernel3d(unit.kernel, mGlobalWorkSize, mLocalSize);
    unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
    unit.localWorkSize = {mLocalSize[0], mLocalSize[1], mLocalSize[2]};
    return NO_ERROR;
}

class ArgMaxBufCreator : public OpenCLBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        for (int i = 0; i < inputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(inputs[i], false);
        }
        for (int i = 0; i < outputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(outputs[i], false);
        }
        auto inputDimensionFromat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        if(inputDimensionFromat == MNN_DATA_FORMAT_NC4HW4){
            return nullptr;
        }
        int axis = op->main_as_ArgMax()->axis();
        if (op->type() == OpType_ArgMax) {
            return new ArgMaxBufExecution("-DARGMAX", op, backend, axis);
        }else{
            return new ArgMaxBufExecution("", op, backend, axis);
        }
    }
};

REGISTER_OPENCL_OP_CREATOR(ArgMaxBufCreator, OpType_ArgMax, BUFFER);
REGISTER_OPENCL_OP_CREATOR(ArgMaxBufCreator, OpType_ArgMin, BUFFER);

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
