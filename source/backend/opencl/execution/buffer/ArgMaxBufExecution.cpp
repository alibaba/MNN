//
//  ArgMaxBufExecution.cpp
//  MNN
//
//  Created by MNN on 2023/08/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED
#include "backend/opencl/execution/buffer/ArgMaxBufExecution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"

namespace MNN {
namespace OpenCL {

ArgMaxBufExecution::ArgMaxBufExecution(const std::string &compute, Backend* backend, const int axis) : Execution(backend) {
    mBuildOptions.emplace(compute);
    mAxis = axis;
    // Do nothing
}
ErrorCode ArgMaxBufExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto openCLBackend = static_cast<OpenCLBackend*>(backend());
    auto runtime       = openCLBackend->getOpenCLRuntime();
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
    mGlobalWorkSize = {
        static_cast<uint32_t>(outputWidth),
        static_cast<uint32_t>(outputHeight),
        static_cast<uint32_t>(outputBatch * outputChannelBlocks)
    };
    
    if(batch * inputHeight * inputChannels == outside && 1 == inside && dim == inputWidth){
        mKernel = runtime->buildKernel("argmax_buf", "argmax_width_buf", mBuildOptions);
    }else if(batch * inputChannels == outside && inputWidth == inside && dim == inputHeight){
        mKernel = runtime->buildKernel("argmax_buf", "argmax_height_buf", mBuildOptions);
    }else if(batch == outside && inputWidth * inputHeight == inside && dim == inputChannels){
        if(output->buffer().dimensions == 1){
            mKernel = runtime->buildKernel("argmax_buf", "argmax_channel_dim1_buf", mBuildOptions);
        }else{
            mKernel = runtime->buildKernel("argmax_buf", "argmax_channel_buf", mBuildOptions);
        }
        mGlobalWorkSize[2] = static_cast<uint32_t>(outputBatch * outputChannels);
    }else if(1 == outside && inputWidth * inputHeight * inputChannels == inside && dim == batch){
        mKernel = runtime->buildKernel("argmax_buf", "argmax_batch_buf", mBuildOptions);
    }
    mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel));

    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= mKernel.setArg(idx++, mGlobalWorkSize[0]);
    ret |= mKernel.setArg(idx++, mGlobalWorkSize[1]);
    ret |= mKernel.setArg(idx++, mGlobalWorkSize[2]);
    ret |= mKernel.setArg(idx++, openCLBuffer(input));
    ret |= mKernel.setArg(idx++, openCLBuffer(output));
    ret |= mKernel.setArg(idx++, inputWidth);
    ret |= mKernel.setArg(idx++, inputHeight);
    ret |= mKernel.setArg(idx++, inputChannels);
    ret |= mKernel.setArg(idx++, batch);
    ret |= mKernel.setArg(idx++, inputChannelBlocks);
    ret |= mKernel.setArg(idx++, outputWidth);
    ret |= mKernel.setArg(idx++, outputHeight);
    ret |= mKernel.setArg(idx++, outputChannels);
    ret |= mKernel.setArg(idx++, outputChannelBlocks);
    MNN_CHECK_CL_SUCCESS(ret, "setArg ArgMaxBufExecution");

    std::string kernelName = "gargmax_buf";
    mLocalSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, openCLBackend->getOpenCLRuntime(), kernelName, mKernel).first;
    return NO_ERROR;
}

ErrorCode ArgMaxBufExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start ArgMaxBufExecution onExecute...");
#endif
    auto mOpenCLBackend = static_cast<OpenCLBackend*>(backend());
    
#ifdef ENABLE_OPENCL_TIME_PROFILER
    cl::Event event;
    run3DKernelDefault(mKernel, mGlobalWorkSize, mLocalSize,
                       mOpenCLBackend->getOpenCLRuntime(), &event);
    
    mOpenCLBackend->getOpenCLRuntime()->pushEvent({"ArgMax", event});
#else
    run3DKernelDefault(mKernel, mGlobalWorkSize, mLocalSize,
                       mOpenCLBackend->getOpenCLRuntime());
#endif

#ifdef LOG_VERBOSE
    MNN_PRINT("end ArgMaxBufExecution onExecute...");
#endif
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
            return new ArgMaxBufExecution("-DARGMAX", backend, axis);
        }else{
            return new ArgMaxBufExecution("", backend, axis);
        }
    }
};

OpenCLCreatorRegister<ArgMaxBufCreator> __ArgMaxBuf__(OpType_ArgMax, BUFFER);
OpenCLCreatorRegister<ArgMaxBufCreator> __ArgMinBuf__(OpType_ArgMin, BUFFER);
} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
