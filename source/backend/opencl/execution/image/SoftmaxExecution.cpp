//
//  SoftmaxExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/image/SoftmaxExecution.hpp"
#include "core/Macro.h"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"

namespace MNN {
namespace OpenCL {

SoftmaxExecution::SoftmaxExecution(const std::vector<Tensor *> &inputs, int axis, Backend *backend)
    : Execution(backend) {
    mAxis          = axis;
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
}

bool SoftmaxExecution::buildSoftmaxKernel(int localSize) {
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    if (mKernel.get() == nullptr) {
        std::set<std::string> buildOptions;
        buildOptions.emplace("-DSOFTMAX_LOCAL_SIZE=" + std::to_string(localSize));
        std::string kernelName;
        if (mAxis == 1) {
            mKernel           = runtime->buildKernel("softmax", "softmax_channel", buildOptions);
        } else if (mAxis == 2) {
            mKernel           = runtime->buildKernel("softmax", "softmax_height", buildOptions);
        } else {
            MNN_ASSERT(mAxis == 3);
            mKernel           = runtime->buildKernel("softmax", "softmax_width", buildOptions);
        }
        mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel));
    }
    return true;
}

int SoftmaxExecution::getLocalSize(int size, int maxGroupSize){
    int local_size = 1;
    while(local_size * 2 <= maxGroupSize && local_size * 2 <= size){
        local_size *= 2;
    }
    return local_size;
}

ErrorCode SoftmaxExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    startRecord(mOpenCLBackend->getOpenCLRuntime(), mRecording);
    Tensor *input  = inputs[0];
    Tensor *output = outputs[0];
    
    const auto dims = input->buffer().dimensions;
    int inside  = 1;
    int outside = 1;
    int channel = 1;
    for (int i = 0; i < mAxis; ++i) {
        outside *= input->length(i);
    }
    channel = input->length(mAxis);
    for (int i = mAxis + 1; i < dims; ++i) {
        inside *= input->length(i);
    }
    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);

    const int inputBatch    = inputShape.at(0);
    const int inputHeight   = inputShape.at(1);
    const int inputWidth    = inputShape.at(2);
    const int inputChannels = inputShape.at(3);
    
    const int outputBatch    = outputShape.at(0);
    const int outputHeight   = outputShape.at(1);
    const int outputWidth    = outputShape.at(2);
    const int outputChannels = outputShape.at(3);

    const int channelBlocks  = UP_DIV(outputChannels, 4);
    const int remainChannels = channelBlocks * 4 - outputChannels;
    auto MaxWorkItems = mOpenCLBackend->getOpenCLRuntime()->getMaxWorkItemSizes();
    int localSize = getLocalSize(channel, MaxWorkItems[0]);
    if(localSize < 4){
        localSize = 1;
    }
    if(inputBatch == outside && channel == inputChannels && inside == inputWidth * inputHeight){
        mAxis = 1;
        localSize = getLocalSize(channelBlocks, MaxWorkItems[0]);
    }else if(inputBatch * inputChannels == outside && channel == inputHeight && inside == inputWidth){
        mAxis = 2;
    }else if(inputBatch * inputChannels * inputHeight == outside && channel == inputWidth && inside == 1){
        mAxis = 3;
    }
    buildSoftmaxKernel(localSize);
    
    cl_int ret = CL_SUCCESS;
    int shape[] = {outputBatch, channelBlocks, outputHeight, outputWidth};
    if (mAxis == 1) {
        mGlobalWorkSize = {(uint32_t)(localSize), (uint32_t)outputWidth, (uint32_t)outputHeight * outputBatch};

    } else if (mAxis == 2){
        mGlobalWorkSize = {(uint32_t)(localSize), (uint32_t)channelBlocks*outputWidth, (uint32_t)outputBatch};
    } else {
        MNN_ASSERT(mAxis == 3);
        mGlobalWorkSize = {(uint32_t)(localSize), (uint32_t)channelBlocks, (uint32_t)outputBatch*outputHeight};
    }
    mLocalWorkSize = {(uint32_t)(localSize), 1, 1};
    
    uint32_t idx    = 0;
    ret |= mKernel.setArg(idx++, mGlobalWorkSize[0]);
    ret |= mKernel.setArg(idx++, mGlobalWorkSize[1]);
    ret |= mKernel.setArg(idx++, mGlobalWorkSize[2]);

    ret |= mKernel.setArg(idx++, openCLImage(input));
    ret |= mKernel.setArg(idx++, openCLImage(output));
    ret |= mKernel.setArg(idx++, remainChannels);
    ret |= mKernel.setArg(idx++, shape);
    MNN_CHECK_CL_SUCCESS(ret, "setArg SoftmaxExecution");
    if(localSize == 1){
        mLocalWorkSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), "softmax", mKernel).first;
    }
    recordKernel3d(mKernel, mGlobalWorkSize, mLocalWorkSize, mOpenCLBackend->getOpenCLRuntime());
    endRecord(mOpenCLBackend->getOpenCLRuntime(), mRecording);
    return NO_ERROR;
}

ErrorCode SoftmaxExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start SoftmaxExecution onExecute !\n");
#endif
    
#ifdef ENABLE_OPENCL_TIME_PROFILER
    cl::Event event;
    run3DKernelDefault(mKernel, mGlobalWorkSize, mLocalWorkSize,
                       mOpenCLBackend->getOpenCLRuntime(), &event);
    
    mOpenCLBackend->getOpenCLRuntime()->pushEvent({"Softmax", event});
#else
    if(mOpenCLBackend->getOpenCLRuntime()->isUseRecordQueue()){
        if(mOpenCLBackend->getOpenCLRuntime()->isDevideOpRecord())
            mOpenCLBackend->getOpenCLRuntime()->getRecordings()->emplace_back(mRecording);
#ifdef LOG_VERBOSE
        MNN_PRINT("End SoftmaxExecution onExecute... \n");
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

class SoftmaxCreator : public OpenCLBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        auto dimType = inputs[0]->getDimensionType();
        if (dimType == Tensor::TENSORFLOW && inputs[0]->dimensions() == 4) {
            int index[4] = {0, 2, 3, 1};
            auto axis = op->main_as_Axis()->axis();
            if (axis < 0) {
                axis = inputs[0]->dimensions() + axis;
            }

            axis = index[axis];
            //1 : channel //2 : height
            if (1 == axis || 2 == axis || 3 == axis) {
                return new SoftmaxExecution(inputs, axis, backend);
            }
            return nullptr;
        } else {
            auto axis = op->main_as_Axis()->axis();
            if (axis < 0) {
                axis = inputs[0]->dimensions() + axis;
            }

            if (1 == axis || 2 == axis || 3 == axis) {
                return new SoftmaxExecution(inputs, axis, backend);
            }
            return nullptr;
        }
    }
};
OpenCLCreatorRegister<SoftmaxCreator> __Softmax_op(OpType_Softmax, IMAGE);

} // namespace OpenCL
} // namespace MNN
