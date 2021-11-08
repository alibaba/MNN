//
//  SoftmaxBufExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED

#include "backend/opencl/execution/buffer/SoftmaxBufExecution.hpp"
#include "core/Macro.h"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"

namespace MNN {
namespace OpenCL {

SoftmaxBufExecution::SoftmaxBufExecution(const std::vector<Tensor *> &inputs, int axis, Backend *backend)
    : Execution(backend) {
    mAxis          = axis;
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    buildSoftmaxKernel();
}

bool SoftmaxBufExecution::buildSoftmaxKernel() {
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    if (mKernel.get() == nullptr) {
        std::set<std::string> buildOptions;
        std::string kernelName;
        if (mAxis == 1) {
            mKernel           = runtime->buildKernel("softmax_buf", "softmax_channel", buildOptions);
        } else if (mAxis == 2) {
            mKernel           = runtime->buildKernel("softmax_buf", "softmax_height", buildOptions);
        } else {
            MNN_ASSERT(mAxis == 3);
            mKernel           = runtime->buildKernel("softmax_buf", "softmax_width", buildOptions);
        }
        mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel));
    }
    return true;
}

ErrorCode SoftmaxBufExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
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
    if (mAxis == 1) {
        mGlobalWorkSize = {static_cast<uint32_t>(channelBlocks), static_cast<uint32_t>(outputWidth),
            static_cast<uint32_t>(outputHeight * outputBatch)};
        int shape[] = {outputBatch, channelBlocks, outputHeight, outputWidth};

        uint32_t idx    = 0;
        mKernel.setArg(idx++, mGlobalWorkSize[0]);
        mKernel.setArg(idx++, mGlobalWorkSize[1]);
        mKernel.setArg(idx++, mGlobalWorkSize[2]);

        mKernel.setArg(idx++, openCLBuffer(input));
        mKernel.setArg(idx++, openCLBuffer(output));
        mKernel.setArg(idx++, static_cast<int>(outputChannels));
        mKernel.setArg(idx++, remainChannels);
        mKernel.setArg(idx++, shape);
        
        std::string kernelName = "softmax_buf_channel";
        mLocalWorkSize =
        localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), kernelName, mKernel).first;
    } else if (mAxis == 2){
        mGlobalWorkSize = {(uint32_t)channelBlocks*outputWidth, (uint32_t)outputBatch, 1};
        int shape[] = {outputBatch, channelBlocks, outputHeight, outputWidth};
        mKernel.setArg(0, openCLBuffer(input));
        mKernel.setArg(1, openCLBuffer(output));
        mKernel.setArg(2, shape);
        
        std::string kernelName = "softmax_buf_height";
        mLocalWorkSize =
        localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), kernelName, mKernel).first;
    } else {
        MNN_ASSERT(mAxis == 3);
        mGlobalWorkSize = {(uint32_t)channelBlocks, (uint32_t)outputBatch*outputHeight, 1};
        int shape[] = {outputBatch, channelBlocks, outputHeight, outputWidth};
        mKernel.setArg(0, openCLBuffer(input));
        mKernel.setArg(1, openCLBuffer(output));
        mKernel.setArg(2, shape);
        
        std::string kernelName = "softmax_buf_width";
        mLocalWorkSize =
        localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), kernelName, mKernel).first;
    }

    return NO_ERROR;
}

ErrorCode SoftmaxBufExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start SoftmaxBufExecution onExecute !\n");
#endif
    
#ifdef ENABLE_OPENCL_TIME_PROFILER
    cl::Event event;
    run3DKernelDefault(mKernel, mGlobalWorkSize, mLocalWorkSize,
                       mOpenCLBackend->getOpenCLRuntime(), &event);
    
    int costTime = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
    MNN_PRINT("kernel cost:%d    us Softmax\n",costTime);
#else
    run3DKernelDefault(mKernel, mGlobalWorkSize, mLocalWorkSize, mOpenCLBackend->getOpenCLRuntime());
#endif

#ifdef LOG_VERBOSE
    MNN_PRINT("end SoftmaxBufExecution onExecute !\n");
#endif

    return NO_ERROR;
}

class SoftmaxBufCreator : public OpenCLBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        if(inputs[0]->dimensions() == 3 || outputs[0]->dimensions() == 3){
            MNN_PRINT("softmax not support dimensions == 3 \n");
            return nullptr;
        }
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
                return new SoftmaxBufExecution(inputs, axis, backend);
            }
            return nullptr;
        } else {
            auto axis = op->main_as_Axis()->axis();
            if (axis < 0) {
                axis = inputs[0]->dimensions() + axis;
            }

            if (1 == axis || 2 == axis || 3 == axis) {
                return new SoftmaxBufExecution(inputs, axis, backend);
            }
            return nullptr;
        }
    }
};
OpenCLCreatorRegister<SoftmaxBufCreator> __SoftmaxBuf_op(OpType_Softmax, BUFFER);

} // namespace OpenCL
} // namespace MNN
#endif/* MNN_OPENCL_BUFFER_CLOSED */

