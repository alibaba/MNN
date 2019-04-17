//
//  SoftmaxExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "execution/SoftmaxExecution.hpp"
#include <Macro.h>

namespace MNN {
namespace OpenCL {

SoftmaxExecution::SoftmaxExecution(const std::vector<Tensor *> &inputs, int axis, Backend *backend)
    : Execution(backend) {
    mAxis          = axis;
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    buildSoftmaxKernel();
}

std::vector<uint32_t> SoftmaxExecution::softmaxLocalWS(const std::vector<uint32_t> &gws,
                                                       const uint32_t maxWorkGroupSize) {
    std::vector<uint32_t> lws(4, 0);
    GpuType gpuType             = mOpenCLBackend->getOpenCLRuntime()->getGpuType();
    uint32_t deviceComputeUnits = mOpenCLBackend->getOpenCLRuntime()->deviceComputeUnits();
    if (gpuType == GpuType::ADRENO) {
        int coreNum   = deviceComputeUnits;
        int remain    = gws[0] % coreNum;
        int groupSize = gws[0] / coreNum;
        if (remain == 0) {
            lws[0] = groupSize;
        } else {
            while (groupSize) {
                int remain = gws[0] % groupSize;
                if (remain == 0 && groupSize <= maxWorkGroupSize) {
                    lws[0] = groupSize;
                    break;
                }
                groupSize--;
            }
        }
        lws[0] = std::max<uint32_t>(std::min<uint32_t>(maxWorkGroupSize, lws[0]), 1);

        remain    = gws[1] % coreNum;
        groupSize = gws[1] / coreNum;
        if (remain == 0) {
            lws[1] = groupSize;
        } else {
            while (groupSize) {
                int remain = gws[1] % groupSize;
                if (remain == 0) {
                    lws[1] = groupSize;
                    break;
                }
                groupSize--;
            }
        }
        lws[1] = std::max<uint32_t>(std::min<uint32_t>(maxWorkGroupSize / lws[0], lws[1]), 1);

        remain    = gws[2] % coreNum;
        groupSize = gws[2] / coreNum;
        if (remain == 0) {
            lws[2] = groupSize;
        } else {
            while (groupSize) {
                int remain = gws[2] % groupSize;
                if (remain == 0) {
                    lws[2] = groupSize;
                    break;
                }
                groupSize--;
            }
        }

        lws[2] = std::max<uint32_t>(std::min<uint32_t>(maxWorkGroupSize / (lws[0] * lws[1]), lws[2]), 1);
    } else {
        lws[0] = deviceComputeUnits * 2;
        lws[1] = 4;
        lws[2] = 1;
    }
    return lws;
}

bool SoftmaxExecution::buildSoftmaxKernel() {
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    if (mKernel.get() == nullptr) {
        std::set<std::string> buildOptions;
        std::string kernelName;
        if (mAxis == 1) {
            mKernel           = runtime->buildKernel("softmax", "softmax_channel", buildOptions);
        } else {
            MNN_ASSERT(2 == mAxis);
            mKernel           = runtime->buildKernel("softmax_common", "softmax_height", buildOptions);
        }
        mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel));
    }
    return true;
}

ErrorCode SoftmaxExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
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
    if (1 == mAxis) {
        mGlobalWorkSize = {static_cast<uint32_t>(channelBlocks), static_cast<uint32_t>(outputWidth),
            static_cast<uint32_t>(outputHeight * outputBatch)};
        uint32_t idx    = 0;
        mKernel.setArg(idx++, mGlobalWorkSize[0]);
        mKernel.setArg(idx++, mGlobalWorkSize[1]);
        mKernel.setArg(idx++, mGlobalWorkSize[2]);
        
        mKernel.setArg(idx++, openCLImage(input));
        mKernel.setArg(idx++, openCLImage(output));
        mKernel.setArg(idx++, static_cast<int>(outputChannels));
        mKernel.setArg(idx++, remainChannels);
        mLocalWorkSize = softmaxLocalWS(mGlobalWorkSize, mMaxWorkGroupSize);
    } else {
        MNN_ASSERT(2 == mAxis);
        //FUNC_PRINT(mMaxWorkGroupSize);
        if (mMaxWorkGroupSize > 256) {
            mLocalWorkSize = {16, 16, 1};
        } else {
            mLocalWorkSize = {8, 8, 1};
        }
        mGlobalWorkSize = {(uint32_t)channelBlocks*outputWidth, (uint32_t)outputBatch, 1};
        int shape[] = {outputBatch, channelBlocks, outputHeight, outputWidth};
        mKernel.setArg(0, openCLImage(input));
        mKernel.setArg(1, openCLImage(output));
        mKernel.setArg(2, shape);
    }
    
    return NO_ERROR;
}

ErrorCode SoftmaxExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start SoftmaxExecution onExecute !\n");
#endif
    run3DKernelDefault(mKernel, mGlobalWorkSize, mLocalWorkSize, mOpenCLBackend->getOpenCLRuntime());

#ifdef LOG_VERBOSE
    MNN_PRINT("end SoftmaxExecution onExecute !\n");
#endif

    return NO_ERROR;
}

class SoftmaxCreator : public OpenCLBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        auto axis = op->main_as_Axis()->axis();
        if (-1 == axis) {
            axis = inputs[0]->dimensions() - 1;
        }
        if (1 == axis || 2 == axis) {
            return new SoftmaxExecution(inputs, axis, backend);
        }
        return nullptr;
    }
};
OpenCLCreatorRegister<SoftmaxCreator> __Softmax_op(OpType_Softmax);

} // namespace OpenCL
} // namespace MNN
