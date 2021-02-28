//
//  SoftmaxExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/SoftmaxExecution.hpp"
#include "core/Macro.h"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"

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
#ifdef MNN_OPENCL_LWS_TUNE
    MNN_ASSERT(gws.size() == 3);

    auto maxWorkItemSizes = mOpenCLBackend->getOpenCLRuntime()->getMaxWorkItemSizes();
    MNN_ASSERT(maxWorkItemSizes.size() >= 3);
    auto& tunedLws = mOpenCLBackend->getOpenCLRuntime()->tunedLwsMap();
    std::pair<std::string, std::vector<uint32_t>> info = std::make_pair("softmaxLocalWS", gws);
    if (tunedLws.find(info) != tunedLws.end()) {
        //printf("softmaxLocalWS Found! gws:%d %d lws:%d %d\n", gws[0], gws[1], tunedLws[info][0], tunedLws[info][1]);
        return tunedLws[info];
    }
    
    std::vector<uint32_t> lws(3, 1);
    std::vector<uint32_t> lws_prefer(4, 1);
    int min_cost = INT_MAX;
    while(lws[2] <= gws[2]*2  || lws[2] <= 4) {
        lws[1] = 1;
        while(lws[1] <= gws[1]*2 || lws[1] <= 4) {
            lws[0] = 1;
            while(lws[0] <= gws[0]*2  || lws[0] <= 4) {
                if(lws[0] <= maxWorkItemSizes[0] && lws[1] <= maxWorkItemSizes[1] && lws[2] <= maxWorkItemSizes[2] && lws[0]*lws[1]*lws[2] <= maxWorkGroupSize) {
                    cl::Event event;
                    std::vector<uint32_t> internalGlobalWS(3, 1);
                    for (size_t i = 0; i < gws.size(); ++i) {
                        internalGlobalWS[i] = ROUND_UP(gws[i], std::max((uint32_t)1, lws[i]));
                    }
                    cl_int error = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueNDRangeKernel(
                                    mKernel, cl::NullRange,
                                    cl::NDRange(internalGlobalWS[0], internalGlobalWS[1], internalGlobalWS[2]),
                                    cl::NDRange(lws[0], lws[1], lws[2]),
                                    nullptr, &event);
                    MNN_CHECK_CL_SUCCESS(error);

                    int cost_time = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
                    if(cost_time < min_cost) {
                        min_cost = cost_time;
                        lws_prefer[0] = lws[0];
                        lws_prefer[1] = lws[1];
                        lws_prefer[2] = lws[2];
                    }
                }
                lws[0] *= 2;
            }
            lws[1] *= 2;
        }
        lws[2] *= 2;
    }
    
    if (tunedLws.find(info) == tunedLws.end()) {
        //printf("softmaxLocalWS %d Insert! gws:%d %d, lws:%d %d\n", (int)tunedLws.size(), gws[0], gws[1], lws_prefer[0], lws_prefer[1]);
        tunedLws.insert(std::make_pair(info, lws_prefer));
    }
    
    return lws_prefer;
#else
    std::vector<uint32_t> lws(4, 0);
    
    uint32_t deviceComputeUnits = mOpenCLBackend->getOpenCLRuntime()->deviceComputeUnits();
    auto maxWorkItemSizes       = mOpenCLBackend->getOpenCLRuntime()->getMaxWorkItemSizes();
    int coreNum = deviceComputeUnits;
    for (int i = 0, totalSizeNow = 1; i < gws.size(); ++i) {
        int remain = gws[i] % coreNum, groupSize = gws[i] / coreNum;
        if (remain == 0) {
            lws[i] = groupSize;
        } else {
            while(groupSize) {
                int remain = gws[i] % groupSize;
                if (remain == 0 && (i > 0 || groupSize <= maxWorkGroupSize)) {
                    lws[i] = groupSize;
                    break;
                }
                --groupSize;
            }
        }
        int limit = std::min<uint32_t>(maxWorkGroupSize / totalSizeNow, maxWorkItemSizes[i]);
        lws[i] = std::max<uint32_t>(std::min<uint32_t>(lws[i], limit), 1);
        totalSizeNow *= lws[i];
    }
    return lws;
#endif
}

bool SoftmaxExecution::buildSoftmaxKernel() {
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    if (mKernel.get() == nullptr) {
        std::set<std::string> buildOptions;
        std::string kernelName;
        if (mAxis == 1) {
            mKernel           = runtime->buildKernel("softmax", "softmax_channel", buildOptions);
        } else if (mAxis == 2) {
            mKernel           = runtime->buildKernel("softmax_common", "softmax_height", buildOptions);
        } else {
            MNN_ASSERT(mAxis == 3);
            mKernel           = runtime->buildKernel("softmax_common", "softmax_width", buildOptions);
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
    if (mAxis == 1) {
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
    } else if (mAxis == 2){
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
    } else {
        MNN_ASSERT(mAxis == 3);
        if (mMaxWorkGroupSize > 256) {
            mLocalWorkSize = {16, 16, 1};
        } else {
            mLocalWorkSize = {8, 8, 1};
        }
        mGlobalWorkSize = {(uint32_t)channelBlocks, (uint32_t)outputBatch*outputHeight, 1};
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
    MNN_PRINT("end SoftmaxExecution onExecute !\n");
#endif

    return NO_ERROR;
}

class SoftmaxCreator : public OpenCLBackend::Creator {
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
OpenCLCreatorRegister<SoftmaxCreator> __Softmax_op(OpType_Softmax);

} // namespace OpenCL
} // namespace MNN
