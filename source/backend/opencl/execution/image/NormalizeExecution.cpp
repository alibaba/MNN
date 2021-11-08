//
//  NormalizeExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/image/NormalizeExecution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"

namespace MNN {
namespace OpenCL {

std::vector<uint32_t> NormalizeExecution::normalizeLocalWS(const std::vector<uint32_t> &gws,
                                                           const uint32_t maxWorkGroupSize) {
    std::vector<uint32_t> lws(4, 0);
    auto maxWorkItemSizes       = mOpenCLBackend->getOpenCLRuntime()->getMaxWorkItemSizes();
    uint32_t deviceComputeUnits = mOpenCLBackend->getOpenCLRuntime()->deviceComputeUnits();
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
}

NormalizeExecution::NormalizeExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend)
    : Execution(backend) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start NormalizeExecution init !\n");
#endif

    mOpenCLBackend         = static_cast<OpenCLBackend *>(backend);
    mNormalizeParams       = op->main_as_Normalize();
    int scaleSize          = mNormalizeParams->scale()->size();
    const float *scaleData = mNormalizeParams->scale()->data();
    cl::Buffer scaleBuffer(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                           UP_DIV(scaleSize, 4) * 4 * sizeof(float));
    cl_int error;
    auto biasPtrCL = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(
        scaleBuffer, true, CL_MAP_WRITE, 0, ALIGN_UP4(scaleSize) * sizeof(float), nullptr, nullptr, &error);
    if (nullptr != biasPtrCL && error == CL_SUCCESS){
        ::memset(biasPtrCL, 0, ALIGN_UP4(scaleSize) * sizeof(float));
        ::memcpy(biasPtrCL, scaleData, scaleSize * sizeof(float));
    }else{
        MNN_ERROR("Map error biasPtrCL == nullptr \n");
    }
    mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(scaleBuffer, biasPtrCL);

    mScale.reset(Tensor::createDevice<float>({1, 1, 1, scaleSize}));
    mOpenCLBackend->onAcquireBuffer(mScale.get(), Backend::STATIC);
    copyBufferToImage(mOpenCLBackend->getOpenCLRuntime(), scaleBuffer, openCLImage(mScale.get()), UP_DIV(scaleSize, 4),
                      1);

    mEps          = mNormalizeParams->eps();
#ifdef LOG_VERBOSE
    MNN_PRINT("end NormalizeExecution init !\n");
#endif
}

NormalizeExecution::~NormalizeExecution() {
    backend()->onReleaseBuffer(mScale.get(), Backend::STATIC);
}

ErrorCode NormalizeExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start NormalizeExecution onResize !\n");
#endif
    auto runtime = mOpenCLBackend->getOpenCLRuntime();

    if (mKernel.get() == nullptr) {
        std::set<std::string> buildOptions;
        std::string kernelName = "normalize_kernel";
        mKernel                = runtime->buildKernel("normalize", kernelName, buildOptions);
        mMaxWorkGroupSize      = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel));
    }

    Tensor *input  = inputs[0];
    Tensor *output = outputs[0];

    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);

    const int batch    = inputShape.at(0);
    const int height   = inputShape.at(1);
    const int width    = inputShape.at(2);
    const int channels = inputShape.at(3);

    const int channelBlocks  = UP_DIV(channels, 4);
    const int remainChannels = channelBlocks * 4 - channels;

    mGlobalWorkSize = {static_cast<uint32_t>(channelBlocks),
                       static_cast<uint32_t>(width),
                       static_cast<uint32_t>(height * batch)};
    uint32_t idx    = 0;
    mKernel.setArg(idx++, mGlobalWorkSize[0]);
    mKernel.setArg(idx++, mGlobalWorkSize[1]);
    mKernel.setArg(idx++, mGlobalWorkSize[2]);

    mKernel.setArg(idx++, openCLImage(input));
    mKernel.setArg(idx++, openCLImage(mScale.get()));
    mKernel.setArg(idx++, mEps);
    mKernel.setArg(idx++, channelBlocks);
    mKernel.setArg(idx++, remainChannels);
    mKernel.setArg(idx++, openCLImage(output));
    mLocalWorkSize = normalizeLocalWS(mGlobalWorkSize, mMaxWorkGroupSize);
    
#ifdef LOG_VERBOSE
    MNN_PRINT("end NormalizeExecution onResize !\n");
#endif
    return NO_ERROR;
}

ErrorCode NormalizeExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start NormalizeExecution onExecute !\n");
#endif

#ifdef ENABLE_OPENCL_TIME_PROFILER
    cl::Event event;
    run3DKernelDefault(mKernel, mGlobalWorkSize, mLocalWorkSize,
                       mOpenCLBackend->getOpenCLRuntime(), &event);
    
    int costTime = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
    MNN_PRINT("kernel cost:%d    us Normalize\n",costTime);
#else
    run3DKernelDefault(mKernel, mGlobalWorkSize, mLocalWorkSize,
                       mOpenCLBackend->getOpenCLRuntime());
#endif
    
#ifdef LOG_VERBOSE
    MNN_PRINT("end NormalizeExecution onExecute !\n");
#endif
    return NO_ERROR;
}

OpenCLCreatorRegister<TypedCreator<NormalizeExecution>> __normalize_op(OpType_Normalize, IMAGE);

} // namespace OpenCL
} // namespace MNN
