//
//  NormalizeExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "execution/NormalizeExecution.hpp"
#include "Macro.h"
#include "TensorUtils.hpp"
#include "core/OpenCLBackend.hpp"
#include "core/OpenCLRunningUtils.hpp"

namespace MNN {
namespace OpenCL {

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
    mAreadySetArg = false;
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

#ifdef LOG_VERBOSE
    MNN_PRINT("end NormalizeExecution onResize !\n");
#endif
    return NO_ERROR;
}

std::vector<uint32_t> NormalizeExecution::normalizeLocalWS(const std::vector<uint32_t> &gws,
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

ErrorCode NormalizeExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start NormalizeExecution onExecute !\n");
#endif

    auto runtime = mOpenCLBackend->getOpenCLRuntime();

    if (!mAreadySetArg) {
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

        mGlobalWorkSize = {static_cast<uint32_t>(channelBlocks), static_cast<uint32_t>(width),
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
        mAreadySetArg  = true;
    }

    run3DKernelDefault(mKernel, mGlobalWorkSize, mLocalWorkSize, runtime);

#ifdef LOG_VERBOSE
    MNN_PRINT("end NormalizeExecution onExecute !\n");
#endif
    return NO_ERROR;
}

OpenCLCreatorRegister<TypedCreator<NormalizeExecution>> __normalize_op(OpType_Normalize);

} // namespace OpenCL
} // namespace MNN
