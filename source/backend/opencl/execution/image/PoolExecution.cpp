//
//  PoolExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/image/PoolExecution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"

namespace MNN {
namespace OpenCL {

std::vector<uint32_t> PoolExecution::poolLocalWS(const std::vector<uint32_t> &gws, const uint32_t maxWorkGroupSize) {
    std::vector<uint32_t> lws(3, 0);
    auto maxWorkItemSizes = mOpenCLBackend->getOpenCLRuntime()->getMaxWorkItemSizes();
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

PoolExecution::PoolExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend)
    : CommonExecution(backend, op) {
    mUnits.resize(1);
    auto &unit = mUnits[0];
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    mPoolParams    = op->main_as_Pool();
    mPoolType      = mPoolParams->type();

    mStrides[0] = mPoolParams->strideY();
    mStrides[1] = mPoolParams->strideX();
    mKernels[0] = mPoolParams->kernelY();
    mKernels[1] = mPoolParams->kernelX();

    mPaddings[0] = mPoolParams->padY() * 2;
    mPaddings[1] = mPoolParams->padX() * 2;
    mPadType     = mPoolParams->padType();
    unit.kernel = mOpenCLBackend->getOpenCLRuntime()->buildKernel("pooling", "global_pooling", {"-DLOCAL_SIZE=512"});
    mMaxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(unit.kernel));
}

int PoolExecution::getLocalSize(int size, int maxGroupSize){
    int local_size = 1;
    while(local_size * 2 <= maxGroupSize && local_size * 2 <= size){
        local_size *= 2;
    }
    return local_size;
}

ErrorCode PoolExecution::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start PoolExecution onResize !\n");
#endif
    auto &unit = mUnits[0];
    auto input  = inputs[0];
    auto output = outputs[0];
    bool returnRedice = outputs.size() == 2;
    auto redice = returnRedice ? outputs[1] : outputs[0];
    std::set<std::string> buildOptions;
    std::string kernelName = "pooling";
    auto runtime           = mOpenCLBackend->getOpenCLRuntime();
    int local_size;

    if (mPoolParams->isGlobal()) {
        std::vector<int> inputShape = tensorShapeFormat(inputs[0]);
        mKernels                    = {inputShape.at(1), inputShape.at(2)};
        mStrides                    = {inputShape.at(1), inputShape.at(2)};
        mPaddings                   = {0, 0};
        kernelName                  = "global_pooling";
        auto MaxLocalSize = std::min(runtime->getMaxWorkItemSizes()[0], mMaxWorkGroupSize);
        local_size = getLocalSize(inputShape.at(1) * inputShape.at(2), MaxLocalSize);
        buildOptions.emplace("-DLOCAL_SIZE=" + std::to_string(local_size));
    }

    if (mPadType == PoolPadType_SAME) {
        int padNeededHeight = std::max(0, (output->height() - 1) * mStrides[0] + mKernels[0] - input->height());
        int padNeededWidth  = std::max(0, (output->width() - 1) * mStrides[1] + mKernels[1] - input->width());

        mPaddings[0] = padNeededHeight;
        mPaddings[1] = padNeededWidth;
    }else if (mPadType == PoolPadType_VALID) {
        mPaddings[0] = mPaddings[1] = 0;
    }
    
    auto countType = mPoolParams->countType();
    if (mPoolParams->pads() != nullptr && mPadType == PoolPadType_CAFFE) {
        mPadType = PoolPadType_VALID;
    }
    
    if (countType == MNN::AvgPoolCountType_DEFAULT) {
        if (mPadType == MNN::PoolPadType_CAFFE) {
            countType = MNN::AvgPoolCountType_INCLUDE_PADDING;
        } else {
            countType = MNN::AvgPoolCountType_EXCLUDE_PADDING;
        }
    }

    if (mPoolType == PoolType_AVEPOOL) {
        buildOptions.emplace("-DPOOL_AVG");
        if(countType == MNN::AvgPoolCountType_INCLUDE_PADDING){
            buildOptions.emplace("-DCOUNT_INCLUDE_PADDING");
        }
    }
    if(returnRedice){
        buildOptions.emplace("-DRETURN_REDICE");
    }
    unit.kernel           = runtime->buildKernel("pooling", kernelName, buildOptions);
    mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));

    MNN_ASSERT(mDilations[0] == 1 && mDilations[1] == 1);

    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);

    const int batch        = outputShape.at(0);
    const int outputHeight = outputShape.at(1);
    const int outputWidth  = outputShape.at(2);
    const int channels     = outputShape.at(3);

    const int inputHeight = inputShape.at(1);
    const int inputWidth  = inputShape.at(2);

    int channelBlocks = (channels + 3) / 4;
    std::vector<uint32_t> mGlobalWorkSize{1, 1, 1};
    std::vector<uint32_t> mLocalWorkSize{1, 1, 1, 1};
    if (mPoolParams->isGlobal()) {
        mGlobalWorkSize = {
            static_cast<uint32_t>(local_size),
            static_cast<uint32_t>(channelBlocks),
            static_cast<uint32_t>(batch),
        };
        mLocalWorkSize = {
            static_cast<uint32_t>(local_size),
            static_cast<uint32_t>(1),
            static_cast<uint32_t>(1),
        };
    }else{
        mGlobalWorkSize = {
            static_cast<uint32_t>(channelBlocks),
            static_cast<uint32_t>(outputWidth),
            static_cast<uint32_t>(batch * outputHeight),
        };
        mLocalWorkSize = poolLocalWS(mGlobalWorkSize, mMaxWorkGroupSize);
    }

    int inputImageShape[2] = {inputHeight, inputWidth};
    int paddingShape[2]    = {mPaddings[0] / 2, mPaddings[1] / 2};
    int strideShape[2]     = {mStrides[0], mStrides[1]};
    int kernelShape[2]     = {mKernels[0], mKernels[1]};

    uint32_t idx   = 0;
    cl_int ret = CL_SUCCESS;
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[0]);
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[1]);
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[2]);
    ret |= unit.kernel->get().setArg(idx++, openCLImage(input));
    ret |= unit.kernel->get().setArg(idx++, sizeof(inputImageShape), inputImageShape);
    ret |= unit.kernel->get().setArg(idx++, static_cast<int32_t>(outputHeight));
    ret |= unit.kernel->get().setArg(idx++, sizeof(paddingShape), paddingShape);
    ret |= unit.kernel->get().setArg(idx++, sizeof(strideShape), strideShape);
    ret |= unit.kernel->get().setArg(idx++, sizeof(kernelShape), kernelShape);
    ret |= unit.kernel->get().setArg(idx++, openCLImage(output));
    ret |= unit.kernel->get().setArg(idx++, openCLImage(redice));
    MNN_CHECK_CL_SUCCESS(ret, "setArg PoolExecution");

    mOpenCLBackend->recordKernel3d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
    unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
    unit.localWorkSize = {mLocalWorkSize[0], mLocalWorkSize[1], mLocalWorkSize[2]};
#ifdef LOG_VERBOSE
    MNN_PRINT("end PoolExecution onResize !\n");
#endif
    return NO_ERROR;
}

using PoolCreator = TypedCreator<PoolExecution>;
REGISTER_OPENCL_OP_CREATOR(PoolCreator, OpType_Pooling, IMAGE);

} // namespace OpenCL
} // namespace MNN
