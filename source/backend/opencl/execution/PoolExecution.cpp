//
//  PoolExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "execution/PoolExecution.hpp"
#include <Macro.h>
#include "TensorUtils.hpp"
#include "core/OpenCLBackend.hpp"

namespace MNN {
namespace OpenCL {

PoolExecution::PoolExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend)
    : Execution(backend) {
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
    if (mPadType == PoolPadType_VALID) {
        mPaddings[0] = 0;
        mPaddings[1] = 0;
    }
    std::set<std::string> buildOptions;
    std::string kernelName = "pooling";
    auto runtime           = mOpenCLBackend->getOpenCLRuntime();

    if (mPoolType == PoolType_AVEPOOL) {
        buildOptions.emplace("-DPOOL_AVG");
    }
    mKernel           = runtime->buildKernel("pooling", kernelName, buildOptions);
    mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel));
}

ErrorCode PoolExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start PoolExecution onResize !\n");
#endif
    auto input  = inputs[0];
    auto output = outputs[0];

    if (mPoolParams->isGlobal()) {
        std::vector<int> inputShape = tensorShapeFormat(inputs[0]);
        mKernels                    = {inputShape.at(1), inputShape.at(2)};
        mStrides                    = {inputShape.at(1), inputShape.at(2)};
        mPaddings                   = {0, 0};
    }

    if (mPadType == PoolPadType_SAME) {
        int padNeededHeight = std::max(0, (output->height() - 1) * mStrides[0] + mKernels[0] - input->height());
        int padNeededWidth  = std::max(0, (output->width() - 1) * mStrides[1] + mKernels[1] - input->width());

        mPaddings[0] = padNeededHeight;
        mPaddings[1] = padNeededWidth;
    }

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

    mGlobalWorkSize = {
        static_cast<uint32_t>(channelBlocks),
        static_cast<uint32_t>(outputWidth),
        static_cast<uint32_t>(batch * outputHeight),
    };

    int inputImageShape[2] = {inputHeight, inputWidth};
    int paddingShape[2]    = {mPaddings[0] / 2, mPaddings[1] / 2};
    int strideShape[2]     = {mStrides[0], mStrides[1]};
    int kernelShape[2]     = {mKernels[0], mKernels[1]};

    mLocalWorkSize = poolLocalWS(mGlobalWorkSize, mMaxWorkGroupSize);
    uint32_t idx   = 0;
    mKernel.setArg(idx++, mGlobalWorkSize[0]);
    mKernel.setArg(idx++, mGlobalWorkSize[1]);
    mKernel.setArg(idx++, mGlobalWorkSize[2]);
    mKernel.setArg(idx++, openCLImage(input));
    mKernel.setArg(idx++, sizeof(inputImageShape), inputImageShape);
    mKernel.setArg(idx++, static_cast<int32_t>(outputHeight));
    mKernel.setArg(idx++, sizeof(paddingShape), paddingShape);
    mKernel.setArg(idx++, sizeof(strideShape), strideShape);
    mKernel.setArg(idx++, sizeof(kernelShape), kernelShape);
    mKernel.setArg(idx++, openCLImage(output));
#ifdef LOG_VERBOSE
    MNN_PRINT("end PoolExecution onResize !\n");
#endif
    return NO_ERROR;
}

std::vector<uint32_t> PoolExecution::poolLocalWS(const std::vector<uint32_t> &gws, const uint32_t maxWorkGroupSize) {
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

ErrorCode PoolExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start PoolExecution onExecute !\n");
#endif
    run3DKernelDefault(mKernel, mGlobalWorkSize, mLocalWorkSize, mOpenCLBackend->getOpenCLRuntime());

#ifdef LOG_VERBOSE
    MNN_PRINT("end PoolExecution onExecute !\n");
#endif
    return NO_ERROR;
}

OpenCLCreatorRegister<TypedCreator<PoolExecution>> __Pool_op(OpType_Pooling);
} // namespace OpenCL
} // namespace MNN
