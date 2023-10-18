//
//  PoolBufExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED

#include "backend/opencl/execution/buffer/PoolBufExecution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"

namespace MNN {
namespace OpenCL {

PoolBufExecution::PoolBufExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend)
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
    if (inputs[0]->channel() >= 16) {
        TensorUtils::setTensorChannelPack(inputs[0], 16);
    }
}

ErrorCode PoolBufExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start PoolBufExecution onResize !\n");
#endif
    auto input  = inputs[0];
    auto output = outputs[0];
    bool returnRedice = outputs.size() == 2;
    auto redice = returnRedice ? outputs[1] : outputs[0];

    auto runtime = mOpenCLBackend->getOpenCLRuntime();
#ifdef MNN_SUPPORT_INTEL_SUBGROUP
    if (runtime->isSupportedIntelSubgroup()) {
        return SubgrouponResize(inputs, outputs);
    }
#endif /* MNN_SUPPORT_INTEL_SUBGROUP */
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
    }else if (mPoolParams->padType() == PoolPadType_VALID) {
        mPaddings[0] = mPaddings[1] = 0;
    }
    
    auto countType         = mPoolParams->countType();
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
    
    std::set<std::string> buildOptions;
    std::string kernelName = "pooling";

    if (mPoolType == PoolType_AVEPOOL) {
        buildOptions.emplace("-DPOOL_AVG");
        if(countType == MNN::AvgPoolCountType_INCLUDE_PADDING){
            buildOptions.emplace("-DCOUNT_INCLUDE_PADDING");
        }
    }
    if(returnRedice){
        buildOptions.emplace("-DRETURN_REDICE");
    }
    
    mKernel           = runtime->buildKernel("pooling_buf", kernelName, buildOptions);
    mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel));
    
    mGlobalWorkSize = {
        static_cast<uint32_t>(outputWidth),
        static_cast<uint32_t>(batch * outputHeight),
        static_cast<uint32_t>(channelBlocks),
    };

    int inputImageShape[2] = {inputHeight, inputWidth};
    int outputImageShape[2] = {outputHeight, outputWidth};
    int paddingShape[2]    = {mPaddings[0] / 2, mPaddings[1] / 2};
    int strideShape[2]     = {mStrides[0], mStrides[1]};
    int kernelShape[2]     = {mKernels[0], mKernels[1]};

    uint32_t idx   = 0;
    cl_int ret = CL_SUCCESS;
    ret |= mKernel.setArg(idx++, mGlobalWorkSize[0]);
    ret |= mKernel.setArg(idx++, mGlobalWorkSize[1]);
    ret |= mKernel.setArg(idx++, mGlobalWorkSize[2]);
    ret |= mKernel.setArg(idx++, openCLBuffer(input));
    ret |= mKernel.setArg(idx++, sizeof(inputImageShape), inputImageShape);
    ret |= mKernel.setArg(idx++, sizeof(outputImageShape), outputImageShape);
    ret |= mKernel.setArg(idx++, sizeof(paddingShape), paddingShape);
    ret |= mKernel.setArg(idx++, sizeof(strideShape), strideShape);
    ret |= mKernel.setArg(idx++, sizeof(kernelShape), kernelShape);
    ret |= mKernel.setArg(idx++, openCLBuffer(output));
    ret |= mKernel.setArg(idx++, openCLBuffer(redice));
    ret |= mKernel.setArg(idx++, channelBlocks);
    MNN_CHECK_CL_SUCCESS(ret, "setArg PoolBufExecution");
    
    std::string kernelNameTune = "pooling_buf";
    mLocalWorkSize =
    localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), kernelNameTune, mKernel).first;

#ifdef LOG_VERBOSE
    MNN_PRINT("end PoolBufExecution onResize !\n");
#endif
    return NO_ERROR;
}

#ifdef MNN_SUPPORT_INTEL_SUBGROUP
ErrorCode PoolBufExecution::SubgrouponResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start PoolBufExecution onResize !\n");
#endif
    auto input  = inputs[0];
    auto output = outputs[0];
    bool returnRedice = outputs.size() == 2;
    auto redice = returnRedice ? outputs[1] : outputs[0];

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
    int input_c_pack = TensorUtils::getTensorChannelPack(input);
    int output_c_pack = TensorUtils::getTensorChannelPack(output);

    auto inputpad  = TensorUtils::getDescribe(input)->mPads;
    auto outputpad = TensorUtils::getDescribe(output)->mPads;
    
    int inputImageShape[2] = {inputHeight, inputWidth};
    int outputImageShape[2] = {outputHeight, outputWidth};
    int paddingShape[2] = {mPaddings[0] / 2, mPaddings[1] / 2};
    int strideShape[2] = {mStrides[0], mStrides[1]};
    int kernelShape[2] = {mKernels[0], mKernels[1]};
    int in_channel_block    = UP_DIV(channels, input_c_pack);
    int out_channel_block   = UP_DIV(channels, output_c_pack);

    std::set<std::string> buildOptions;

    std::string KernelName = "pooling_c" + std::to_string(input_c_pack) + "_c" + std::to_string(output_c_pack);
    auto runtime           = mOpenCLBackend->getOpenCLRuntime();

    if (mPoolType == PoolType_AVEPOOL) {
        buildOptions.emplace("-DPOOL_AVG");
    }
    if(returnRedice){
        buildOptions.emplace("-DRETURN_REDICE");
    }
    int input_line_size = mStrides[1] * (8 - 1) + mKernels[1];
    buildOptions.emplace("-DINPUT_LINE_SIZE=" + std::to_string(input_line_size));
    if (channels % 16 != 0) {
        buildOptions.emplace("-DOUTPUT_LEFTOVERS=" + std::to_string(1));
    }
    buildOptions.emplace("-DSTRIDE_Y=" + std::to_string(strideShape[0]));
    buildOptions.emplace("-DSTRIDE_X=" + std::to_string(strideShape[1]));
    buildOptions.emplace("-DKERNEL_Y=" + std::to_string(kernelShape[0]));
    buildOptions.emplace("-DKERNEL_X=" + std::to_string(kernelShape[1]));
    
    mKernel           = runtime->buildKernel("pooling_subgroup_buf", KernelName, buildOptions);
    mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel));

    mGlobalWorkSize = {
        static_cast<uint32_t>(ROUND_UP(channels, 16)),
        static_cast<uint32_t>(UP_DIV(outputWidth, 8)),
        static_cast<uint32_t>(batch * outputHeight),
    };
    if (input_c_pack == 4) {
        mGlobalWorkSize = {
            static_cast<uint32_t>(outputWidth),
            static_cast<uint32_t>(batch * outputHeight),
            static_cast<uint32_t>(UP_DIV(channels, 4)),
        };
    }

    uint32_t idx   = 0;
    cl_int ret = CL_SUCCESS;
    ret |= mKernel.setArg(idx++, mGlobalWorkSize[0]);
    ret |= mKernel.setArg(idx++, mGlobalWorkSize[1]);
    ret |= mKernel.setArg(idx++, mGlobalWorkSize[2]);
    ret |= mKernel.setArg(idx++, openCLBuffer(input));
    ret |= mKernel.setArg(idx++, sizeof(inputImageShape), inputImageShape);
    ret |= mKernel.setArg(idx++, sizeof(outputImageShape), outputImageShape);
    ret |= mKernel.setArg(idx++, sizeof(paddingShape), paddingShape);
    ret |= mKernel.setArg(idx++, openCLBuffer(output));
    ret |= mKernel.setArg(idx++, openCLBuffer(redice));
    ret |= mKernel.setArg(idx++, channels);
    ret |= mKernel.setArg(idx++, in_channel_block);
    ret |= mKernel.setArg(idx++, out_channel_block);
    ret |= mKernel.setArg(idx++, static_cast<uint32_t>(inputpad.left));
    ret |= mKernel.setArg(idx++, static_cast<uint32_t>(inputpad.right));
    ret |= mKernel.setArg(idx++, static_cast<uint32_t>(outputpad.left));
    ret |= mKernel.setArg(idx++, static_cast<uint32_t>(outputpad.right));
    MNN_CHECK_CL_SUCCESS(ret, "setArg PoolBufExecution SubGroup");

    std::string kernelNameTune = "pooling_subgroup_buf";
    if (input_c_pack == 16) {
        mLocalWorkSize = {16, 1, 1};
    } else {
        mLocalWorkSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), kernelNameTune, mKernel).first;
    }

#ifdef LOG_VERBOSE
    MNN_PRINT("end PoolBufExecution onResize !\n");
#endif
    return NO_ERROR;
}
#endif /* MNN_SUPPORT_INTEL_SUBGROUP */

ErrorCode PoolBufExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start PoolBufExecution onExecute !\n");
#endif
    
#ifdef ENABLE_OPENCL_TIME_PROFILER
    cl::Event event;
    run3DKernelDefault(mKernel, mGlobalWorkSize, mLocalWorkSize,
                       mOpenCLBackend->getOpenCLRuntime(), &event);
    
    mOpenCLBackend->getOpenCLRuntime()->pushEvent({"Pooling", event});
#else
    run3DKernelDefault(mKernel, mGlobalWorkSize, mLocalWorkSize,
                       mOpenCLBackend->getOpenCLRuntime());
#endif
    
#ifdef LOG_VERBOSE
    MNN_PRINT("end PoolBufExecution onExecute !\n");
#endif
    return NO_ERROR;
}

OpenCLCreatorRegister<TypedCreator<PoolBufExecution>> __PoolBuf_op(OpType_Pooling, BUFFER);
} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
