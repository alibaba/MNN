//
//  DepthwiseConvExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "DepthwiseConvExecution.hpp"
#include <Macro.h>
#include <string.h>
#include "TensorUtils.hpp"

namespace MNN {
namespace OpenCL {

DepthwiseConvExecution::DepthwiseConvExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend)
    : ConvCommonExecution(op->main_as_Convolution2D(), backend) {
    mOpenCLBackend      = static_cast<OpenCLBackend *>(backend);
    mCon2dParams        = op->main_as_Convolution2D();
    mConv2dCommonParams = mCon2dParams->common();
    mStrides            = {mConv2dCommonParams->strideY(), mConv2dCommonParams->strideX()};
    mDilations          = {mConv2dCommonParams->dilateY(), mConv2dCommonParams->dilateX()};

    mPaddings[0]    = mConv2dCommonParams->padY() * 2;
    mPaddings[1]    = mConv2dCommonParams->padX() * 2;
    PadMode padMode = mConv2dCommonParams->padMode();
    if (padMode == PadMode_VALID) {
        mPaddings[0] = 0;
        mPaddings[1] = 0;
    }

    int kernelWidth   = mConv2dCommonParams->kernelX();
    int kernelHeight  = mConv2dCommonParams->kernelY();
    int outputChannel = mConv2dCommonParams->outputCount();

    std::vector<int> filterShape{1, outputChannel, kernelHeight, kernelWidth};
    std::vector<int> filterImageShape{(int)kernelHeight * kernelWidth, (int)UP_DIV(outputChannel, 4)};
    const float *filterDataPtr = mCon2dParams->weight()->data();
    mFilter.reset(Tensor::createDevice<float>({1, filterImageShape[1], 1, 4 * filterImageShape[0]}));
    std::shared_ptr<Tensor> filterBuffer(Tensor::createDevice<float>(filterShape));
    cl::Buffer filterBufferCL(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                              filterBuffer->size());
    filterBuffer->buffer().device = (uint64_t)(&filterBufferCL);
    cl_int error;
    auto ptrCL = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(filterBufferCL, true, CL_MAP_WRITE,
                                                                                     0, filterBuffer->size(), nullptr, nullptr, &error);
    if(ptrCL != nullptr && error == CL_SUCCESS){
        ::memcpy(ptrCL, filterDataPtr, filterBuffer->size());
    }else{
        MNN_ERROR("Map error ptrCL == nullptr \n");
    }
    mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(filterBufferCL, ptrCL);

    mOpenCLBackend->onAcquireBuffer(mFilter.get(), Backend::STATIC);

    MNN::OpenCL::ImageBufferConvertor imageBufferConvertor{mOpenCLBackend->getOpenCLRuntime()};
    imageBufferConvertor.convertBufferToImage(filterBuffer.get(), MNN::OpenCL::DW_CONV2D_FILTER, mFilter.get());
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    std::set<std::string> buildOptions;
    std::string kernelName = "depthwise_conv2d";
    if (mConv2dCommonParams->strideX() == 1 && mConv2dCommonParams->strideY() == 1 &&
        mConv2dCommonParams->dilateX() == 1 && mConv2dCommonParams->dilateY() == 1) {
        kernelName = "depthwise_conv2d_s1";
    }

    if (mConv2dCommonParams->relu() == true) {
        buildOptions.emplace("-DRELU");
    } else if (mConv2dCommonParams->relu6() == true) {
        buildOptions.emplace("-DRELU6");
    }

    mKernel           = runtime->buildKernel("depthwise_conv2d", kernelName, buildOptions);
    mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel));
}

DepthwiseConvExecution::~DepthwiseConvExecution() {
    mOpenCLBackend->onReleaseBuffer(mFilter.get(), Backend::STATIC);
}

ErrorCode DepthwiseConvExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input                   = inputs[0];
    auto output                  = outputs[0];
    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);

    mGlobalWorkSize = {static_cast<uint32_t>(UP_DIV(outputShape.at(3), 4) * UP_DIV(outputShape.at(2), 4)),
                       static_cast<uint32_t>(outputShape.at(0) * outputShape.at(1))};
    mLocalWorkSize  = depthwiseConvLocalWS(mGlobalWorkSize, mMaxWorkGroupSize);

    if (mConv2dCommonParams->padMode() == PadMode_SAME) {
        int kernelHeightSize = (mConv2dCommonParams->kernelY() - 1) * mConv2dCommonParams->dilateY() + 1;
        int padNeededHeight =
            (output->height() - 1) * mConv2dCommonParams->strideY() + kernelHeightSize - input->height();
        int kernelWidthSize = (mConv2dCommonParams->kernelX() - 1) * mConv2dCommonParams->dilateX() + 1;
        int padNeededWidth =
            (output->width() - 1) * mConv2dCommonParams->strideX() + kernelWidthSize - input->width();

        mPaddings[0] = padNeededHeight;
        mPaddings[1] = padNeededWidth;
    }

    const int outputHeight = outputShape.at(1);
    const int outputWidth  = outputShape.at(2);

    const int inputHeight   = inputShape.at(1);
    const int inputWidth    = inputShape.at(2);
    const int inputChannels = inputShape.at(3);

    const int inputChannelBlocks = UP_DIV(inputChannels, 4);
    const int filterHeight       = mCon2dParams->common()->kernelY();
    const int filterWidth        = mCon2dParams->common()->kernelX();
    uint32_t idx                 = 0;
    auto kernel                  = &mKernel;

    int inputImageShape[2]  = {inputHeight, inputWidth};
    int outputImageShape[2] = {outputHeight, outputWidth};
    int strideShape[2]      = {mStrides[0], mStrides[1]};
    int paddingShape[2]     = {mPaddings[0] / 2, mPaddings[1] / 2};
    int kernelShape[2]      = {filterHeight, filterWidth};
    int dilationShape[2]    = {mDilations[0], mDilations[1]};

    kernel->setArg(idx++, mGlobalWorkSize[0]);
    kernel->setArg(idx++, mGlobalWorkSize[1]);
    kernel->setArg(idx++, openCLImage(input));
    kernel->setArg(idx++, openCLImage(mFilter.get()));
    kernel->setArg(idx++, openCLImage(mBias.get()));
    kernel->setArg(idx++, openCLImage(output));
    kernel->setArg(idx++, sizeof(inputImageShape), inputImageShape);
    kernel->setArg(idx++, static_cast<int>(inputChannelBlocks));
    kernel->setArg(idx++, sizeof(outputImageShape), outputImageShape);
    kernel->setArg(idx++, sizeof(kernelShape), kernelShape);
    kernel->setArg(idx++, sizeof(paddingShape), paddingShape);
    if (mStrides[0] != 1 || mStrides[1] != 1 || mDilations[0] != 1 || mDilations[1] != 1) {
        kernel->setArg(idx++, sizeof(dilationShape), dilationShape);
        kernel->setArg(idx++, sizeof(strideShape), strideShape);
    }

    return NO_ERROR;
}

std::vector<uint32_t> DepthwiseConvExecution::depthwiseConvLocalWS(const std::vector<uint32_t> &gws,
                                                                   const uint32_t maxWorkGroupSize) {
    uint32_t deviceComputeUnits = mOpenCLBackend->getOpenCLRuntime()->deviceComputeUnits();
    std::vector<uint32_t> lws(4, 0);

    int coreNum   = deviceComputeUnits * 4;
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

    return lws;
}

ErrorCode DepthwiseConvExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start DepthwiseConvExecution onExecute !\n");
#endif

    runKernel2D(mKernel, mGlobalWorkSize, mLocalWorkSize, mOpenCLBackend->getOpenCLRuntime());

#ifdef LOG_VERBOSE
    MNN_PRINT("end DepthwiseConvExecution onExecute !\n");
#endif
    return NO_ERROR;
}

OpenCLCreatorRegister<TypedCreator<DepthwiseConvExecution>> __DepthwiseConv_op(OpType_ConvolutionDepthwise);

} // namespace OpenCL
} // namespace MNN
