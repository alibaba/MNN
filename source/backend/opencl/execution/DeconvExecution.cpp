//
//  DeconvExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "execution/DeconvExecution.hpp"
#include "Macro.h"
#include "TensorUtils.hpp"
#include "core/OpenCLBackend.hpp"
#include "core/OpenCLRunningUtils.hpp"

namespace MNN {
namespace OpenCL {

DeconvExecution::DeconvExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend)
    : ConvCommonExecution(op->main_as_Convolution2D(), backend) {
    mOpenCLBackend                 = static_cast<OpenCLBackend *>(backend);
    const auto *conv2dParams       = op->main_as_Convolution2D();
    const auto *conv2dCommonParams = conv2dParams->common();
    mConv2dCommonParams            = conv2dCommonParams;
    mStrides                       = {conv2dCommonParams->strideY(), conv2dCommonParams->strideX()};
    mDilations                     = {conv2dCommonParams->dilateY(), conv2dCommonParams->dilateX()};
    int kernelWidth                = conv2dCommonParams->kernelX();
    int kernelHeight               = conv2dCommonParams->kernelY();

    MNN_ASSERT(mStrides[0] > 0 && mStrides[1] > 0);
    mPaddings[0]    = (kernelHeight - 1 - conv2dCommonParams->padY()) * 2;
    mPaddings[1]    = (kernelWidth - 1 - conv2dCommonParams->padX()) * 2;
    PadMode padMode = conv2dCommonParams->padMode();
    if (padMode == PadMode_VALID) {
        mPaddings[0] = 0;
        mPaddings[1] = 0;
    }

    int outputChannel = conv2dCommonParams->outputCount();
    int weightSize    = conv2dParams->weight()->size();
    int inputChannel  = weightSize / (kernelWidth * kernelHeight * outputChannel);
    std::vector<int> filterShape{outputChannel, inputChannel, kernelHeight, kernelWidth};
    std::vector<int> filterImageShape{(int)inputChannel, (int)UP_DIV(outputChannel, 4) * kernelWidth * kernelHeight};
    const float *filterDataPtr = conv2dParams->weight()->data();
    std::vector<float> filterDataPtrTransformed;
    filterDataPtrTransformed.resize(conv2dParams->weight()->size());
    IOHW2OIHW<float, int>(filterDataPtr, filterDataPtrTransformed.data(), outputChannel, inputChannel, kernelHeight,
                          kernelWidth);

    std::shared_ptr<Tensor> filterBuffer(
        Tensor::createDevice<float>({outputChannel, inputChannel, kernelHeight, kernelWidth}));
    cl::Buffer filterBufferCL(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                              filterBuffer->size());
    filterBuffer->buffer().device = (uint64_t)(&filterBufferCL);
    cl_int error;
    auto ptrCL = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(filterBufferCL, true, CL_MAP_WRITE,
                                                                                     0, filterBuffer->size(), nullptr, nullptr, &error);
    if(ptrCL != nullptr && error == CL_SUCCESS){
        ::memcpy(ptrCL, filterDataPtrTransformed.data(), filterBuffer->size());
    }else{
        MNN_ERROR("Map error ptrCL == nullptr \n");
    }
    mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(filterBufferCL, ptrCL);

    mFilter.reset(Tensor::createDevice<float>({1, filterImageShape[1], 1, 4 * filterImageShape[0]}));
    mOpenCLBackend->onAcquireBuffer(mFilter.get(), Backend::STATIC);
    MNN::OpenCL::ImageBufferConvertor imageBufferConvertor{mOpenCLBackend->getOpenCLRuntime()};
    imageBufferConvertor.convertBufferToImage(filterBuffer.get(), MNN::OpenCL::CONV2D_FILTER, mFilter.get());

    auto runtime = mOpenCLBackend->getOpenCLRuntime();

    std::set<std::string> buildOptions;
    std::string kernelName = "deconv_2d";
    if (conv2dCommonParams->relu() == true) {
        buildOptions.emplace("-DRELU");
    } else if (conv2dCommonParams->relu6() == true) {
        buildOptions.emplace("-DRELU6");
    }
    mKernel = runtime->buildKernel("deconv_2d", kernelName, buildOptions);
}

DeconvExecution::~DeconvExecution() {
    mOpenCLBackend->onReleaseBuffer(mFilter.get(), Backend::STATIC);
}

std::vector<uint32_t> DeconvExecution::deconvLocalWS(const uint32_t *gws, const uint32_t maxWorkGroupSize) {
    std::vector<uint32_t> lws(4, 0);
    uint32_t deviceComputeUnits = mOpenCLBackend->getOpenCLRuntime()->deviceComputeUnits();
    int coreNum                 = deviceComputeUnits;
    int remain                  = gws[0] % coreNum;
    int groupSize               = gws[0] / coreNum;
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
    return lws;
}

ErrorCode DeconvExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto output = outputs[0];
    auto input  = inputs[0];

    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);

    if (mConv2dCommonParams->padMode() == PadMode_SAME) {
        int extendedInputHeight = (input->height() - 1) * mConv2dCommonParams->strideY() + 1;
        int extended_inputWidth = (input->width() - 1) * mConv2dCommonParams->strideX() + 1;
        mPaddings[0]            = (output->height() + mConv2dCommonParams->kernelY() - 1 - extendedInputHeight);
        mPaddings[1]            = (output->width() + mConv2dCommonParams->kernelX() - 1 - extended_inputWidth);
    }

    const int outputBatch    = outputShape.at(0);
    const int outputHeight   = outputShape.at(1);
    const int outputWidth    = outputShape.at(2);
    const int outputChannels = outputShape.at(3);

    const int inputChannels = inputShape.at(3);

    const int outputChannelBlocks = UP_DIV(outputChannels, 4);
    const int strideHeight        = mStrides[0];
    const int strideWidth         = mStrides[1];

    const int paddingHeight = UP_DIV(mPaddings[0], 2);
    const int paddingWidth  = UP_DIV(mPaddings[1], 2);

    const int alignHeight = mStrides[0] - 1 - paddingHeight;
    const int alignWidth  = mStrides[1] - 1 - paddingWidth;

    const int kernelSize = mConv2dCommonParams->kernelY() * mConv2dCommonParams->kernelX();
    auto ky              = mConv2dCommonParams->kernelY();
    auto kx              = mConv2dCommonParams->kernelX();

    auto runtime      = mOpenCLBackend->getOpenCLRuntime();
    auto kernel       = &mKernel;
    mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel));
    mGWS              = {static_cast<uint32_t>(outputChannelBlocks), static_cast<uint32_t>(outputWidth),
            static_cast<uint32_t>(outputHeight * outputBatch)};

    int inputImageShape[2]  = {inputShape.at(1), inputShape.at(2)};
    int outputImageShape[2] = {outputHeight, outputWidth};
    int strideShape[2]      = {strideHeight, strideWidth};
    int paddingShape[2]     = {paddingHeight, paddingWidth};
    int alignShape[2]       = {alignHeight, alignWidth};
    int kernelShape[2]      = {ky, kx};

    uint32_t idx = 0;
    kernel->setArg(idx++, mGWS[0]);
    kernel->setArg(idx++, mGWS[1]);
    kernel->setArg(idx++, mGWS[2]);
    kernel->setArg(idx++, openCLImage(input));
    kernel->setArg(idx++, openCLImage(mFilter.get()));
    kernel->setArg(idx++, openCLImage(mBias.get()));
    kernel->setArg(idx++, openCLImage(output));
    kernel->setArg(idx++, sizeof(inputImageShape), inputImageShape);
    kernel->setArg(idx++, sizeof(outputImageShape), outputImageShape);
    kernel->setArg(idx++, sizeof(strideShape), strideShape);
    kernel->setArg(idx++, sizeof(alignShape), alignShape);
    kernel->setArg(idx++, sizeof(paddingShape), paddingShape);
    kernel->setArg(idx++, sizeof(kernelShape), kernelShape);
    kernel->setArg(idx++, static_cast<int32_t>(kernelSize));
    kernel->setArg(idx++, static_cast<int32_t>(UP_DIV(inputChannels, 4)));
    kernel->setArg(idx++, static_cast<int32_t>(outputChannelBlocks));
    mLWS = localWS3DDefault(mGWS, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime());
    return NO_ERROR;
}

ErrorCode DeconvExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start DeconvExecution onExecute... \n");
#endif
    run3DKernelDefault(mKernel, mGWS, mLWS, mOpenCLBackend->getOpenCLRuntime());

#ifdef LOG_VERBOSE
    MNN_PRINT("End DeconvExecution onExecute... \n");
#endif
    return NO_ERROR;
}

OpenCLCreatorRegister<TypedCreator<DeconvExecution>> __deconv_op(OpType_Deconvolution);

} // namespace OpenCL
} // namespace MNN
