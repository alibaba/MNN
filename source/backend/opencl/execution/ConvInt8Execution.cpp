//
//  ConvInt8Execution.cpp
//  MNN
//
//  Created by MNN on 2019/5/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/ConvInt8Execution.hpp"
#include "backend/opencl/execution/InterpExecution.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "core/Concurrency.h"
#include "backend/cpu/compute/Int8FunctionsOpt.h"
#include "core/Macro.h"
#include "backend/opencl/core/OpenCLBackend.hpp"

namespace MNN {
namespace OpenCL {

std::vector<uint32_t> ConvInt8Execution::conv2dGeneralLocalWS(const std::vector<uint32_t> &gws, const uint32_t kernelSize,
                                                          const uint32_t maxWorkGroupSize) {
    uint32_t deviceComputeUnits = mOpenCLBackend->getOpenCLRuntime()->deviceComputeUnits();

    GpuType gpuType = mOpenCLBackend->getOpenCLRuntime()->getGpuType();

    std::vector<uint32_t> lws(4, 0);

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

ConvInt8Execution::ConvInt8Execution(Backend* backend, const MNN::Op* op) : Execution(backend) {
    mOpenCLBackend                 = static_cast<OpenCLBackend *>(backend);
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    const auto *conv2dParams       = op->main_as_Convolution2D();
    const auto *conv2dCommonParams = conv2dParams->common();
    mConv2dCommonParams            = conv2dCommonParams;
    mStrides                       = {conv2dCommonParams->strideY(), conv2dCommonParams->strideX()};
    mDilations                     = {conv2dCommonParams->dilateY(), conv2dCommonParams->dilateX()};

    mPaddings[0]    = conv2dCommonParams->padY() * 2;
    mPaddings[1]    = conv2dCommonParams->padX() * 2;
    PadMode padMode = conv2dCommonParams->padMode();
    if (padMode == PadMode_VALID) {
        mPaddings[0] = 0;
        mPaddings[1] = 0;
    }

    int kernelWidth   = conv2dCommonParams->kernelX();
    int kernelHeight  = conv2dCommonParams->kernelY();
    int outputChannel = conv2dCommonParams->outputCount();

    int weightSize  = conv2dParams->symmetricQuan()->weight()->size();
    int inputChannel = weightSize / (kernelWidth * kernelHeight * outputChannel);
    const int8_t* weightSrc  = conv2dParams->symmetricQuan()->weight()->data();

//weight
    int needFilterSize = ALIGN_UP4(inputChannel) * kernelHeight * kernelWidth * ALIGN_UP4(outputChannel) * sizeof(int8_t);
    mFilterBuffer.reset(new cl::Buffer(runtime->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, needFilterSize));

    auto filterDeviceBuffer = (cl::Buffer*)mFilterBuffer.get();
    cl_int error                = CL_SUCCESS;
    int8_t* filterBufferPtr = (int8_t*)runtime->commandQueue().enqueueMapBuffer(*filterDeviceBuffer, CL_TRUE, CL_MAP_WRITE, 0,
                                                                         needFilterSize, nullptr, nullptr, &error);
    if (error != CL_SUCCESS) {
        MNN_ERROR("Error to map buffer in copy buffer, error=%d\n", error);
        return;
    }
    if(filterBufferPtr != nullptr){
        memset(filterBufferPtr, 0, needFilterSize);
        int outputChannel4 = ALIGN_UP4(outputChannel);
        int inputChannel4 = ALIGN_UP4(inputChannel);
        for(int ks = 0; ks < kernelHeight*kernelWidth; ks++){
            for (int ic = 0; ic < inputChannel; ic++){
                for(int oc = 0; oc < outputChannel; oc++){
                    filterBufferPtr[ks*inputChannel4*outputChannel4 + (ic/4)*outputChannel4*4 + (oc/4)*16 + (ic%4)*4 + oc%4] = weightSrc[oc*kernelHeight*kernelWidth*inputChannel + ic*kernelHeight*kernelWidth + ks];
                }
            }
        }
    }

    runtime->commandQueue().enqueueUnmapMemObject(*filterDeviceBuffer, filterBufferPtr);


//Bias
    int needBiasSize = ALIGN_UP4(outputChannel) * sizeof(int32_t);
    mBiasBuffer.reset(new cl::Buffer(runtime->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, needBiasSize));

    auto BiasDeviceBuffer = (cl::Buffer*)mBiasBuffer.get();
    int32_t* BiasbufferPtr = (int32_t*)runtime->commandQueue().enqueueMapBuffer(*BiasDeviceBuffer, CL_TRUE, CL_MAP_WRITE, 0,
                                                                         needBiasSize, nullptr, nullptr, &error);
    if (error != CL_SUCCESS) {
        MNN_ERROR("Error to map buffer in copy buffer, error=%d\n", error);
        return;
    }
    if(BiasbufferPtr != nullptr){
        memset(BiasbufferPtr, 0, needBiasSize);
        memcpy(BiasbufferPtr, conv2dParams->symmetricQuan()->bias()->data(), outputChannel * sizeof(int32_t));
    }
    runtime->commandQueue().enqueueUnmapMemObject(*BiasDeviceBuffer, BiasbufferPtr);


//scale
    int needScaleSize = ALIGN_UP4(outputChannel) * sizeof(float);
    mScaleBuffer.reset(new cl::Buffer(runtime->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, needScaleSize));
    auto scaleDeviceBuffer = (cl::Buffer*)mScaleBuffer.get();
    float* scaleBufferPtr = (float*)runtime->commandQueue().enqueueMapBuffer(*scaleDeviceBuffer, CL_TRUE, CL_MAP_WRITE, 0,
                                                                         needScaleSize, nullptr, nullptr, &error);
    if (error != CL_SUCCESS) {
        MNN_ERROR("Error to map buffer in copy buffer, error=%d\n", error);
        return;
    }
    if(scaleBufferPtr != nullptr){
        memset(scaleBufferPtr, 0, needScaleSize);
        memcpy(scaleBufferPtr, conv2dParams->symmetricQuan()->scale()->data(), needScaleSize);
    }

    runtime->commandQueue().enqueueUnmapMemObject(*mScaleBuffer, scaleBufferPtr);

    // Create Kernel
    std::set<std::string> buildOptions;
    buildOptions.emplace("-DBIAS");
    if (mConv2dCommonParams->relu()) {
        buildOptions.emplace("-DRELU");
    } else if (mConv2dCommonParams->relu6()) {
        buildOptions.emplace("-DRELU6");
    }

    std::string kernelName;
    if(kernelWidth == 1 && kernelHeight == 1){
        kernelName = "conv_2d1x1";
        mIsConv1x1 = true;
    }else{
        kernelName = "conv_2d";
    }

    mKernel           = mOpenCLBackend->getOpenCLRuntime()->buildKernel("conv_2d_int8", kernelName, buildOptions);
    mMaxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(mKernel));

}
ConvInt8Execution::~ConvInt8Execution() {

}

ErrorCode ConvInt8Execution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];

    MNN_ASSERT(input->batch() == 1);
    MNN_ASSERT(mDilations[0] == 1);
    const int inputChannelBlocks = UP_DIV(input->channel(), 4);

    if (mConv2dCommonParams->padMode() == PadMode_SAME) {
        int kernelHeightSize = (mConv2dCommonParams->kernelY() - 1) * mConv2dCommonParams->dilateY() + 1;
        int padNeededHeight = (output->height() - 1) * mConv2dCommonParams->strideY() +
                kernelHeightSize - input->height();
        int kernelWidthSize = (mConv2dCommonParams->kernelX() - 1) * mConv2dCommonParams->dilateX() + 1;
        int padNeededWidth = (output->width() - 1) * mConv2dCommonParams->strideX() + kernelWidthSize -
                             input->width();
        mPaddings[0] = padNeededHeight;
        mPaddings[1] = padNeededWidth;
    }

    int kernelHeight = mConv2dCommonParams->kernelY();
    int kernelWidth  = mConv2dCommonParams->kernelX();

    mGlobalWorkSize         = {static_cast<uint32_t>(UP_DIV(output->channel(), 4)), static_cast<uint32_t>(UP_DIV(output->width(), 4)),
                        static_cast<uint32_t>(output->batch() * output->height())};

    mLocalWorkSize          = conv2dGeneralLocalWS(mGlobalWorkSize, kernelHeight * kernelWidth, mMaxWorkGroupSize);

    int inputImageShape[2]  = {input->height(), input->width()};
    int outputImageShape[2] = {output->height(), output->width()};
    int kernelShape[2]      = {kernelHeight, kernelWidth};
    int strideShape[2]      = {mStrides[0], mStrides[1]};
    int paddingShape[2]     = {mPaddings[0] / 2, mPaddings[1] / 2};
    int dilationShape[2]    = {mDilations[0], mDilations[1]};

    if(mIsConv1x1){
        uint32_t idx            = 0;
        auto kernel             = &mKernel;
        kernel->setArg(idx++, mGlobalWorkSize[0]);
        kernel->setArg(idx++, mGlobalWorkSize[1]);
        kernel->setArg(idx++, mGlobalWorkSize[2]);
        kernel->setArg(idx++, openCLBuffer(input));
        kernel->setArg(idx++, *(mFilterBuffer.get()));
        kernel->setArg(idx++, *(mBiasBuffer.get()));
        kernel->setArg(idx++, openCLBuffer(output));
        kernel->setArg(idx++, *(mScaleBuffer.get()));
        kernel->setArg(idx++, input->width());
        kernel->setArg(idx++, input->width()*input->height());
        kernel->setArg(idx++, inputChannelBlocks);
        kernel->setArg(idx++, UP_DIV(output->channel(), 4));
    }else{
        uint32_t idx            = 0;
        auto kernel             = &mKernel;
        kernel->setArg(idx++, mGlobalWorkSize[0]);
        kernel->setArg(idx++, mGlobalWorkSize[1]);
        kernel->setArg(idx++, mGlobalWorkSize[2]);
        kernel->setArg(idx++, openCLBuffer(input));
        kernel->setArg(idx++, *(mFilterBuffer.get()));
        kernel->setArg(idx++, *(mBiasBuffer.get()));
        kernel->setArg(idx++, openCLBuffer(output));
        kernel->setArg(idx++, *(mScaleBuffer.get()));
        kernel->setArg(idx++, sizeof(inputImageShape), inputImageShape);
        kernel->setArg(idx++, inputChannelBlocks);
        kernel->setArg(idx++, sizeof(outputImageShape), outputImageShape);
        kernel->setArg(idx++, sizeof(kernelShape), kernelShape);
        kernel->setArg(idx++, sizeof(strideShape), strideShape);
        kernel->setArg(idx++, sizeof(paddingShape), paddingShape);
        kernel->setArg(idx++, sizeof(dilationShape), dilationShape);
        kernel->setArg(idx++, UP_DIV(output->width(), 4));
        kernel->setArg(idx++, UP_DIV(output->channel(), 4));
    }

    return NO_ERROR;
}

ErrorCode ConvInt8Execution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {

    run3DKernelDefault(mKernel, mGlobalWorkSize, mLocalWorkSize, mOpenCLBackend->getOpenCLRuntime());

    return NO_ERROR;
}

class ConvInt8ExecutionCreator : public OpenCLBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new ConvInt8Execution(backend, op);
    }
};

OpenCLCreatorRegister<ConvInt8ExecutionCreator> __conv_int8_op_(OpType_ConvInt8);
}
} // namespace MNN
