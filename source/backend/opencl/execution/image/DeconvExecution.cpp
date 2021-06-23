//
//  DeconvExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/image/DeconvExecution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"
#include "core/ConvolutionCommon.hpp"

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
    int outputChannel = conv2dCommonParams->outputCount();

    const float* filterDataPtr = nullptr;
    int weightSize = 0;
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
    ConvolutionCommon::getConvParameters(&quanCommon, conv2dParams, &filterDataPtr, &weightSize);

    int inputChannel  = weightSize / (kernelWidth * kernelHeight * outputChannel);
    std::vector<int> filterShape{outputChannel, inputChannel, kernelHeight, kernelWidth};
    std::vector<int> filterImageShape{(int)inputChannel, (int)UP_DIV(outputChannel, 4) * kernelWidth * kernelHeight};
    std::vector<float> filterDataPtrTransformed;
    filterDataPtrTransformed.resize(weightSize);
    IOHW2OIHW<float, int>(filterDataPtr, filterDataPtrTransformed.data(), outputChannel, inputChannel, kernelHeight,
                          kernelWidth);

    std::shared_ptr<Tensor> filterBuffer(
        Tensor::createDevice<float>({outputChannel, inputChannel, kernelHeight, kernelWidth}));
        
    int buffer_size = filterBuffer->elementSize();
    if(mOpenCLBackend->getOpenCLRuntime()->isWeightCpuTransHalf()) {
        buffer_size *= sizeof(half_float::half);
    } else {
        buffer_size *= sizeof(float);
    }
    cl::Buffer filterBufferCL(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, buffer_size);
    filterBuffer->buffer().device = (uint64_t)(&filterBufferCL);
    cl_int error;
    auto ptrCL = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(filterBufferCL, true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &error);
    if(ptrCL != nullptr && error == CL_SUCCESS){
        if(mOpenCLBackend->getOpenCLRuntime()->isWeightCpuTransHalf()){
            for(int i=0; i<filterBuffer->elementSize(); i++) {
                ((half_float::half*)ptrCL)[i] = (half_float::half)(filterDataPtrTransformed[i]);
            }
        }else{
            ::memcpy(ptrCL, filterDataPtrTransformed.data(), filterBuffer->size());
        }
    }else{
        MNN_ERROR("Map error ptrCL == nullptr \n");
    }
    mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(filterBufferCL, ptrCL);

    mFilter.reset(Tensor::createDevice<float>({1, filterImageShape[1], 1, 4 * filterImageShape[0]}));
    mOpenCLBackend->onAcquireBuffer(mFilter.get(), Backend::STATIC);
    MNN::OpenCL::ImageBufferConvertor imageBufferConvertor{mOpenCLBackend->getOpenCLRuntime()};
    
    std::string buildOption = "";
    if(mOpenCLBackend->getOpenCLRuntime()->isWeightCpuTransHalf() == false){
        buildOption = "-DBUFFER_INP_FP32";
    }
    imageBufferConvertor.convertBufferToImage(filterBuffer.get(), MNN::OpenCL::CONV2D_FILTER, mFilter.get(), false, buildOption);

    auto runtime = mOpenCLBackend->getOpenCLRuntime();
        
    std::set<std::string> buildOptions;
    std::string kernelName = "deconv_2d";
    buildOptions.emplace("-DBIAS");
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

ErrorCode DeconvExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto output = outputs[0];
    auto input  = inputs[0];

    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);

    const int outputBatch    = outputShape.at(0);
    const int outputHeight   = outputShape.at(1);
    const int outputWidth    = outputShape.at(2);
    const int outputChannels = outputShape.at(3);

    const int inputChannels = inputShape.at(3);

    const int outputChannelBlocks = UP_DIV(outputChannels, 4);
    const int strideHeight        = mStrides[0];
    const int strideWidth         = mStrides[1];
    
    auto pad = ConvolutionCommon::convolutionTransposePad(input, output, mConv2dCommonParams);
    const int paddingHeight = pad.second;
    const int paddingWidth  = pad.first;

    auto ky              = mConv2dCommonParams->kernelY();
    auto kx              = mConv2dCommonParams->kernelX();
    auto kernelSize      = kx * ky;
    
    const int transPadH  = ky - 1 - pad.second;
    const int transPadW   = kx - 1 - pad.first;
    
    const int alignHeight = mStrides[0] - 1 - transPadH;
    const int alignWidth  = mStrides[1] - 1 - transPadW;
    
    auto runtime      = mOpenCLBackend->getOpenCLRuntime();
    auto kernel       = &mKernel;
    mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel));
    mGWS              = {static_cast<uint32_t>(outputChannelBlocks), static_cast<uint32_t>(outputWidth),
            static_cast<uint32_t>(outputHeight * outputBatch)};

    int inputImageShape[2]  = {inputShape.at(1), inputShape.at(2)};
    int outputImageShape[2] = {outputHeight, outputWidth};
    int strideShape[2]      = {strideHeight, strideWidth};
    int paddingShape[2]     = {transPadH, transPadW};
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
    
    std::string name = "deconv2d";
    mLWS = localWS3DDefault(mGWS, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), name, mKernel).first;
    return NO_ERROR;
}

ErrorCode DeconvExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start DeconvExecution onExecute... \n");
#endif
    
#ifdef ENABLE_OPENCL_TIME_PROFILER
    cl::Event event;
    run3DKernelDefault(mKernel, mGWS, mLWS,
                       mOpenCLBackend->getOpenCLRuntime(),
                       &event);
    
    int costTime = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
    MNN_PRINT("kernel cost:%d    us Deconv\n",costTime);
#else
    run3DKernelDefault(mKernel, mGWS, mLWS,
                       mOpenCLBackend->getOpenCLRuntime());
#endif
    
#ifdef LOG_VERBOSE
    MNN_PRINT("End DeconvExecution onExecute... \n");
#endif
    return NO_ERROR;
}

class DeconvolutionCreator : public OpenCLBackend::Creator {
public:
    virtual ~DeconvolutionCreator() = default;
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        return new DeconvExecution(inputs, op, backend);
    }
};

OpenCLCreatorRegister<DeconvolutionCreator> __deconv_op(OpType_Deconvolution, IMAGE);

} // namespace OpenCL
} // namespace MNN
