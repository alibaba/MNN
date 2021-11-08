//
//  DepthwiseDeconvExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/image/DepthwiseDeconvExecution.hpp"
#include "backend/opencl/execution/image/MultiInputDWDeconvExecution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "core/ConvolutionCommon.hpp"

namespace MNN {
namespace OpenCL {

DepthwiseDeconvExecution::DepthwiseDeconvExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op,
                                                   Backend *backend)
    : ConvCommonExecution(op->main_as_Convolution2D(), backend) {
    mOpenCLBackend      = static_cast<OpenCLBackend *>(backend);
    mCon2dParams        = op->main_as_Convolution2D();
    mConv2dCommonParams = mCon2dParams->common();
    mStrides            = {mConv2dCommonParams->strideY(), mConv2dCommonParams->strideX()};
    mDilations          = {mConv2dCommonParams->dilateY(), mConv2dCommonParams->dilateX()};

    MNN_ASSERT(mStrides[0] > 0 && mStrides[1] > 0);

    int kernelWidth   = mConv2dCommonParams->kernelX();
    int kernelHeight  = mConv2dCommonParams->kernelY();
    int outputChannel = mConv2dCommonParams->outputCount();

    std::vector<int> filterShape{1, outputChannel, kernelHeight, kernelWidth};
    std::vector<int> filterImageShape{(int)kernelHeight * kernelWidth, (int)UP_DIV(outputChannel, 4)};

    const float* filterDataPtr = nullptr;
    int tempWeightSize   = 0;
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
    ConvolutionCommon::getConvParameters(&quanCommon, mCon2dParams, &filterDataPtr, &tempWeightSize);

    mFilter.reset(Tensor::createDevice<float>({1, filterImageShape[1], 1, 4 * filterImageShape[0]}));
    std::shared_ptr<Tensor> filterBuffer(Tensor::createDevice<float>(filterShape));
        
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
    if(nullptr != ptrCL && error == CL_SUCCESS){
        if(mOpenCLBackend->getOpenCLRuntime()->isWeightCpuTransHalf()){
            for(int i=0; i<filterBuffer->elementSize(); i++) {
                ((half_float::half*)ptrCL)[i] = (half_float::half)(filterDataPtr[i]);
            }
        }else{
            ::memcpy(ptrCL, filterDataPtr, filterBuffer->size());
        }
    }else{
        MNN_ERROR("Map error ptrCL == nullptr \n");
    }
    mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(filterBufferCL, ptrCL);
    mOpenCLBackend->onAcquireBuffer(mFilter.get(), Backend::STATIC);

    MNN::OpenCL::ImageBufferConvertor imageBufferConvertor{mOpenCLBackend->getOpenCLRuntime()};
    std::string buildOption = "";
    if(mOpenCLBackend->getOpenCLRuntime()->isWeightCpuTransHalf() == false){
        buildOption = "-DBUFFER_INP_FP32";
    }
    imageBufferConvertor.convertBufferToImage(filterBuffer.get(), MNN::OpenCL::DW_CONV2D_FILTER, mFilter.get(), false, buildOption);
    std::set<std::string> buildOptions;
    std::string kernelName = "depthwise_deconv2d";
    if (mConv2dCommonParams->relu() == true) {
        buildOptions.emplace("-DRELU");
    } else if (mConv2dCommonParams->relu6() == true) {
        buildOptions.emplace("-DRELU6");
    }
    auto runtime = mOpenCLBackend->getOpenCLRuntime();

    mKernel           = runtime->buildKernel("depthwise_deconv2d", kernelName, buildOptions);
    mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel));
}
DepthwiseDeconvExecution::~DepthwiseDeconvExecution() {
    mOpenCLBackend->onReleaseBuffer(mFilter.get(), Backend::STATIC);
}
ErrorCode DepthwiseDeconvExecution::onResize(const std::vector<Tensor *> &inputs,
                                             const std::vector<Tensor *> &outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];

    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);

    const int outputBatch    = outputShape.at(0);
    const int outputHeight   = outputShape.at(1);
    const int outputWidth    = outputShape.at(2);
    const int outputChannels = outputShape.at(3);

    const int inputHeight   = inputShape.at(1);
    const int inputWidth    = inputShape.at(2);
    const int inputChannels = inputShape.at(3);

    const int strideHeight = mStrides[0];
    const int strideWidth  = mStrides[1];

    const int channelBlocks = UP_DIV(outputChannels, 4);

    auto pad = ConvolutionCommon::convolutionTransposePad(input, output, mConv2dCommonParams);
    const int paddingHeight = pad.second;
    const int paddingWidth  = pad.first;

    const int alignHeight = strideHeight - 1 - paddingHeight;
    const int alignWidth  = strideWidth - 1 - paddingWidth;

    const int filterHeight = mConv2dCommonParams->kernelY();
    const int filterWidth  = mConv2dCommonParams->kernelX();
    const int kernelSize   = filterHeight * filterWidth;

    mGWS        = {static_cast<uint32_t>(channelBlocks), static_cast<uint32_t>(outputWidth),
            static_cast<uint32_t>(outputHeight * outputBatch)};
    auto kernel = &mKernel;

    int inputImageShape[2]  = {inputHeight, inputWidth};
    int outputImageShape[2] = {outputHeight, outputWidth};
    int strideShape[2]      = {strideHeight, strideWidth};
    int paddingShape[2]     = {paddingHeight, paddingWidth};
    int alignShape[2]       = {alignHeight, alignWidth};
    int kernelShape[2]      = {filterHeight, filterWidth};

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
    kernel->setArg(idx++, static_cast<int32_t>(channelBlocks));
    
    std::string name = "depthwiseDeconv";
    mLWS = localWS3DDefault(mGWS, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), name, mKernel).first;

    return NO_ERROR;
}

ErrorCode DepthwiseDeconvExecution::onExecute(const std::vector<Tensor *> &inputs,
                                              const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start DepthwiseDeconvExecution onExecute !\n");
#endif
    
#ifdef ENABLE_OPENCL_TIME_PROFILER
    cl::Event event;
    run3DKernelDefault(mKernel, mGWS, mLWS,
                       mOpenCLBackend->getOpenCLRuntime(),
                       &event);
    
    int costTime = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
    MNN_PRINT("kernel cost:%d    us DepthwiseDeconv\n",costTime);
#else
    run3DKernelDefault(mKernel, mGWS, mLWS,
                       mOpenCLBackend->getOpenCLRuntime());
#endif
    
#ifdef LOG_VERBOSE
    MNN_PRINT("End DepthwiseDeconvExecution onExecute !\n");
#endif
    return NO_ERROR;
}


class DepthwiseDeconvolutionCreator : public OpenCLBackend::Creator {
public:
    virtual ~DepthwiseDeconvolutionCreator() = default;
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        
        MNN_ASSERT(inputs.size() <= 3);
        if (inputs.size() == 2 || inputs.size() == 3) {
            return new MultiInputDWDeconvExecution(op, backend);
        }
        
        MNN_ASSERT(inputs.size() == 1);
        return new DepthwiseDeconvExecution(inputs, op, backend);
    }
};

OpenCLCreatorRegister<DepthwiseDeconvolutionCreator> __DepthwiseDeconv_op(OpType_DeconvolutionDepthwise, IMAGE);

} // namespace OpenCL
} // namespace MNN
