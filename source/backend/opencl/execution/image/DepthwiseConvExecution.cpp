//
//  DepthwiseConvExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/image/DepthwiseConvExecution.hpp"
#include "backend/opencl/execution/image/MultiInputDWConvExecution.hpp"
#include "core/Macro.h"
#include <string.h>
#include "core/TensorUtils.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"
#include "core/ConvolutionCommon.hpp"

namespace MNN {
namespace OpenCL {


DepthwiseConvExecution::DepthwiseConvExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend)
    : ConvCommonExecution(op->main_as_Convolution2D(), backend) {
    mOpenCLBackend      = static_cast<OpenCLBackend *>(backend);
    mCon2dParams        = op->main_as_Convolution2D();
    mConv2dCommonParams = mCon2dParams->common();
    mStrides            = {mConv2dCommonParams->strideY(), mConv2dCommonParams->strideX()};
    mDilations          = {mConv2dCommonParams->dilateY(), mConv2dCommonParams->dilateX()};

    int kernelWidth   = mConv2dCommonParams->kernelX();
    int kernelHeight  = mConv2dCommonParams->kernelY();
    int outputChannel = mConv2dCommonParams->outputCount();

    std::vector<int> filterShape{1, outputChannel, kernelHeight, kernelWidth};
    std::vector<int> filterImageShape{(int)kernelHeight * kernelWidth, (int)UP_DIV(outputChannel, 4)};

        
    const float* filterDataPtr = nullptr;
    int filterDataSize   = 0;
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
    ConvolutionCommon::getConvParameters(&quanCommon, mCon2dParams, &filterDataPtr, &filterDataSize);

    mFilter.reset(Tensor::createDevice<float>({1, filterImageShape[1], 1, 4 * filterImageShape[0]}));
    std::shared_ptr<Tensor> filterBuffer(Tensor::createDevice<float>(filterShape));
        
    int buffer_size = filterBuffer->elementSize();
    if(mOpenCLBackend->getOpenCLRuntime()->isWeightCpuTransHalf()) {
        buffer_size *= sizeof(half_float::half);
    } else {
        buffer_size *= sizeof(float);
    }
    cl::Buffer filterBufferCL(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size);
    filterBuffer->buffer().device = (uint64_t)(&filterBufferCL);
    cl_int error;
    auto ptrCL = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(filterBufferCL, true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &error);
    if(ptrCL != nullptr && error == CL_SUCCESS){
        if(mOpenCLBackend->getOpenCLRuntime()->isWeightCpuTransHalf()){
            for (int i = 0; i < filterBuffer->elementSize(); i++) {
                ((half_float::half *)ptrCL)[i] = (half_float::half)(filterDataPtr[i]);
            }
        } else {
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

    auto padding = ConvolutionCommon::convolutionPad(input, output, mConv2dCommonParams);
    mPaddings[0] = padding.second;//padY
    mPaddings[1] = padding.first;//padX

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
    int paddingShape[2]     = {mPaddings[0], mPaddings[1]};
    int kernelShape[2]      = {filterHeight, filterWidth};
    int dilationShape[2]    = {mDilations[0], mDilations[1]};

    std::string kernelName = "depthwise_conv2d_s1";

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
        kernelName = "depthwise_conv2d";
    }
    
    mLocalWorkSize = localWS2DDefault(mGlobalWorkSize, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), kernelName, mKernel).first;
    return NO_ERROR;
}

ErrorCode DepthwiseConvExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start DepthwiseConvExecution onExecute !\n");
#endif

#ifdef ENABLE_OPENCL_TIME_PROFILER
    cl::Event event;
    runKernel2D(mKernel, mGlobalWorkSize, mLocalWorkSize,
                mOpenCLBackend->getOpenCLRuntime(),
                &event);
    
    int costTime = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
    MNN_PRINT("kernel cost:%d    us DepthwiseConv\n",costTime);
#else
    runKernel2D(mKernel, mGlobalWorkSize, mLocalWorkSize,
                mOpenCLBackend->getOpenCLRuntime());
#endif

#ifdef LOG_VERBOSE
    MNN_PRINT("end DepthwiseConvExecution onExecute !\n");
#endif
    return NO_ERROR;
}

class DepthwiseConvolutionCreator : public OpenCLBackend::Creator {
public:
    virtual ~DepthwiseConvolutionCreator() = default;
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        
        MNN_ASSERT(inputs.size() <= 3);
        if (inputs.size() == 2 || inputs.size() == 3) {
            return new MultiInputDWConvExecution(op, backend);
        }
        
        MNN_ASSERT(inputs.size() == 1);
        return new DepthwiseConvExecution(inputs, op, backend);
    }
};

OpenCLCreatorRegister<DepthwiseConvolutionCreator> __DepthwiseConv_op(OpType_ConvolutionDepthwise, IMAGE);

} // namespace OpenCL
} // namespace MNN
