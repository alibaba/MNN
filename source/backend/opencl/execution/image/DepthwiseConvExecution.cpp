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
    : ConvCommonExecution(op->main_as_Convolution2D(), backend), CommonExecution(backend, op) {
    mOpenCLBackend      = static_cast<OpenCLBackend *>(backend);
    mResource->mConv2dParams       = op->main_as_Convolution2D();
    mResource->mConv2dCommonParams = mResource->mConv2dParams->common();
    mResource->mStrides            = {mResource->mConv2dCommonParams->strideY(), mResource->mConv2dCommonParams->strideX()};
    mResource->mDilations          = {mResource->mConv2dCommonParams->dilateY(), mResource->mConv2dCommonParams->dilateX()};

    int kernelWidth   = mResource->mConv2dCommonParams->kernelX();
    int kernelHeight  = mResource->mConv2dCommonParams->kernelY();
    int outputChannel = mResource->mConv2dCommonParams->outputCount();

    std::vector<int> filterShape{1, outputChannel, kernelHeight, kernelWidth};
    std::vector<int> filterImageShape{(int)kernelHeight * kernelWidth, (int)UP_DIV(outputChannel, 4)};

        
    const float* filterDataPtr = nullptr;
    int filterDataSize   = 0;
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
    ConvolutionCommon::getConvParameters(&quanCommon, backend, mResource->mConv2dParams, &filterDataPtr, &filterDataSize);

    mResource->mFilter.reset(Tensor::createDevice<float>({1, filterImageShape[1], 1, 4 * filterImageShape[0]}));
    std::shared_ptr<Tensor> filterBuffer(Tensor::createDevice<float>(filterShape));
        
    size_t buffer_size = filterBuffer->elementSize() * sizeof(float);
    cl::Buffer filterBufferCL(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size);
    filterBuffer->buffer().device = (uint64_t)(&filterBufferCL);
    cl_int error;
    auto ptrCL = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(filterBufferCL, true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &error);
    if(ptrCL != nullptr && error == CL_SUCCESS){
        ::memcpy(ptrCL, filterDataPtr, filterBuffer->size());
    }else{
        MNN_ERROR("Map error ptrCL == nullptr \n");
    }
    mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(filterBufferCL, ptrCL);

    mOpenCLBackend->onAcquireBuffer(mResource->mFilter.get(), Backend::STATIC);

    MNN::OpenCL::ImageBufferConvertor imageBufferConvertor{mOpenCLBackend->getOpenCLRuntime()};
    std::string buildOption = "-DBUFFER_INP_FP32";
    imageBufferConvertor.convertBufferToImage(filterBuffer.get(), MNN::OpenCL::DW_CONV2D_FILTER, mResource->mFilter.get(), false, buildOption);

    if (mResource->mConv2dCommonParams->relu() == true) {
        mResource->mBuildOptions.emplace("-DRELU");
    } else if (mResource->mConv2dCommonParams->relu6() == true) {
        mResource->mBuildOptions.emplace("-DRELU6");
    }
}

DepthwiseConvExecution::~DepthwiseConvExecution() {
    // Do nothing
}

DepthwiseConvExecution::DepthwiseConvExecution(std::shared_ptr<ConvResource> resource, const MNN::Op* op, Backend *backend)
    : ConvCommonExecution(backend), CommonExecution(backend, op) {
    mResource = resource;
    const auto *conv2dParams       = op->main_as_Convolution2D();
    const auto *conv2dCommonParams = conv2dParams->common();
    mResource->mConv2dParams       = conv2dParams;
    mResource->mConv2dCommonParams  = conv2dCommonParams;
}

bool DepthwiseConvExecution::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    *dst = new DepthwiseConvExecution(mResource, op, bn);
    return true;
}

ErrorCode DepthwiseConvExecution::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mUnits.resize(1);
    auto &unit = mUnits[0];
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    auto input                   = inputs[0];
    auto output                  = outputs[0];
    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);
    
    std::string kernelName = "depthwise_conv2d";
    bool S1D1 = false;
    if (mResource->mConv2dCommonParams->strideX() == 1 && mResource->mConv2dCommonParams->strideY() == 1 &&
        mResource->mConv2dCommonParams->dilateX() == 1 && mResource->mConv2dCommonParams->dilateY() == 1) {
        kernelName = "depthwise_conv2d_s1";
        S1D1 = true;
    }
    unit.kernel       = runtime->buildKernel("depthwise_conv2d", kernelName, mResource->mBuildOptions);
    mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));

    mGlobalWorkSize = {static_cast<uint32_t>(UP_DIV(outputShape.at(3), 4) * UP_DIV(outputShape.at(2), 4)),
                       static_cast<uint32_t>(outputShape.at(0) * outputShape.at(1))};

    auto padding = ConvolutionCommon::convolutionPad(input, output, mResource->mConv2dCommonParams);
    mPaddings[0] = padding.second;//padY
    mPaddings[1] = padding.first;//padX

    const int outputHeight = outputShape.at(1);
    const int outputWidth  = outputShape.at(2);
    const int outputChannels = outputShape.at(3);

    const int inputHeight   = inputShape.at(1);
    const int inputWidth    = inputShape.at(2);
    const int inputChannels = inputShape.at(3);

    const int inputChannelBlocks = UP_DIV(inputChannels, 4);
    const int filterHeight       = mResource->mConv2dParams->common()->kernelY();
    const int filterWidth        = mResource->mConv2dParams->common()->kernelX();
    uint32_t idx                 = 0;

    int inputImageShape[2]  = {inputHeight, inputWidth};
    int outputImageShape[2] = {outputHeight, outputWidth};
    int strideShape[2]      = {mResource->mStrides[0], mResource->mStrides[1]};
    int paddingShape[2]     = {mPaddings[0], mPaddings[1]};
    int kernelShape[2]      = {filterHeight, filterWidth};
    int dilationShape[2]    = {mResource->mDilations[0], mResource->mDilations[1]};
    
    std::string info = std::to_string(inputChannels) + "_" + std::to_string(outputChannels) + "_" + std::to_string(filterHeight) + "_" + std::to_string(filterWidth) + "_" + std::to_string(mResource->mStrides[0]) + "_" + std::to_string(mResource->mStrides[1]) + std::to_string(mResource->mDilations[0]) + "_" + std::to_string(mResource->mDilations[1]);

    unit.kernel->get().setArg(idx++, mGlobalWorkSize[0]);
    unit.kernel->get().setArg(idx++, mGlobalWorkSize[1]);
    unit.kernel->get().setArg(idx++, openCLImage(input));
    unit.kernel->get().setArg(idx++, openCLImage(mResource->mFilter.get()));
    unit.kernel->get().setArg(idx++, openCLImage(mResource->mBias.get()));
    unit.kernel->get().setArg(idx++, openCLImage(output));
    unit.kernel->get().setArg(idx++, sizeof(inputImageShape), inputImageShape);
    unit.kernel->get().setArg(idx++, static_cast<int>(inputChannelBlocks));
    unit.kernel->get().setArg(idx++, sizeof(outputImageShape), outputImageShape);
    unit.kernel->get().setArg(idx++, sizeof(kernelShape), kernelShape);
    unit.kernel->get().setArg(idx++, sizeof(paddingShape), paddingShape);
    if (!S1D1) {
        unit.kernel->get().setArg(idx++, sizeof(dilationShape), dilationShape);
        unit.kernel->get().setArg(idx++, sizeof(strideShape), strideShape);
    }
    
    mLocalWorkSize = localWS2DDefault(mGlobalWorkSize, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), kernelName + info, unit.kernel).first;
    mOpenCLBackend->recordKernel2d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
    unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1]};
    unit.localWorkSize = {mLocalWorkSize[0], mLocalWorkSize[1]};
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

REGISTER_OPENCL_OP_CREATOR(DepthwiseConvolutionCreator, OpType_ConvolutionDepthwise, IMAGE);

} // namespace OpenCL
} // namespace MNN
