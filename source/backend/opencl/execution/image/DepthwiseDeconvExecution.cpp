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
    : ConvCommonExecution(op->main_as_Convolution2D(), backend), CommonExecution(backend, op){
    mResource->mConv2dParams        = op->main_as_Convolution2D();
    mResource->mConv2dCommonParams = mResource->mConv2dParams->common();
    mResource->mStrides            = {mResource->mConv2dCommonParams->strideY(), mResource->mConv2dCommonParams->strideX()};
    mResource->mDilations          = {mResource->mConv2dCommonParams->dilateY(), mResource->mConv2dCommonParams->dilateX()};

    int kernelWidth   = mResource->mConv2dCommonParams->kernelX();
    int kernelHeight  = mResource->mConv2dCommonParams->kernelY();
    int outputChannel = mResource->mConv2dCommonParams->outputCount();

    std::vector<int> filterShape{1, outputChannel, kernelHeight, kernelWidth};
    std::vector<int> filterImageShape{(int)kernelHeight * kernelWidth, (int)UP_DIV(outputChannel, 4)};

    const float* filterDataPtr = nullptr;
    int tempWeightSize   = 0;
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
    ConvolutionCommon::getConvParameters(&quanCommon, backend, mResource->mConv2dParams, &filterDataPtr, &tempWeightSize);

        mResource->mFilter.reset(Tensor::createDevice<float>({1, filterImageShape[1], 1, 4 * filterImageShape[0]}));
    std::shared_ptr<Tensor> filterBuffer(Tensor::createDevice<float>(filterShape));
        
    size_t buffer_size = filterBuffer->elementSize() * sizeof(float);
    cl::Buffer filterBufferCL(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, buffer_size);
    filterBuffer->buffer().device = (uint64_t)(&filterBufferCL);
    cl_int error;
    auto ptrCL = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(filterBufferCL, true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &error);
    if(nullptr != ptrCL && error == CL_SUCCESS){
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
DepthwiseDeconvExecution::~DepthwiseDeconvExecution() {
    // Do nothing
}
DepthwiseDeconvExecution::DepthwiseDeconvExecution(std::shared_ptr<ConvResource> resource, const MNN::Op* op, Backend *backend)
    : ConvCommonExecution(backend), CommonExecution(backend, op) {
    mResource = resource;
    const auto *conv2dParams       = op->main_as_Convolution2D();
    const auto *conv2dCommonParams = conv2dParams->common();
    mResource->mConv2dParams       = conv2dParams;
    mResource->mConv2dCommonParams  = conv2dCommonParams;
}

bool DepthwiseDeconvExecution::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    *dst = new DepthwiseDeconvExecution(mResource, op, bn);
    return true;
}
ErrorCode DepthwiseDeconvExecution::onEncode(const std::vector<Tensor *> &inputs,
                                             const std::vector<Tensor *> &outputs) {
    mUnits.resize(1);
    auto &unit  = mUnits[0];
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

    const int strideHeight = mResource->mStrides[0];
    const int strideWidth  = mResource->mStrides[1];

    const int channelBlocks = UP_DIV(outputChannels, 4);

    auto pad = ConvolutionCommon::convolutionTransposePad(input, output, mResource->mConv2dCommonParams);
    const int paddingHeight = pad.second;
    const int paddingWidth  = pad.first;

    const int alignHeight = strideHeight - 1 - paddingHeight;
    const int alignWidth  = strideWidth - 1 - paddingWidth;

    const int filterHeight = mResource->mConv2dCommonParams->kernelY();
    const int filterWidth  = mResource->mConv2dCommonParams->kernelX();
    const int kernelSize   = filterHeight * filterWidth;

    mGWS        = {static_cast<uint32_t>(channelBlocks), static_cast<uint32_t>(outputWidth),
            static_cast<uint32_t>(outputHeight * outputBatch)};
    std::string info = std::to_string(inputChannels) + "_" + std::to_string(outputChannels) + "_" + std::to_string(filterHeight) + "_" + std::to_string(filterWidth) + "_" + std::to_string(strideHeight) + "_" + std::to_string(strideWidth);
    auto runtime      = mOpenCLBackend->getOpenCLRuntime();
    unit.kernel       = runtime->buildKernel("depthwise_deconv2d", "depthwise_deconv2d", mResource->mBuildOptions);
    mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));

    int inputImageShape[2]  = {inputHeight, inputWidth};
    int outputImageShape[2] = {outputHeight, outputWidth};
    int strideShape[2]      = {strideHeight, strideWidth};
    int paddingShape[2]     = {paddingHeight, paddingWidth};
    int alignShape[2]       = {alignHeight, alignWidth};
    int kernelShape[2]      = {filterHeight, filterWidth};

    uint32_t idx = 0;
    unit.kernel->get().setArg(idx++, mGWS[0]);
    unit.kernel->get().setArg(idx++, mGWS[1]);
    unit.kernel->get().setArg(idx++, mGWS[2]);

    unit.kernel->get().setArg(idx++, openCLImage(input));
    unit.kernel->get().setArg(idx++, openCLImage(mResource->mFilter.get()));
    unit.kernel->get().setArg(idx++, openCLImage(mResource->mBias.get()));
    unit.kernel->get().setArg(idx++, openCLImage(output));
    unit.kernel->get().setArg(idx++, sizeof(inputImageShape), inputImageShape);
    unit.kernel->get().setArg(idx++, sizeof(outputImageShape), outputImageShape);
    unit.kernel->get().setArg(idx++, sizeof(strideShape), strideShape);
    unit.kernel->get().setArg(idx++, sizeof(alignShape), alignShape);
    unit.kernel->get().setArg(idx++, sizeof(paddingShape), paddingShape);
    unit.kernel->get().setArg(idx++, sizeof(kernelShape), kernelShape);
    unit.kernel->get().setArg(idx++, static_cast<int32_t>(kernelSize));
    unit.kernel->get().setArg(idx++, static_cast<int32_t>(channelBlocks));
    
    std::string name = "depthwiseDeconv";
    mLWS = localWS3DDefault(mGWS, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), name + info, unit.kernel).first;
    mOpenCLBackend->recordKernel3d(unit.kernel, mGWS, mLWS);
    unit.globalWorkSize = {mGWS[0], mGWS[1], mGWS[2]};
    unit.localWorkSize = {mLWS[0], mLWS[1], mLWS[2]};
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

REGISTER_OPENCL_OP_CREATOR(DepthwiseDeconvolutionCreator, OpType_DeconvolutionDepthwise, IMAGE);

} // namespace OpenCL
} // namespace MNN
