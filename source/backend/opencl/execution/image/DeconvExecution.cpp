//
//  DeconvExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/image/DeconvExecution.hpp"
#include "core/ConvolutionCommon.hpp"

namespace MNN {
namespace OpenCL {

DeconvExecution::DeconvExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend)
    : ConvCommonExecution(op->main_as_Convolution2D(), backend), CommonExecution(backend, op) {
    mOpenCLBackend                 = static_cast<OpenCLBackend *>(backend);
    const auto *conv2dParams       = op->main_as_Convolution2D();
    const auto *conv2dCommonParams = conv2dParams->common();
    mResource->mConv2dCommonParams = conv2dCommonParams;
    mResource->mStrides            = {conv2dCommonParams->strideY(), conv2dCommonParams->strideX()};
    mResource->mDilations          = {conv2dCommonParams->dilateY(), conv2dCommonParams->dilateX()};
    int kernelWidth                = conv2dCommonParams->kernelX();
    int kernelHeight               = conv2dCommonParams->kernelY();

    int outputChannel = conv2dCommonParams->outputCount();

    const float* filterDataPtr = nullptr;
    int weightSize = 0;
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
    ConvolutionCommon::getConvParameters(&quanCommon, backend, conv2dParams, &filterDataPtr, &weightSize);

    int inputChannel  = weightSize / (kernelWidth * kernelHeight * outputChannel);
    std::vector<int> filterShape{outputChannel, inputChannel, kernelHeight, kernelWidth};
    std::vector<int> filterImageShape{(int)inputChannel, (int)UP_DIV(outputChannel, 4) * kernelWidth * kernelHeight};
    std::vector<float> filterDataPtrTransformed;
    filterDataPtrTransformed.resize(weightSize);
    IOHW2OIHW<float, int>(filterDataPtr, filterDataPtrTransformed.data(), outputChannel, inputChannel, kernelHeight,
                          kernelWidth);

    std::shared_ptr<Tensor> filterBuffer(
        Tensor::createDevice<float>({outputChannel, inputChannel, kernelHeight, kernelWidth}));
        
    size_t buffer_size = filterBuffer->elementSize() * sizeof(float);
    cl::Buffer filterBufferCL(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, buffer_size);
    filterBuffer->buffer().device = (uint64_t)(&filterBufferCL);
    cl_int error;
    auto ptrCL = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(filterBufferCL, true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &error);
    if(ptrCL != nullptr && error == CL_SUCCESS){
        ::memcpy(ptrCL, filterDataPtrTransformed.data(), filterBuffer->size());
    }else{
        MNN_ERROR("Map error ptrCL == nullptr \n");
    }
    mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(filterBufferCL, ptrCL);

        mResource->mFilter.reset(Tensor::createDevice<float>({1, filterImageShape[1], 1, 4 * filterImageShape[0]}));
    mOpenCLBackend->onAcquireBuffer(mResource->mFilter.get(), Backend::STATIC);
    MNN::OpenCL::ImageBufferConvertor imageBufferConvertor{mOpenCLBackend->getOpenCLRuntime()};
    
    std::string buildOption = "-DBUFFER_INP_FP32";
    imageBufferConvertor.convertBufferToImage(filterBuffer.get(), MNN::OpenCL::CONV2D_FILTER, mResource->mFilter.get(), false, buildOption);
        
    mResource->mBuildOptions.emplace("-DBIAS");
    if (conv2dCommonParams->relu() == true) {
        mResource->mBuildOptions.emplace("-DRELU");
    } else if (conv2dCommonParams->relu6() == true) {
        mResource->mBuildOptions.emplace("-DRELU6");
    }
}

DeconvExecution::~DeconvExecution() {
    // Do nothing
}

DeconvExecution::DeconvExecution(std::shared_ptr<ConvResource> resource, const MNN::Op* op, Backend *backend)
    : ConvCommonExecution(backend), CommonExecution(backend, op) {
    mResource = resource;
    const auto *conv2dParams       = op->main_as_Convolution2D();
    const auto *conv2dCommonParams = conv2dParams->common();
    mResource->mConv2dParams       = conv2dParams;
    mResource->mConv2dCommonParams  = conv2dCommonParams;
}

bool DeconvExecution::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    *dst = new DeconvExecution(mResource, op, bn);
    return true;
}

ErrorCode DeconvExecution::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mUnits.resize(1);
    auto &unit = mUnits[0];
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
    const int strideHeight        = mResource->mStrides[0];
    const int strideWidth         = mResource->mStrides[1];
    
    auto pad = ConvolutionCommon::convolutionTransposePad(input, output, mResource->mConv2dCommonParams);
    const int paddingHeight = pad.second;
    const int paddingWidth  = pad.first;

    auto ky              = mResource->mConv2dCommonParams->kernelY();
    auto kx              = mResource->mConv2dCommonParams->kernelX();
    auto kernelSize      = kx * ky;
    
    const int transPadH  = ky - 1 - pad.second;
    const int transPadW   = kx - 1 - pad.first;
    
    const int alignHeight = mResource->mStrides[0] - 1 - transPadH;
    const int alignWidth  = mResource->mStrides[1] - 1 - transPadW;
    
    auto runtime      = mOpenCLBackend->getOpenCLRuntime();
    unit.kernel = runtime->buildKernel("deconv_2d", "deconv_2d", mResource->mBuildOptions);
    auto maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));
    mGWS              = {static_cast<uint32_t>(outputChannelBlocks), static_cast<uint32_t>(outputWidth),
            static_cast<uint32_t>(outputHeight * outputBatch)};

    int inputImageShape[2]  = {inputShape.at(1), inputShape.at(2)};
    int outputImageShape[2] = {outputHeight, outputWidth};
    int strideShape[2]      = {strideHeight, strideWidth};
    int paddingShape[2]     = {transPadH, transPadW};
    int alignShape[2]       = {alignHeight, alignWidth};
    int kernelShape[2]      = {ky, kx};

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
    unit.kernel->get().setArg(idx++, static_cast<int32_t>(UP_DIV(inputChannels, 4)));
    unit.kernel->get().setArg(idx++, static_cast<int32_t>(outputChannelBlocks));
    
    std::string name = "deconv2d";
    std::string info = std::to_string(inputChannels) + "_" + std::to_string(outputChannels) + "_" + std::to_string(ky) + "_" + std::to_string(kx) + "_" + std::to_string(strideHeight) + "_" + std::to_string(strideWidth);
    mLWS = localWS3DDefault(mGWS, maxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), name + info, unit.kernel).first;
    mOpenCLBackend->recordKernel3d(unit.kernel, mGWS, mLWS);
    unit.globalWorkSize = {mGWS[0], mGWS[1], mGWS[2]};
    unit.localWorkSize = {mLWS[0], mLWS[1], mLWS[2]};
    return NO_ERROR;
}

class DeconvolutionCreator : public OpenCLBackend::Creator {
public:
    virtual ~DeconvolutionCreator() = default;
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        if(inputs.size() != 1){
            return nullptr;
        }
        return new DeconvExecution(inputs, op, backend);
    }
};

REGISTER_OPENCL_OP_CREATOR(DeconvolutionCreator, OpType_Deconvolution, IMAGE);
} // namespace OpenCL
} // namespace MNN
