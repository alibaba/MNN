//
//  DepthwiseConvSubgroupBufExecution.cpp
//  MNN
//
//  Created by MNN on 2023/07/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED
#ifdef MNN_SUPPORT_INTEL_SUBGROUP

#include "backend/opencl/execution/buffer/DepthwiseConvSubgroupBufExecution.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"
#include "core/ConvolutionCommon.hpp"

namespace MNN {
namespace OpenCL {

DepthwiseConvSubgroupBufExecution::DepthwiseConvSubgroupBufExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend)
    : ConvBufCommonExecution(op->main_as_Convolution2D(), backend), CommonExecution(backend, op) {
    mOpenCLBackend      = static_cast<OpenCLBackend *>(backend);
    mResource->mConv2dParams        = op->main_as_Convolution2D();
    mResource->mConv2dCommonParams = mResource->mConv2dParams->common();
    mResource->mStrides            = {mResource->mConv2dCommonParams->strideY(), mResource->mConv2dCommonParams->strideX()};
    mResource->mDilations          = {mResource->mConv2dCommonParams->dilateY(), mResource->mConv2dCommonParams->dilateX()};

    int kernelWidth   = mResource->mConv2dCommonParams->kernelX();
    int kernelHeight  = mResource->mConv2dCommonParams->kernelY();
    int outputChannel = mResource->mConv2dCommonParams->outputCount();

    {
        // create tensor for intel filter
        mResource->mFilter.reset(Tensor::createDevice<float>(std::vector<int>{1, UP_DIV(outputChannel, 16), kernelWidth * kernelHeight, 16}));
        auto res = mOpenCLBackend->onAcquireBuffer(mResource->mFilter.get(), Backend::STATIC);
        cl_int ret_code;
        if (!res) {
            mValid = false;
            return;
        }
        const float *filterDataPtr = nullptr;
        int filterDataSize         = 0;
        std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
        ConvolutionCommon::getConvParameters(&quanCommon, backend, mResource->mConv2dParams, &filterDataPtr, &filterDataSize);
        if (filterDataPtr != nullptr) {
            std::shared_ptr<Tensor> sourceWeight(Tensor::create<float>(
                std::vector<int>{1, outputChannel, kernelWidth, kernelHeight},
                                      (void *)filterDataPtr, Tensor::CAFFE));
            std::shared_ptr<Tensor> destWeight(Tensor::create<float>(std::vector<int>{1, UP_DIV(outputChannel, 16), kernelWidth * kernelHeight, 16}));

            transformWeight(destWeight.get(), sourceWeight.get());
            auto weightDestSize = destWeight->size();

            auto buffer_size = destWeight->elementSize();
            if (mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16()) {
                buffer_size *= sizeof(half_float::half);
            } else {
                buffer_size *= sizeof(float);
            }

            cl::Buffer &weightBuffer = *(cl::Buffer *)mResource->mFilter->buffer().device;

            auto runTime = mOpenCLBackend->getOpenCLRuntime();
            auto queue   = runTime->commandQueue();

            auto weight_ptr = queue.enqueueMapBuffer(weightBuffer, CL_TRUE, CL_MAP_WRITE, 0, buffer_size, nullptr,
                                                     nullptr, &ret_code);
            if (weight_ptr != nullptr && ret_code == CL_SUCCESS) {
                if (mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16()) {
                    for (int i = 0; i < destWeight->elementSize(); i++) {
                        ((half_float::half *)weight_ptr)[i] = (half_float::half)(destWeight->host<float>()[i]);
                    }
                } else {
                    ::memcpy(weight_ptr, destWeight->host<float>(), buffer_size);
                }
            } else {
                MNN_ERROR("Map error weightPtr == nullptr \n");
            }

            queue.enqueueUnmapMemObject(weightBuffer, weight_ptr);
        }
    }
    {
        int biasSize    = mResource->mConv2dParams->common()->outputCount();
        int buffer_size = ROUND_UP(biasSize, 16); // pack to 16
        if (mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16()) {
            buffer_size *= sizeof(half_float::half);
        } else {
            buffer_size *= sizeof(float);
        }

        mResource->mBias.reset(Tensor::createDevice<float>({1, 1, 1, ROUND_UP(biasSize, 16)}));
        backend->onAcquireBuffer(mResource->mBias.get(), Backend::STATIC);
        cl::Buffer &biasBuffer = openCLBuffer(mResource->mBias.get());

        cl_int res;
        auto biasPtrCL = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(
            biasBuffer, true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &res);
        if (biasPtrCL != nullptr && res == CL_SUCCESS) {
            ::memset(biasPtrCL, 0, buffer_size);
            if (nullptr != mResource->mConv2dParams->bias()) {
                const float *biasDataPtr = mResource->mConv2dParams->bias()->data();
                if (mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16()) {
                    for (int i = 0; i < biasSize; i++) {
                        ((half_float::half *)biasPtrCL)[i] = (half_float::half)(biasDataPtr[i]);
                    }
                } else {
                    ::memcpy(biasPtrCL, biasDataPtr, biasSize * sizeof(float));
                }
            }
        } else {
            MNN_ERROR("Map error biasPtrCL == nullptr \n");
        }
        mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(biasBuffer, biasPtrCL);
    }
    
    if (mResource->mConv2dCommonParams->relu() == true) {
        mResource->mBuildOptions.emplace("-DRELU");
    } else if (mResource->mConv2dCommonParams->relu6() == true) {
        mResource->mBuildOptions.emplace("-DRELU6");
    }
    int type_size = mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16() ? 2 : 4;
        mResource->mBuildOptions.emplace("-DTYPE_SIZE=" + std::to_string(type_size));
}

void DepthwiseConvSubgroupBufExecution::transformWeight(const Tensor *weightDest, const Tensor *source) {
    int co      = source->length(1);
    int KernelY = source->length(2);
    int KernelX = source->length(3);

    ::memset(weightDest->host<float>(), 0, weightDest->size());
    auto weightPtr = source->host<float>();
    for (int oz = 0; oz < co; ++oz) {
        auto src = weightPtr + oz * KernelY * KernelX;

        int ozC4 = oz / 16;
        int mx   = oz % 16;

        auto dst = weightDest->host<float>() + weightDest->stride(1) * ozC4 + mx;
        for (int i = 0; i < KernelY * KernelX; ++i) {
            *(dst + i * weightDest->stride(2)) = src[i];
        }
    }
}

DepthwiseConvSubgroupBufExecution::~DepthwiseConvSubgroupBufExecution() {
    // Do nothing
}


DepthwiseConvSubgroupBufExecution::DepthwiseConvSubgroupBufExecution(std::shared_ptr<ConvBufResource> resource, const MNN::Op* op, Backend *backend) : ConvBufCommonExecution(backend), CommonExecution(backend, op) {
    mResource = resource;
    mOpenCLBackend = static_cast<OpenCLBackend*>(backend);
    const auto *conv2dParams       = op->main_as_Convolution2D();
    const auto *conv2dCommonParams = conv2dParams->common();
    mResource->mConv2dParams       = conv2dParams;
    mResource->mConv2dCommonParams  = conv2dCommonParams;
}

bool DepthwiseConvSubgroupBufExecution::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    *dst = new DepthwiseConvSubgroupBufExecution(mResource, op, bn);
    return true;
}

ErrorCode DepthwiseConvSubgroupBufExecution::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mUnits.clear();
    auto input                   = inputs[0];
    auto output                  = outputs[0];
    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);
    auto runTime = mOpenCLBackend->getOpenCLRuntime();

    auto padding = ConvolutionCommon::convolutionPad(input, output, mResource->mConv2dCommonParams);
    mPaddings[0] = padding.second;//padY
    mPaddings[1] = padding.first;//padX
    
    const int outputHeight = outputShape.at(1);
    const int outputWidth  = outputShape.at(2);
    const int outputChannel  = outputShape.at(3);

    const int inputHeight   = inputShape.at(1);
    const int inputWidth    = inputShape.at(2);
    const int inputChannels = inputShape.at(3);

    const int inputChannelBlocks = UP_DIV(inputChannels, 4);
    const int filterHeight       = mResource->mConv2dParams->common()->kernelY();
    const int filterWidth        = mResource->mConv2dParams->common()->kernelX();
    
    int inputImageShape[2]  = {inputHeight, inputWidth};
    int outputImageShape[2] = {outputHeight, outputWidth};
    int strideShape[2]      = {mResource->mStrides[0], mResource->mStrides[1]};
    int paddingShape[2]     = {mPaddings[0], mPaddings[1]};
    int kernelShape[2]      = {filterHeight, filterWidth};
    int dilationShape[2]    = {mResource->mDilations[0], mResource->mDilations[1]};
    auto inputpad           = TensorUtils::getDescribe(input)->mPads;
    auto outputpad          = TensorUtils::getDescribe(output)->mPads;
    int input_c_pack        = TensorUtils::getTensorChannelPack(input);
    int output_c_pack       = TensorUtils::getTensorChannelPack(output);

    std::set<std::string> buildOptions = mResource->mBuildOptions;
    buildOptions.emplace("-DFILTER_HEIGHT=" + std::to_string(kernelShape[0]));
    buildOptions.emplace("-DFILTER_WIDTH=" + std::to_string(kernelShape[1]));
    buildOptions.emplace("-DDILATION_HEIGHT=" + std::to_string(dilationShape[0]));
    buildOptions.emplace("-DDILATION_WIDTH=" + std::to_string(dilationShape[1]));
    buildOptions.emplace("-DSTRIDE_HEIGHT=" + std::to_string(strideShape[0]));
    buildOptions.emplace("-DSTRIDE_WIDTH=" + std::to_string(strideShape[1]));
    if (input_c_pack == 4) {
        Unit unit;
        mNeedTranse = true;
        mSource.reset(Tensor::createDevice<float>(std::vector<int>{inputShape.at(0), UP_DIV(input->channel(), 16), inputHeight * (inputWidth + inputpad.left + inputpad.right), 16}, Tensor::CAFFE_C4));
        mOpenCLBackend->onAcquireBuffer(mSource.get(), Backend::DYNAMIC);
        mOpenCLBackend->onReleaseBuffer(mSource.get(), Backend::DYNAMIC);
        unit.kernel =
            mOpenCLBackend->getOpenCLRuntime()->buildKernel("input_transe_buf", "conv_transe_c4_c16", {});

        uint32_t mMaxWGS_S =
            static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(unit.kernel));

        mTranseGlobalWorkSize = {static_cast<uint32_t>(inputWidth * inputHeight),
                                 static_cast<uint32_t>(UP_DIV(inputShape.at(3), 4)),
                                 static_cast<uint32_t>(inputShape.at(0))};
        uint32_t idx          = 0;
        unit.kernel->get().setArg(idx++, mTranseGlobalWorkSize[0]);
        unit.kernel->get().setArg(idx++, mTranseGlobalWorkSize[1]);
        unit.kernel->get().setArg(idx++, mTranseGlobalWorkSize[2]);
        unit.kernel->get().setArg(idx++, openCLBuffer(input));
        unit.kernel->get().setArg(idx++, openCLBuffer(mSource.get()));
        unit.kernel->get().setArg(idx++, static_cast<uint32_t>(inputWidth));
        unit.kernel->get().setArg(idx++, static_cast<uint32_t>(inputHeight));
        unit.kernel->get().setArg(idx++, static_cast<uint32_t>(inputChannels));
        unit.kernel->get().setArg(idx++, UP_DIV(inputShape.at(3), 4));
        unit.kernel->get().setArg(idx++, static_cast<uint32_t>(inputpad.left));
        unit.kernel->get().setArg(idx++, static_cast<uint32_t>(inputpad.right));

        mTranseLocalWorkSize = localWS3DDefault(mTranseGlobalWorkSize, mMaxWGS_S, mOpenCLBackend->getOpenCLRuntime(),"conv_transe_c4_c16", unit.kernel).first;
        mOpenCLBackend->recordKernel3d(unit.kernel, mTranseGlobalWorkSize, mTranseLocalWorkSize);
        unit.globalWorkSize = {mTranseGlobalWorkSize[0], mTranseGlobalWorkSize[1], mTranseGlobalWorkSize[2]};
        unit.localWorkSize = {mTranseLocalWorkSize[0], mTranseLocalWorkSize[1], mTranseLocalWorkSize[2]};
        mUnits.emplace_back(unit);
    }
    Unit unit;
    mGlobalWorkSize = {static_cast<uint32_t>(UP_DIV(outputShape.at(2), 8) * outputShape.at(1)),
                       static_cast<uint32_t>(ROUND_UP(outputShape.at(3), 16)),
                       static_cast<uint32_t>(outputShape.at(0))};
    mLocalWorkSize  = {1, 16, 1};


    std::string kernelname = "depthwise_conv_2d_buf_c16_c" + std::to_string(output_c_pack);
    unit.kernel = mOpenCLBackend->getOpenCLRuntime()->buildKernel("depthwise_conv2d_subgroup_buf", kernelname, buildOptions);
    uint32_t idx = 0;
    if (mNeedTranse) {
        unit.kernel->get().setArg(idx++, openCLBuffer(mSource.get()));
    }
    else {
        unit.kernel->get().setArg(idx++, openCLBuffer(input));
    }
    unit.kernel->get().setArg(idx++, openCLBuffer(output));
    unit.kernel->get().setArg(idx++, openCLBuffer(mResource->mFilter.get()));
    unit.kernel->get().setArg(idx++, openCLBuffer(mResource->mBias.get()));
    unit.kernel->get().setArg(idx++, static_cast<uint32_t>(inputHeight));
    unit.kernel->get().setArg(idx++, static_cast<uint32_t>(inputWidth));
    unit.kernel->get().setArg(idx++, static_cast<uint32_t>(inputChannels));
    unit.kernel->get().setArg(idx++, static_cast<uint32_t>(inputpad.left));
    unit.kernel->get().setArg(idx++, static_cast<uint32_t>(inputpad.right));
    unit.kernel->get().setArg(idx++, static_cast<uint32_t>(outputHeight));
    unit.kernel->get().setArg(idx++, static_cast<uint32_t>(outputWidth));
    unit.kernel->get().setArg(idx++, static_cast<uint32_t>(outputpad.left));
    unit.kernel->get().setArg(idx++, static_cast<uint32_t>(outputpad.right));
    unit.kernel->get().setArg(idx++, static_cast<uint32_t>(paddingShape[1]));
    unit.kernel->get().setArg(idx++, static_cast<uint32_t>(paddingShape[0]));
    
    mOpenCLBackend->recordKernel3d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
    unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
    unit.localWorkSize = {mLocalWorkSize[0], mLocalWorkSize[1], mLocalWorkSize[2]};
    mUnits.emplace_back(unit);
    return NO_ERROR;
}

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_SUPPORT_INTEL_SUBGROUP */
#endif /* MNN_OPENCL_BUFFER_CLOSED */
