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
    : ConvBufCommonExecution(op->main_as_Convolution2D(), backend) {
    mOpenCLBackend      = static_cast<OpenCLBackend *>(backend);
    mCon2dParams        = op->main_as_Convolution2D();
    mConv2dCommonParams = mCon2dParams->common();
    mStrides            = {mConv2dCommonParams->strideY(), mConv2dCommonParams->strideX()};
    mDilations          = {mConv2dCommonParams->dilateY(), mConv2dCommonParams->dilateX()};

    int kernelWidth   = mConv2dCommonParams->kernelX();
    int kernelHeight  = mConv2dCommonParams->kernelY();
    int outputChannel = mConv2dCommonParams->outputCount();

    {
        // create tensor for intel filter
        mFilter.reset(Tensor::createDevice<float>(std::vector<int>{1, UP_DIV(outputChannel, 16), kernelWidth * kernelHeight, 16}));
        auto res = mOpenCLBackend->onAcquireBuffer(mFilter.get(), Backend::STATIC);
        cl_int ret_code;
        if (!res) {
            mValid = false;
            return;
        }
        const float *filterDataPtr = nullptr;
        int filterDataSize         = 0;
        std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
        ConvolutionCommon::getConvParameters(&quanCommon, backend, mCon2dParams, &filterDataPtr, &filterDataSize);
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

            cl::Buffer &weightBuffer = *(cl::Buffer *)mFilter->buffer().device;

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
        int biasSize    = mCon2dParams->common()->outputCount();
        int buffer_size = ROUND_UP(biasSize, 16); // pack to 16
        if (mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16()) {
            buffer_size *= sizeof(half_float::half);
        } else {
            buffer_size *= sizeof(float);
        }

        mBias.reset(Tensor::createDevice<float>({1, 1, 1, ROUND_UP(biasSize, 16)}));
        backend->onAcquireBuffer(mBias.get(), Backend::STATIC);
        cl::Buffer &biasBuffer = openCLBuffer(mBias.get());

        cl_int res;
        auto biasPtrCL = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(
            biasBuffer, true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &res);
        if (biasPtrCL != nullptr && res == CL_SUCCESS) {
            ::memset(biasPtrCL, 0, buffer_size);
            if (nullptr != mCon2dParams->bias()) {
                const float *biasDataPtr = mCon2dParams->bias()->data();
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
    
    if (mConv2dCommonParams->relu() == true) {
        mBuildOptions.emplace("-DRELU");
    } else if (mConv2dCommonParams->relu6() == true) {
        mBuildOptions.emplace("-DRELU6");
    }
    int type_size = mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16() ? 2 : 4;
    mBuildOptions.emplace("-DTYPE_SIZE=" + std::to_string(type_size));
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
    mOpenCLBackend->onReleaseBuffer(mFilter.get(), Backend::STATIC);
    mOpenCLBackend->onReleaseBuffer(mBias.get(), Backend::STATIC);
}

ErrorCode DepthwiseConvSubgroupBufExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input                   = inputs[0];
    auto output                  = outputs[0];
    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);

    auto padding = ConvolutionCommon::convolutionPad(input, output, mConv2dCommonParams);
    mPaddings[0] = padding.second;//padY
    mPaddings[1] = padding.first;//padX
    
    const int outputHeight = outputShape.at(1);
    const int outputWidth  = outputShape.at(2);
    const int outputChannel  = outputShape.at(3);

    const int inputHeight   = inputShape.at(1);
    const int inputWidth    = inputShape.at(2);
    const int inputChannels = inputShape.at(3);

    const int inputChannelBlocks = UP_DIV(inputChannels, 4);
    const int filterHeight       = mCon2dParams->common()->kernelY();
    const int filterWidth        = mCon2dParams->common()->kernelX();
    
    int inputImageShape[2]  = {inputHeight, inputWidth};
    int outputImageShape[2] = {outputHeight, outputWidth};
    int strideShape[2]      = {mStrides[0], mStrides[1]};
    int paddingShape[2]     = {mPaddings[0], mPaddings[1]};
    int kernelShape[2]      = {filterHeight, filterWidth};
    int dilationShape[2]    = {mDilations[0], mDilations[1]};
    auto inputpad           = TensorUtils::getDescribe(input)->mPads;
    auto outputpad          = TensorUtils::getDescribe(output)->mPads;
    int input_c_pack        = TensorUtils::getTensorChannelPack(input);
    int output_c_pack       = TensorUtils::getTensorChannelPack(output);

    std::set<std::string> buildOptions = mBuildOptions;
    buildOptions.emplace("-DFILTER_HEIGHT=" + std::to_string(kernelShape[0]));
    buildOptions.emplace("-DFILTER_WIDTH=" + std::to_string(kernelShape[1]));
    buildOptions.emplace("-DDILATION_HEIGHT=" + std::to_string(dilationShape[0]));
    buildOptions.emplace("-DDILATION_WIDTH=" + std::to_string(dilationShape[1]));
    buildOptions.emplace("-DSTRIDE_HEIGHT=" + std::to_string(strideShape[0]));
    buildOptions.emplace("-DSTRIDE_WIDTH=" + std::to_string(strideShape[1]));
    if (input_c_pack == 4) {
        mNeedTranse = true;
        mSource.reset(Tensor::createDevice<float>(std::vector<int>{inputShape.at(0), UP_DIV(input->channel(), 16), inputHeight * (inputWidth + inputpad.left + inputpad.right), 16}, Tensor::CAFFE_C4));
        mOpenCLBackend->onAcquireBuffer(mSource.get(), Backend::DYNAMIC);
        mOpenCLBackend->onReleaseBuffer(mSource.get(), Backend::DYNAMIC);
        mTranseKernel =
            mOpenCLBackend->getOpenCLRuntime()->buildKernel("input_transe_buf", "conv_transe_c4_c16", {});

        uint32_t mMaxWGS_S =
            static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(mTranseKernel));

        mTranseGlobalWorkSize = {static_cast<uint32_t>(inputWidth * inputHeight),
                                 static_cast<uint32_t>(UP_DIV(inputShape.at(3), 4)),
                                 static_cast<uint32_t>(inputShape.at(0))};
        uint32_t idx          = 0;
        mTranseKernel.setArg(idx++, mTranseGlobalWorkSize[0]);
        mTranseKernel.setArg(idx++, mTranseGlobalWorkSize[1]);
        mTranseKernel.setArg(idx++, mTranseGlobalWorkSize[2]);
        mTranseKernel.setArg(idx++, openCLBuffer(input));
        mTranseKernel.setArg(idx++, openCLBuffer(mSource.get()));
        mTranseKernel.setArg(idx++, static_cast<uint32_t>(inputWidth));
        mTranseKernel.setArg(idx++, static_cast<uint32_t>(inputHeight));
        mTranseKernel.setArg(idx++, static_cast<uint32_t>(inputChannels));
        mTranseKernel.setArg(idx++, UP_DIV(inputShape.at(3), 4));
        mTranseKernel.setArg(idx++, static_cast<uint32_t>(inputpad.left));
        mTranseKernel.setArg(idx++, static_cast<uint32_t>(inputpad.right));

        mTranseLocalWorkSize = localWS3DDefault(mTranseGlobalWorkSize, mMaxWGS_S, mOpenCLBackend->getOpenCLRuntime(),"conv_transe_c4_c16", mTranseKernel).first;
    }
    mGlobalWorkSize = {static_cast<uint32_t>(UP_DIV(outputShape.at(2), 8) * outputShape.at(1)),
                       static_cast<uint32_t>(ROUND_UP(outputShape.at(3), 16)),
                       static_cast<uint32_t>(outputShape.at(0))};
    mLocalWorkSize  = {1, 16, 1};


    std::string kernelname = "depthwise_conv_2d_buf_c16_c" + std::to_string(output_c_pack);
    mKernel = mOpenCLBackend->getOpenCLRuntime()->buildKernel("depthwise_conv2d_subgroup_buf", kernelname, buildOptions);
    uint32_t idx = 0;
    if (mNeedTranse) {
        mKernel.setArg(idx++, openCLBuffer(mSource.get()));
    }
    else {
        mKernel.setArg(idx++, openCLBuffer(input));
    }
    mKernel.setArg(idx++, openCLBuffer(output));
    mKernel.setArg(idx++, openCLBuffer(mFilter.get()));
    mKernel.setArg(idx++, openCLBuffer(mBias.get()));
    mKernel.setArg(idx++, static_cast<uint32_t>(inputHeight));
    mKernel.setArg(idx++, static_cast<uint32_t>(inputWidth));
    mKernel.setArg(idx++, static_cast<uint32_t>(inputChannels));
    mKernel.setArg(idx++, static_cast<uint32_t>(inputpad.left));
    mKernel.setArg(idx++, static_cast<uint32_t>(inputpad.right));
    mKernel.setArg(idx++, static_cast<uint32_t>(outputHeight));
    mKernel.setArg(idx++, static_cast<uint32_t>(outputWidth));
    mKernel.setArg(idx++, static_cast<uint32_t>(outputpad.left));
    mKernel.setArg(idx++, static_cast<uint32_t>(outputpad.right));
    mKernel.setArg(idx++, static_cast<uint32_t>(paddingShape[1]));
    mKernel.setArg(idx++, static_cast<uint32_t>(paddingShape[0]));

    return NO_ERROR;
}

ErrorCode DepthwiseConvSubgroupBufExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start DepthwiseConvSubgroupBufExecution onExecute !\n");
#endif

    if (mNeedTranse) {
#ifdef ENABLE_OPENCL_TIME_PROFILER
        cl::Event event;

        run3DKernelDefault(mTranseKernel, mTranseGlobalWorkSize, mTranseLocalWorkSize,
                           mOpenCLBackend->getOpenCLRuntime(), &event);
        
        mOpenCLBackend->getOpenCLRuntime()->pushEvent({"DepthwiseConvSubgroup transe", event});
#else
        run3DKernelDefault(mTranseKernel, mTranseGlobalWorkSize, mTranseLocalWorkSize,
                           mOpenCLBackend->getOpenCLRuntime());
#endif
    }

#ifdef ENABLE_OPENCL_TIME_PROFILER
    cl::Event event;
    run3DKernelDefault(mKernel, mGlobalWorkSize, mLocalWorkSize,
                mOpenCLBackend->getOpenCLRuntime(),
                &event);
    
    mOpenCLBackend->getOpenCLRuntime()->pushEvent({"DepthwiseConvSubgroupBuf", event});
#else
    run3DKernelDefault(mKernel, mGlobalWorkSize, mLocalWorkSize,
                        mOpenCLBackend->getOpenCLRuntime());
#endif

#ifdef LOG_VERBOSE
    MNN_PRINT("end DepthwiseConvSubgroupBufExecution onExecute !\n");
#endif
    return NO_ERROR;
}

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_SUPPORT_INTEL_SUBGROUP */
#endif /* MNN_OPENCL_BUFFER_CLOSED */
