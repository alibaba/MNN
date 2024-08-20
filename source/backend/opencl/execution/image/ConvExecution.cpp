//
//  ConvExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ConvExecution.hpp"
#include "ConvWinograd.hpp"
#include "core/ConvolutionCommon.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"
#include "ConvLowMemoryExecution.hpp"

namespace MNN {
namespace OpenCL {

ConvCommonExecution::ConvCommonExecution(const Convolution2D *conv2dParams, Backend *backend) {
    mResource.reset(new ConvResource);
    mOpenCLBackend           = (OpenCLBackend *)backend;
    auto runtime             = mOpenCLBackend->getOpenCLRuntime();
    int biasSize             = conv2dParams->bias()->size();
    const float *biasDataPtr = conv2dParams->bias()->data();
    
    int buffer_size = ALIGN_UP4(biasSize) * sizeof(float);
    cl::Buffer biasBuffer(runtime->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size);
    cl_int error;
    auto biasPtrCL = runtime->commandQueue().enqueueMapBuffer(biasBuffer, true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &error);
    if(biasPtrCL != nullptr && error == CL_SUCCESS){
        ::memset(biasPtrCL, 0, ALIGN_UP4(biasSize) * sizeof(float));
        ::memcpy(biasPtrCL, biasDataPtr, biasSize * sizeof(float));
    }else{
        MNN_ERROR("Map error biasPtrCL == nullptr \n");
    }
    runtime->commandQueue().enqueueUnmapMemObject(biasBuffer, biasPtrCL);
    mResource->mBias.reset(Tensor::createDevice<float>({1, 1, 1, biasSize}));
    backend->onAcquireBuffer(mResource->mBias.get(), Backend::STATIC);
    copyBufferToImage(runtime, biasBuffer, openCLImage(mResource->mBias.get()), UP_DIV(biasSize, 4), 1);
}
ConvCommonExecution::~ConvCommonExecution() {
    // Do nothinng
}

ConvExecution::ConvExecution(std::shared_ptr<ConvResource> resource, const MNN::Op* op, Backend *backend)
    : CommonExecution(backend, op), ConvCommonExecution(backend) {
    mResource = resource;
    const auto *conv2dParams       = op->main_as_Convolution2D();
    const auto *conv2dCommonParams = conv2dParams->common();
    mResource->mConv2dParams       = conv2dParams;
    mResource->mConv2dCommonParams  = conv2dCommonParams;
}

bool ConvExecution::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    *dst = new ConvExecution(mResource, op, bn);
    return true;
}

ConvExecution::ConvExecution(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, const MNN::Op *op, Backend *backend)
: CommonExecution(backend, op), ConvCommonExecution(op->main_as_Convolution2D(), backend) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start ConvExecution init !\n");
#endif
    mOpenCLBackend                 = static_cast<OpenCLBackend *>(backend);
    const auto *conv2dParams       = op->main_as_Convolution2D();
    const auto *conv2dCommonParams = conv2dParams->common();
    mResource->mConv2dCommonParams = conv2dCommonParams;
    mResource->mStrides            = {conv2dCommonParams->strideY(), conv2dCommonParams->strideX()};
    mResource->mDilations          = {conv2dCommonParams->dilateY(), conv2dCommonParams->dilateX()};
    
    auto pad = ConvolutionCommon::convolutionPad(inputs[0], outputs[0], mResource->mConv2dCommonParams);
    mPaddings[0] = pad.second;
    mPaddings[1] = pad.first;
    
    int kernelWidth   = conv2dCommonParams->kernelX();
    int kernelHeight  = conv2dCommonParams->kernelY();
    int outputChannel = conv2dCommonParams->outputCount();
    auto gpuType = mOpenCLBackend->getOpenCLRuntime()->getGpuType();
#ifndef MNN_OPENCL_BUFFER_CLOSED
    mResource->mWeightUseBuffer = gpuType == GpuType::MALI;
#endif
    
    int weightSize             = 0;
    const float *filterDataPtr = nullptr;
    
    std::shared_ptr<MNN::ConvolutionCommon::Int8Common> quanCommon;
    if (nullptr != conv2dParams->quanParameter()) {
        quanCommon = ConvolutionCommon::load(conv2dParams, backend, true);
        if (nullptr == quanCommon) {
            MNN_ERROR("Memory not Enough, can't extract IDST Convolution: %s \n", op->name()->c_str());
        }
        if (quanCommon->weightFloat.get() == nullptr) {
            MNN_PRINT("quanCommon->weightFloat.get() == nullptr \n");
        }
        // Back to float
        filterDataPtr = quanCommon->weightFloat.get();
        weightSize    = quanCommon->weightFloat.size();
    } else if (nullptr == conv2dParams->weight() || nullptr == conv2dParams->bias()) {
        MNN_ERROR("%s has no weight or bias. The model may be benchmark model, please revert the weight/bias firstly\n", op->name()->c_str());
    }
    
    if (nullptr == filterDataPtr) {
        weightSize    = conv2dParams->weight()->size();
        filterDataPtr = conv2dParams->weight()->data();
    }
    int inputChannel = weightSize / (kernelWidth * kernelHeight * outputChannel);
    
    
    //select opt conv method
    std::string kernelName = "conv_2d_c4h1w4";
    if (kernelHeight == kernelWidth && kernelHeight == 1 && mPaddings[0] == 0 && mPaddings[1] == 0) {
        mResource->mConv1x1Opt = (mResource->mStrides[0] == 1 && mResource->mStrides[1] == 1 && gpuType == GpuType::MALI && !mResource->mWeightUseBuffer);
        if(mResource->mConv1x1Opt){
            kernelName = "conv_2d_1x1_mali";
        }else{
            kernelName = "conv_2d_1x1";
        }
    }
    
    if(mResource->mConv1x1Opt){
        cl_int error;
        std::shared_ptr<Tensor> filterBuffer(Tensor::createDevice<float>({UP_DIV(outputChannel, 4)*4, UP_DIV(inputChannel, 4)*4, kernelWidth, kernelHeight}));
        
        int buffer_size = filterBuffer->elementSize();
        if(mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16()) {
            buffer_size *= sizeof(half_float::half);
        } else {
            buffer_size *= sizeof(float);
        }
        
        mResource->mKernelBuffer.reset(new cl::Buffer(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size));
        auto kernelBufferPtr = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(*(mResource->mKernelBuffer.get()), true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &error);
        if(kernelBufferPtr != nullptr && error == CL_SUCCESS){
            ::memset(kernelBufferPtr, 0, buffer_size);
            for(int o = 0; o < outputChannel; o++){
                for(int i = 0 ; i < inputChannel; i++){
                    int bufferIdx = (o/4) * ROUND_UP(inputChannel, 4)*4 + (i/4)*16 + (o%4)*4 + (i%4);
                    int filterIdx = o*inputChannel + i;
                    if(mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16()){
                        ((half_float::half*)kernelBufferPtr)[bufferIdx] = (half_float::half)(filterDataPtr[filterIdx]);
                    }else{
                        ((float*)kernelBufferPtr)[bufferIdx] = (float)(filterDataPtr[filterIdx]);
                    }
                }
            }
        }else{
            MNN_ERROR("Map error ptrCL == nullptr \n");
        }
        mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(*(mResource->mKernelBuffer.get()), kernelBufferPtr);
        
    }else if(kernelHeight == kernelWidth && kernelHeight == 1 && mPaddings[0] == 0 && mPaddings[1] == 0 && mResource->mWeightUseBuffer){
        cl_int error;
        std::shared_ptr<Tensor> filterBuffer(Tensor::createDevice<float>({UP_DIV(outputChannel, 4), ROUND_UP(inputChannel, 4), 4}));
        
        int buffer_size = filterBuffer->elementSize();
        if(mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16()) {
            buffer_size *= sizeof(half_float::half);
        } else {
            buffer_size *= sizeof(float);
        }
        
        mResource->mKernelBuffer.reset(new cl::Buffer(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size));
        auto kernelBufferPtr = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(*(mResource->mKernelBuffer.get()), true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &error);
        if(kernelBufferPtr != nullptr && error == CL_SUCCESS){
            ::memset(kernelBufferPtr, 0, buffer_size);
            for(int o = 0; o < outputChannel; o++){
                for(int i = 0 ; i < inputChannel; i++){
                    int bufferIdx = (o/4) * ROUND_UP(inputChannel, 4)*4 + i*4 + (o%4);
                    int filterIdx = o*inputChannel + i;
                    if(mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16()){
                        ((half_float::half*)kernelBufferPtr)[bufferIdx] = (half_float::half)(filterDataPtr[filterIdx]);
                    }else{
                        ((float*)kernelBufferPtr)[bufferIdx] = (float)(filterDataPtr[filterIdx]);
                    }
                }
            }
        }else{
            MNN_ERROR("Map error ptrCL == nullptr \n");
        }
        mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(*(mResource->mKernelBuffer.get()), kernelBufferPtr);
    }else{
        std::vector<int> filterImageShape{(int)ROUND_UP(inputChannel, 4), (int)(UP_DIV(outputChannel, 4) * kernelWidth * kernelHeight)};
        std::shared_ptr<Tensor> filterBuffer(
                                             Tensor::createDevice<float>({outputChannel, ROUND_UP(inputChannel, 4), kernelWidth, kernelHeight}));
        
        size_t buffer_size = filterBuffer->elementSize() * sizeof(float);
        cl::Buffer filterBufferCL(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size);
        filterBuffer->buffer().device = (uint64_t)(&filterBufferCL);
        
        cl_int error;
        auto ptrCL = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(filterBufferCL, true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &error);
        if(ptrCL != nullptr && error == CL_SUCCESS) {
            ::memset(ptrCL, 0, buffer_size);
            int cpySrcNum = inputChannel * kernelWidth * kernelHeight;
            int cpyDstNum = ROUND_UP(inputChannel, 4) * kernelWidth * kernelHeight;
            int cpysize = cpySrcNum * sizeof(float);
            for(int o = 0; o < outputChannel; ++o){
                ::memcpy((float*)ptrCL + o * cpyDstNum, filterDataPtr + o * cpySrcNum, cpysize);
            }
            
        }else{
            MNN_ERROR("Map error ptrCL == nullptr \n");
        }
        mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(filterBufferCL, ptrCL);
#ifndef MNN_OPENCL_BUFFER_CLOSED
        if(mResource->mWeightUseBuffer){
            mResource->mFilter.reset(Tensor::createDevice<float>({UP_DIV(inputChannel, 4)*4, UP_DIV(outputChannel, 4), kernelWidth * kernelHeight, 4}));
            int kernel_buffer_size = UP_DIV(outputChannel, 4)*4* UP_DIV(inputChannel, 4)*4* kernelWidth* kernelHeight;
            if(mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16()) {
                kernel_buffer_size *= sizeof(half_float::half);
            } else {
                kernel_buffer_size *= sizeof(float);
            }
            mResource->mKernelBuffer.reset(new cl::Buffer(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, kernel_buffer_size));
            mResource->mFilter.get()->buffer().device = (uint64_t)mResource->mKernelBuffer.get();
            MNN::OpenCL::BufferConvertor bufferConvertor{mOpenCLBackend->getOpenCLRuntime()};
            
            bool needTrans = true;
            bufferConvertor.convertToNC4HW4Buffer(filterBuffer.get(), MNN::OpenCL::CONV2D_FILTER, mResource->mFilter.get(), needTrans);
        } else
#endif
        {
            mResource->mFilter.reset(Tensor::createDevice<float>({1, filterImageShape[1], 1, 4 * filterImageShape[0]}));
            mOpenCLBackend->onAcquireBuffer(mResource->mFilter.get(), Backend::STATIC);
            MNN::OpenCL::ImageBufferConvertor imageBufferConvertor{mOpenCLBackend->getOpenCLRuntime()};
            
            std::string buildOption = "-DBUFFER_INP_FP32";
            imageBufferConvertor.convertBufferToImage(filterBuffer.get(), MNN::OpenCL::CONV2D_FILTER, mResource->mFilter.get(), false, buildOption);
        }
    }
    
    // Create Kernel
    if (mResource->mStrides[0] == 1 && mResource->mStrides[1] == 1 && mResource->mDilations[0] == 1 && mResource->mDilations[1] == 1) {
        mResource->mBuildOptions.emplace("-DMNN_CONV_S1D1");
    }
    mResource->mBuildOptions.emplace("-DBIAS");
    if (mResource->mConv2dCommonParams->relu()) {
        mResource->mBuildOptions.emplace("-DRELU");
    } else if (mResource->mConv2dCommonParams->relu6()) {
        mResource->mBuildOptions.emplace("-DRELU6");
    }
    if(mResource->mWeightUseBuffer){
        mResource->mBuildOptions.emplace("-DUSE_BUFFER");
    }
    
#ifdef LOG_VERBOSE
    MNN_PRINT("end ConvExecution init !\n");
#endif
}

ConvExecution::~ConvExecution() {
    // Do nothing
}

ErrorCode ConvExecution::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start ConvExecution onResize !\n");
#endif
    mUnits.resize(1);
    auto &unit = mUnits[0];
    auto input  = inputs[0];
    auto output = outputs[0];
    
    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);
    const int height             = outputShape.at(1);
    const int width              = outputShape.at(2);
    const int channel            = outputShape.at(3);
    
    const int inputHeight   = inputShape.at(1);
    const int inputWidth    = inputShape.at(2);
    const int inputChannels = inputShape.at(3);
    
    const int inputChannelBlocks = UP_DIV(inputChannels, 4);
    int kernelHeight = mResource->mConv2dCommonParams->kernelY();
    int kernelWidth  = mResource->mConv2dCommonParams->kernelX();
    
    auto pad = ConvolutionCommon::convolutionPad(input, output, mResource->mConv2dCommonParams);
    mPaddings[0] = pad.second;
    mPaddings[1] = pad.first;
    
    std::string info = std::to_string(inputChannels) + "_" + std::to_string(channel) + "_" + std::to_string(kernelHeight) + "_" + std::to_string(kernelWidth) + "_" + std::to_string(mResource->mStrides[0]) + "_" + std::to_string(mResource->mStrides[1]) + "_" + std::to_string(mResource->mDilations[0]) + "_" + std::to_string(mResource->mDilations[1]);
    if (kernelHeight == kernelWidth && kernelHeight == 1 && mPaddings[0] == 0 && mPaddings[1] == 0) {
        if(mResource->mConv1x1Opt){
            
            std::string kernelName = "conv_2d_1x1_mali";
            unit.kernel             = mOpenCLBackend->getOpenCLRuntime()->buildKernel("conv_2d", kernelName, mResource->mBuildOptions);
            uint32_t idx            = 0;
            
            mGlobalWorkSize = {static_cast<uint32_t>(UP_DIV(outputShape.at(3), 4) * UP_DIV(outputShape.at(2), 4)),
                static_cast<uint32_t>(outputShape.at(0) * outputShape.at(1))};
            unit.kernel->get().setArg(idx++, mGlobalWorkSize[0]);
            unit.kernel->get().setArg(idx++, mGlobalWorkSize[1]);
            unit.kernel->get().setArg(idx++, UP_DIV(width, 4));
            unit.kernel->get().setArg(idx++, openCLImage(input));
            unit.kernel->get().setArg(idx++, *mResource->mKernelBuffer.get());
            unit.kernel->get().setArg(idx++, openCLImage(mResource->mBias.get()));
            unit.kernel->get().setArg(idx++, openCLImage(output));
            unit.kernel->get().setArg(idx++, static_cast<int>(inputChannelBlocks));
            unit.kernel->get().setArg(idx++, height);
            unit.kernel->get().setArg(idx++, width);
            
            mLocalWorkSize = localWS2DDefault(mGlobalWorkSize, mResource->mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), kernelName, unit.kernel).first;
            mOpenCLBackend->recordKernel2d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
        }else{
            int inputImageShape[2]  = {inputHeight, inputWidth};
            int outputImageShape[2] = {height, width};
            int stideShape[2]       = {mResource->mStrides[0], mResource->mStrides[1]};
            const int total_kernel = 2;
            std::string kernelName[total_kernel] = {"conv_2d_1x1", "conv_2d_1x1_c8h1w4"};
            int itemC[total_kernel] = {4, 8};
            int itemH[total_kernel] = {1, 1};
            int itemW[total_kernel] = {4, 4};
            
            int actual_kernel = total_kernel;
            
            std::shared_ptr<KernelWrap> kernel[total_kernel];
            std::vector<uint32_t> globalWorkSize[total_kernel];
            std::vector<uint32_t> localWorkSize[total_kernel];
            std::pair<int, int> min_cost(INT_MAX, 0);//(min_time, min_index)
            
            for(int knl_idx = 0; knl_idx < 1; knl_idx++) {
                kernel[knl_idx]        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("conv_2d", kernelName[knl_idx], mResource->mBuildOptions);
                uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(kernel[knl_idx]));
                
                globalWorkSize[knl_idx] = {static_cast<uint32_t>(UP_DIV(outputShape.at(3), itemC[knl_idx]) * UP_DIV(outputShape.at(2), itemW[knl_idx])), static_cast<uint32_t>(outputShape.at(0) * UP_DIV(outputShape.at(1), itemH[knl_idx]))};
                uint32_t idx            = 0;
                kernel[knl_idx]->get().setArg(idx++, globalWorkSize[knl_idx][0]);
                kernel[knl_idx]->get().setArg(idx++, globalWorkSize[knl_idx][1]);
                kernel[knl_idx]->get().setArg(idx++, openCLImage(input));
                if(mResource->mWeightUseBuffer){
                    kernel[knl_idx]->get().setArg(idx++, *mResource->mKernelBuffer.get());
                }else{
                    kernel[knl_idx]->get().setArg(idx++, openCLImage(mResource->mFilter.get()));
                }
                kernel[knl_idx]->get().setArg(idx++, openCLImage(mResource->mBias.get()));
                kernel[knl_idx]->get().setArg(idx++, openCLImage(output));
                kernel[knl_idx]->get().setArg(idx++, sizeof(inputImageShape), inputImageShape);
                kernel[knl_idx]->get().setArg(idx++, static_cast<int>(inputChannelBlocks));
                kernel[knl_idx]->get().setArg(idx++, sizeof(outputImageShape), outputImageShape);
                kernel[knl_idx]->get().setArg(idx++, sizeof(stideShape), stideShape);
                kernel[knl_idx]->get().setArg(idx++, UP_DIV(width, 4));
                kernel[knl_idx]->get().setArg(idx++, UP_DIV(outputShape.at(3), 4));
                
                std::pair<std::vector<uint32_t>, uint32_t> retTune;
                retTune = localWS2DDefault(globalWorkSize[knl_idx], maxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), kernelName[knl_idx] + info, kernel[knl_idx]);
                
                //printf("conv1x1 kernel_%d = %d  [%d, %d]\n", knl_idx, retTune.second, retTune.first[0], retTune.first[1]);
                if(min_cost.first > retTune.second) {
                    min_cost.first = retTune.second;
                    min_cost.second = knl_idx;
                    mLocalWorkSize = {retTune.first[0], retTune.first[1]};
                }
            }
            int min_index  = min_cost.second;
            //printf("min_index = %d  %d\n", min_index, min_cost.first);
            mGlobalWorkSize = {globalWorkSize[min_index][0], globalWorkSize[min_index][1]};
            unit.kernel        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("conv_2d", kernelName[min_index], mResource->mBuildOptions);
            
            uint32_t idx = 0;
            unit.kernel->get().setArg(idx++, mGlobalWorkSize[0]);
            unit.kernel->get().setArg(idx++, mGlobalWorkSize[1]);
            unit.kernel->get().setArg(idx++, openCLImage(input));
            if(mResource->mWeightUseBuffer){
                unit.kernel->get().setArg(idx++, *mResource->mKernelBuffer.get());
            }else{
                unit.kernel->get().setArg(idx++, openCLImage(mResource->mFilter.get()));
            }
            unit.kernel->get().setArg(idx++, openCLImage(mResource->mBias.get()));
            unit.kernel->get().setArg(idx++, openCLImage(output));
            unit.kernel->get().setArg(idx++, sizeof(inputImageShape), inputImageShape);
            unit.kernel->get().setArg(idx++, static_cast<int>(inputChannelBlocks));
            unit.kernel->get().setArg(idx++, sizeof(outputImageShape), outputImageShape);
            unit.kernel->get().setArg(idx++, sizeof(stideShape), stideShape);
            unit.kernel->get().setArg(idx++, UP_DIV(width, 4));
            unit.kernel->get().setArg(idx++, UP_DIV(outputShape.at(3), 4));
            mOpenCLBackend->recordKernel2d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
        }
    }else {
        int inputImageShape[2]  = {inputHeight, inputWidth};
        int outputImageShape[2] = {height, width};
        int kernelShape[2]      = {kernelHeight, kernelWidth};
        int strideShape[2]      = {mResource->mStrides[0], mResource->mStrides[1]};
        int paddingShape[2]     = {mPaddings[0], mPaddings[1]};
        int dilationShape[2]    = {mResource->mDilations[0], mResource->mDilations[1]};
        
        const int total_kernel = 3;
        std::string kernelName[total_kernel] = {"conv_2d_c4h1w4", "conv_2d_c4h4w1", "conv_2d_c8h4w1" };
        int itemC[total_kernel] = {4, 4, 8};
        int itemH[total_kernel] = {1, 4, 4};
        int itemW[total_kernel] = {4, 1, 1};
        
        
        int actual_kernel = total_kernel;
        
        std::shared_ptr<KernelWrap> kernel[total_kernel];
        std::vector<uint32_t> globalWorkSize[total_kernel];
        std::vector<uint32_t> localWorkSize[total_kernel];
        std::pair<int, int> min_cost(INT_MAX, 0);//(min_time, min_index)
        
        for(int knl_idx = 0; knl_idx < total_kernel; knl_idx++) {
            kernel[knl_idx]        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("conv_2d", kernelName[knl_idx], mResource->mBuildOptions);
            uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(kernel[knl_idx]));
            
            globalWorkSize[knl_idx] = {static_cast<uint32_t>(UP_DIV(outputShape.at(3), itemC[knl_idx]) * UP_DIV(outputShape.at(2), itemW[knl_idx])), static_cast<uint32_t>(outputShape.at(0) * UP_DIV(outputShape.at(1), itemH[knl_idx]))};
            uint32_t idx            = 0;
            cl_int ret = CL_SUCCESS;
            ret |= kernel[knl_idx]->get().setArg(idx++, globalWorkSize[knl_idx][0]);
            ret |= kernel[knl_idx]->get().setArg(idx++, globalWorkSize[knl_idx][1]);
            ret |= kernel[knl_idx]->get().setArg(idx++, openCLImage(input));
            if(mResource->mWeightUseBuffer){
                ret |= kernel[knl_idx]->get().setArg(idx++, openCLBuffer(mResource->mFilter.get()));
            }else{
                ret |= kernel[knl_idx]->get().setArg(idx++, openCLImage(mResource->mFilter.get()));
            }
            ret |= kernel[knl_idx]->get().setArg(idx++, openCLImage(mResource->mBias.get()));
            ret |= kernel[knl_idx]->get().setArg(idx++, openCLImage(output));
            ret |= kernel[knl_idx]->get().setArg(idx++, sizeof(inputImageShape), inputImageShape);
            ret |= kernel[knl_idx]->get().setArg(idx++, inputChannelBlocks);
            ret |= kernel[knl_idx]->get().setArg(idx++, sizeof(outputImageShape), outputImageShape);
            ret |= kernel[knl_idx]->get().setArg(idx++, sizeof(kernelShape), kernelShape);
            ret |= kernel[knl_idx]->get().setArg(idx++, sizeof(strideShape), strideShape);
            ret |= kernel[knl_idx]->get().setArg(idx++, sizeof(paddingShape), paddingShape);
            ret |= kernel[knl_idx]->get().setArg(idx++, sizeof(dilationShape), dilationShape);
            ret |= kernel[knl_idx]->get().setArg(idx++, UP_DIV(width, itemW[knl_idx]));
            ret |= kernel[knl_idx]->get().setArg(idx++, UP_DIV(outputShape.at(3), 4));
            ret |= kernel[knl_idx]->get().setArg(idx++, UP_DIV(height, itemH[knl_idx]));
            MNN_CHECK_CL_SUCCESS(ret, "setArg ConvExecution Kernel Select");
            
            std::pair<std::vector<uint32_t>, uint32_t> retTune;
            retTune = localWS2DDefault(globalWorkSize[knl_idx], maxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), kernelName[knl_idx] + info, kernel[knl_idx]);
            
            if(min_cost.first > retTune.second) {
                min_cost.first = retTune.second;
                min_cost.second = knl_idx;
                mLocalWorkSize = {retTune.first[0], retTune.first[1]};
            }
        }
        int min_index  = min_cost.second;
        mGlobalWorkSize = {globalWorkSize[min_index][0], globalWorkSize[min_index][1]};
        unit.kernel        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("conv_2d", kernelName[min_index], mResource->mBuildOptions);
        
        uint32_t idx            = 0;
        cl_int ret = CL_SUCCESS;
        ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[0]);
        ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[1]);
        ret |= unit.kernel->get().setArg(idx++, openCLImage(input));
        if(mResource->mWeightUseBuffer){
            ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mResource->mFilter.get()));
        }else{
            ret |= unit.kernel->get().setArg(idx++, openCLImage(mResource->mFilter.get()));
        }
        ret |= unit.kernel->get().setArg(idx++, openCLImage(mResource->mBias.get()));
        ret |= unit.kernel->get().setArg(idx++, openCLImage(output));
        ret |= unit.kernel->get().setArg(idx++, sizeof(inputImageShape), inputImageShape);
        ret |= unit.kernel->get().setArg(idx++, inputChannelBlocks);
        ret |= unit.kernel->get().setArg(idx++, sizeof(outputImageShape), outputImageShape);
        ret |= unit.kernel->get().setArg(idx++, sizeof(kernelShape), kernelShape);
        ret |= unit.kernel->get().setArg(idx++, sizeof(strideShape), strideShape);
        ret |= unit.kernel->get().setArg(idx++, sizeof(paddingShape), paddingShape);
        ret |= unit.kernel->get().setArg(idx++, sizeof(dilationShape), dilationShape);
        ret |= unit.kernel->get().setArg(idx++, UP_DIV(width, itemW[min_index]));
        ret |= unit.kernel->get().setArg(idx++, UP_DIV(outputShape.at(3), 4));
        ret |= unit.kernel->get().setArg(idx++, UP_DIV(height, itemH[min_index]));
        MNN_CHECK_CL_SUCCESS(ret, "setArg ConvExecution");
        mOpenCLBackend->recordKernel2d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
    }

    unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1]};
    unit.localWorkSize = {mLocalWorkSize[0], mLocalWorkSize[1]};
#ifdef LOG_VERBOSE
    MNN_PRINT("end ConvExecution onResize !\n");
#endif
    return NO_ERROR;
}

class ConvolutionCreator : public OpenCLBackend::Creator {
public:
    virtual ~ConvolutionCreator() = default;
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        auto conv2D = op->main_as_Convolution2D();
        std::vector<int> inputShape  = tensorShapeFormat(inputs[0]);
        const int inputChannels = inputShape.at(3);
#if defined(MNN_LOW_MEMORY) && not defined(MNN_OPENCL_BUFFER_CLOSED)
        {
            auto conv2dParams = op->main_as_Convolution2D();
            if (conv2dParams->quanParameter() != nullptr) {
                if (((conv2dParams->quanParameter()->type() == 4) ||
                     (conv2dParams->quanParameter()->type() == 1) ||
                     (conv2dParams->quanParameter()->type() == 2))) {
                    if ((1 == conv2dParams->quanParameter()->type() || 2 == conv2dParams->quanParameter()->type()) && conv2dParams->quanParameter()->has_scaleInt()) {
                        // Don't support IDST-int8 because of error
                        return nullptr;
                    }
                    return new ConvLowMemoryExecution(inputs, outputs, op, backend);
                } else {
                    //MNN_ERROR("OpenCL Conv buf low memory init error. For Opencl Backend, only support low memory mode of int8 or int4 dequantization currently.\n");
                    return nullptr;
                }
            }
        }
#endif
        if(op->main_as_Convolution2D()->common()->group() > 1){
            // Don't support group > 1 now
            return nullptr;
        }
        
        if (inputs.size() > 1) {
            return nullptr;
        }
        if (nullptr != op->main_as_Convolution2D()->quanParameter()) {
            auto quan = op->main_as_Convolution2D()->quanParameter();
            if (1 == quan->type() || 2 == quan->type()) {
                if (quan->has_scaleInt()) {
                    // Don't support IDST-int8 because of error
                    return nullptr;
                }
            }
        }
        int maxWidth  = static_cast<OpenCLBackend *>(backend)->getOpenCLRuntime()->getMaxImage2DSize()[0];
        int maxHeight = static_cast<OpenCLBackend *>(backend)->getOpenCLRuntime()->getMaxImage2DSize()[1];
        if (ConvWinograd::valid(conv2D->common(), inputs[0], outputs[0], maxWidth, maxHeight)) {
            return new ConvWinograd(op, backend);
        }
        
        return new ConvExecution(inputs, outputs, op, backend);
    }
};

REGISTER_OPENCL_OP_CREATOR(ConvolutionCreator, OpType_Convolution, IMAGE);

} // namespace OpenCL
} // namespace MNN
