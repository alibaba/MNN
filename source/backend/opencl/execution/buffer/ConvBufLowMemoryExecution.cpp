//  ConvBufLowMemoryExecution.cpp
//
//  Created by MNN on 2023/10/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef MNN_LOW_MEMORY
#ifndef MNN_OPENCL_BUFFER_CLOSED
#include "ConvBufLowMemoryExecution.hpp"
// #define LOG_VERBOSE
namespace MNN {
namespace OpenCL {

// set mDequantScale mDequantOffset mNumQuantBit mFilterDataPtr from mConv2dParams
void ConvBufLowMemoryExecution::getInfoFromOpLowMemory(std::shared_ptr<ConvolutionCommon::Int8Common> & quanCommon) {
    quanCommon = ConvolutionCommon::load(mConv2dParams, this->backend(), false, true);
    if ((mOpenCLBackend->getMemory() == BackendConfig::Memory_Low) && (mConv2dParams->quanParameter() != nullptr)) {
        mLowMemoryFlag = true;
    } else {
        MNN_ERROR("Conv buf low memory init error.\n");
        MNN_ASSERT(false);
    }
    // set mNumQuantBit
    if (quanCommon->quan->type() == 4) {
        mNumQuantBit = 8;
    } else if (quanCommon->quan->type() == 1 || quanCommon->quan->type() == 2) {
        mNumQuantBit = 4;
    } else {/* More types to be supported. */}
    // src of alpha in CPU
    float * dequantAlpha = quanCommon->alpha.get();
    int numAlpha = mOutputChannel;
    // set mDequantScale mDequantOffset
    int numAlphaPack = ROUND_UP(numAlpha, 16);
    int numBiasPack = ROUND_UP(mOutputChannel, 16);
    mResource->bias.reset(Tensor::createDevice<float>({1, 1, 1, ROUND_UP(mOutputChannel, 16)}));
    mResource->dequantScale.reset(Tensor::createDevice<float>({numAlphaPack}));
    mResource->dequantOffset.reset(Tensor::createDevice<float>({numAlphaPack}));
    mOpenCLBackend->onAcquireBuffer(mResource->bias.get(), Backend::STATIC);
    mOpenCLBackend->onAcquireBuffer(mResource->dequantScale.get(), Backend::STATIC);
    mOpenCLBackend->onAcquireBuffer(mResource->dequantOffset.get(), Backend::STATIC);
    cl::Buffer &biasBuffer = openCLBuffer(mResource->bias.get());
    cl::Buffer &dequantScaleBuffer = openCLBuffer(mResource->dequantScale.get());
    cl::Buffer &dequantOffsetBuffer = openCLBuffer(mResource->dequantOffset.get());
    // transfer data from src in cpu to dst in gpu
    int bytes = mOpenCLBackend->fpBytes();
    cl_int resBias, resScale, resOffset;
    auto biasPtrCL = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(biasBuffer, true, CL_MAP_WRITE, 0, numBiasPack * bytes, nullptr, nullptr, &resBias);
    void * dequantScaleBufferMap = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(dequantScaleBuffer, true, CL_MAP_WRITE, 0, numAlphaPack * bytes, nullptr, nullptr, &resScale);
    void * dequantOffsetBufferMap = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(dequantOffsetBuffer, true, CL_MAP_WRITE, 0, numAlphaPack * bytes, nullptr, nullptr, &resOffset);

    if (biasPtrCL != nullptr && resBias == CL_SUCCESS) {
        ::memset(biasPtrCL, 0, numBiasPack * bytes);
        if (nullptr != mConv2dParams->bias()) {
            const float *biasDataPtr = mConv2dParams->bias()->data();
            if (bytes == 2){
                for(int i = 0; i < mOutputChannel; i++) {
                    ((half_float::half*)biasPtrCL)[i] = (half_float::half)(biasDataPtr[i]);
                }
            } else {
                ::memcpy(biasPtrCL, biasDataPtr, mOutputChannel * sizeof(float));
            }
        }
    }
    ::memset(dequantScaleBufferMap, -1, numAlphaPack * bytes);
    ::memset(dequantOffsetBufferMap, 0, numAlphaPack * bytes);
    if (dequantScaleBufferMap != nullptr && dequantOffsetBufferMap != nullptr && resScale == CL_SUCCESS && resOffset == CL_SUCCESS) {
        if (bytes == 2) {
            if (quanCommon->asymmetric) {
                for (int i = 0; i < numAlpha; ++i) {
                    ((half_float::half *)dequantOffsetBufferMap)[i] = (half_float::half)dequantAlpha[2 * i];
                    ((half_float::half *)dequantScaleBufferMap)[i] = (half_float::half)dequantAlpha[2 * i + 1];
                }
            } else {
                for (int i = 0; i < numAlpha; ++i) {
                    ((half_float::half *)dequantScaleBufferMap)[i] = (half_float::half)dequantAlpha[i];
                    ((half_float::half *)dequantOffsetBufferMap)[i] = 0.0f;
                }
            }
        } else {
            if (quanCommon->asymmetric) {
                for (int i = 0; i < numAlpha; ++i) {
                    ((float *)dequantOffsetBufferMap)[i] = dequantAlpha[2 * i];
                    ((float *)dequantScaleBufferMap)[i] = dequantAlpha[2 * i + 1];
                }
            } else {
                for (int i = 0; i < numAlpha; ++i) {
                    ((float *)dequantScaleBufferMap)[i] = dequantAlpha[i];
                    ((float *)dequantOffsetBufferMap)[i] = 0.0f;
                }
            }
        }
    } else {
        MNN_ERROR("Map error dequantBufferMap == nullptr \n");
        MNN_ASSERT(false);
    }
    mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(biasBuffer, biasPtrCL);
    mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(dequantScaleBuffer, dequantScaleBufferMap);
    mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(dequantOffsetBuffer, dequantOffsetBufferMap);
    // set mFilterDataPtr
    mFilterDataPtr = (void *)quanCommon->weight.get();
}
// set mKernelBuffer for the 1x1 kernels
void ConvBufLowMemoryExecution::set1x1WeightLowMemory(int packCout, int packCin, void * filterDataPtr, std::shared_ptr<ConvolutionCommon::Int8Common> & quanCommon) {
    cl_int res;
    std::shared_ptr<Tensor> filterBuffer(Tensor::createDevice<float>({ROUND_UP(mOutputChannel, 8)/*Cout pack set to max 8*/, ROUND_UP(mInputChannel, packCin), mResource->mKernelWidth, mResource->mKernelHeight}));
    size_t buffer_size = filterBuffer->usize() / sizeof(float);
    float *dequantAlpha = quanCommon->alpha.get();
    // shared part for all cases
    if (mNumQuantBit == 8) {
        // int8 case
        buffer_size *= sizeof(int8_t);
    } else if (mNumQuantBit == 4){
        // int4 case
        buffer_size /= 2;
    } else {/* More types to be supported. */}
    mResource->kernelBuffer.reset(new cl::Buffer(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size));
    auto kernelBufferPtr = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(*(mResource->kernelBuffer.get()), true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &res);
    if(kernelBufferPtr != nullptr && res == CL_SUCCESS){
        ::memset(kernelBufferPtr, 0, buffer_size);

        for(int o = 0; o < mOutputChannel; o++){
            float zero = 0;
            if(quanCommon->asymmetric){
                zero = (-dequantAlpha[2 * o + 1])/dequantAlpha[2 * o];
            }
            int i = 0;
            for(; i < mInputChannel; i++){
                int bufferIdx = (o/packCout) * packCin*packCout + (i/packCin)*packCin*ROUND_UP(mOutputChannel, packCout) + (i%packCin)*packCout + (o%packCout);//(Ci/packCin， Co/packCout, packCin， packCout)
                int filterIdx = o*mInputChannel + i;
                if (mNumQuantBit == 8) {
                    // int8 case
                    ((int8_t *)kernelBufferPtr)[bufferIdx] = (int8_t)(((int8_t *)filterDataPtr)[filterIdx]);
                } else if (mNumQuantBit == 4){
                    // int4 case
                    if (bufferIdx % 2 == 0) {
                        ((uint8_t *)kernelBufferPtr)[bufferIdx / 2] += (uint8_t)((((int8_t *)filterDataPtr)[filterIdx] + 8) * 16);
                    } else {
                        ((uint8_t *)kernelBufferPtr)[bufferIdx / 2] += (uint8_t)(((int8_t *)filterDataPtr)[filterIdx] + 8);
                    }
                } else {/* More types to be supported. */}
            }
            for(; i < ROUND_UP(mInputChannel, 4); i++){
                int bufferIdx = (o/packCout) * packCin*packCout + (i/packCin)*packCin*ROUND_UP(mOutputChannel, packCout) + (i%packCin)*packCout + (o%packCout);//(Ci/packCin， Co/packCout, packCin， packCout)
                if (mNumQuantBit == 8) {
                    // int8 case
                    ((int8_t *)kernelBufferPtr)[bufferIdx] = (int8_t)(zero);
                } else if (mNumQuantBit == 4){
                    // int4 case
                    if (bufferIdx % 2 == 0) {
                        ((uint8_t *)kernelBufferPtr)[bufferIdx / 2] += (uint8_t)((zero + 8) * 16);
                    } else {
                        ((uint8_t *)kernelBufferPtr)[bufferIdx / 2] += (uint8_t)(zero + 8);
                    }
                }
            }
        }
    } else {
        MNN_ERROR("set1x1WeightLowMemory: Map error ptrCL == nullptr \n");
        MNN_ASSERT(false);
    }
    mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(*(mResource->kernelBuffer.get()), kernelBufferPtr);
}
// set mFilter for the general kernels
void ConvBufLowMemoryExecution::setGeneralWeightLowMemory(void* filterDataPtr, std::shared_ptr<ConvolutionCommon::Int8Common> & quanCommon) {
    if (filterDataPtr != nullptr) {
        std::vector<int> filterImageShape{ROUND_UP(mInputChannel, 4), (UP_DIV(mOutputChannel, 4) * mResource->mKernelWidth * mResource->mKernelHeight)};
        std::shared_ptr<Tensor> filterBuffer(Tensor::createDevice<float>({mOutputChannel, ROUND_UP(mInputChannel, 4), mResource->mKernelWidth, mResource->mKernelHeight}));
        // int buffer_size = filterBuffer->elementSize();
        size_t buffer_size = filterBuffer->usize() / sizeof(float);
        buffer_size *= sizeof(int8_t);
        cl::Buffer filterBufferCL(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size);
        filterBuffer->buffer().device = (uint64_t)(&filterBufferCL);
        float *dequantAlpha = quanCommon->alpha.get();
        // map and pack data from filterDataPtr
        cl_int res;
        auto ptrCL = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(filterBufferCL, true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &res);
        if(ptrCL != nullptr && res == CL_SUCCESS) {
            ::memset(ptrCL, 0, buffer_size);
            const int copy_size = mResource->mKernelWidth * mResource->mKernelHeight * sizeof(int8_t);
            for(int oc=0; oc<mOutputChannel; oc++) {
                float zero = 0;
                if(quanCommon->asymmetric){
                    zero = (-dequantAlpha[2 * oc + 1])/dequantAlpha[2 * oc];
                }
                int ic = 0;
                for(; ic<mInputChannel; ic++) {
                    ::memcpy((int8_t *)ptrCL + (oc * ROUND_UP(mInputChannel, 4) + ic) * mResource->mKernelWidth * mResource->mKernelHeight, ((int8_t *)filterDataPtr) + (oc * mInputChannel + ic) * mResource->mKernelWidth * mResource->mKernelHeight, copy_size);
                }
                for(; ic<ROUND_UP(mInputChannel, 4); ic++) {
                    ((int8_t *)ptrCL)[(oc * ROUND_UP(mInputChannel, 4) + ic) * mResource->mKernelWidth * mResource->mKernelHeight] = (int8_t)(zero);
                }
            }
        } else {
            MNN_ERROR("setGeneralWeightLowMemory: Map error ptrCL == nullptr \n");
        }
        mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(filterBufferCL, ptrCL);
        // convert to NC4HW4
        if (mNumQuantBit == 8) {
            // ROUND_UP(IC, 4), UP_DIV(OC, 4) * mKernelWidth * mKernelHeight
            mResource->filter.reset(Tensor::createDevice<int8_t>({1, filterImageShape[1], 1, 4 * filterImageShape[0]}));
            mOpenCLBackend->onAcquireBuffer(mResource->filter.get(), Backend::STATIC);
            MNN::OpenCL::BufferConvertor bufferConvertor{mOpenCLBackend->getOpenCLRuntime()};
            // filterBuffer shape: {OC, ROUND_UP(IC, 4), mKernelWidth, mKernelHeight}
            bufferConvertor.convertToNC4HW4Buffer(filterBuffer.get(), MNN::OpenCL::CONV2D_FILTER, mResource->filter.get(), false, true, mLowMemoryFlag, mNumQuantBit);
        } else if (mNumQuantBit == 4){
            // ROUND_UP(IC, 4), UP_DIV(OC, 4) * mKernelWidth * mKernelHeight
            // For int4 case, data stored in mFilter should be uint8_t,
            // while "Tensor::createDevice<uint8_t>" occupies more memory than "Tensor::createDevice<int8_t>".
            // Therefore, we use "Tensor::createDevice<int8_t>" currently, leaving "Tensor::createDevice<uint8_t>" to be supported.
            mResource->filter.reset(Tensor::createDevice<int8_t>({1, filterImageShape[1], 1, 2 * filterImageShape[0]}));
            mOpenCLBackend->onAcquireBuffer(mResource->filter.get(), Backend::STATIC);
            MNN::OpenCL::BufferConvertor bufferConvertor{mOpenCLBackend->getOpenCLRuntime()};
            // filterBuffer shape: {OC, ROUND_UP(IC, 4), mKernelWidth, mKernelHeight}
            bufferConvertor.convertToNC4HW4Buffer(filterBuffer.get(), MNN::OpenCL::CONV2D_FILTER, mResource->filter.get(), false, true, mLowMemoryFlag, mNumQuantBit);
        } else {/* More types to be supported. */}
    } else {
        MNN_ERROR("GetConvParams Error: filterDataPtr == nullptr. \n");
        MNN_ASSERT(false);
    }
}
// select the fastest kernel for the 1x1 cases by tuning
void ConvBufLowMemoryExecution::tune1x1CaseLowMemory(Tensor * input, Tensor * output) {
    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);
    auto runTime     = ((OpenCLBackend *)backend())->getOpenCLRuntime();
    mOpenCLBackend->startRecord(mRecording);
    const int height             = outputShape.at(1);
    const int width              = outputShape.at(2);
    const int outChannel         = outputShape.at(3);
    const int inputHeight   = inputShape.at(1);
    const int inputWidth    = inputShape.at(2);
    const int inputChannels = inputShape.at(3);
    const int inputChannelBlocks = UP_DIV(inputChannels, 4);
    std::string info = std::to_string(inputChannels) + "_" + std::to_string(mResource->mKernelHeight) + "_" + std::to_string(mResource->mKernelWidth) + "_" + std::to_string(mResource->mStrides[0]) + "_" + std::to_string(mResource->mStrides[1]) + "_" + std::to_string(mResource->mDilations[0]) + "_" + std::to_string(mResource->mDilations[1]);
    // {"conv_2d_1x1_c4h1w4", "conv_2d_1x1_c4h1w2", "conv_2d_1x1_c4h1w1", "conv_2d_1x1_c8h1w4"};
    const int total_kernel = 5;
    std::string kernelName[total_kernel] = {"conv_2d_1x1_c4h1w4", "conv_2d_1x1_c4h1w2", "conv_2d_1x1_c4h1w1", "conv_2d_1x1_c8h1w4", "conv_2d_1x1_c8h1w2"};
    int itemC[total_kernel] = {4, 4, 4, 8, 8};
    int itemW[total_kernel] = {4, 2, 1, 4, 2};
    int actual_kernel = total_kernel;
    if(mOpenCLBackend->getOpenCLRuntime()->getCLTuneLevel() == Normal) {
        actual_kernel = 2;
        kernelName[0] = "conv_2d_1x1_c4h1w1";
        itemC[0]      = 4;
        itemW[0]      = 1;
        kernelName[1] = "conv_2d_1x1_c8h1w2";
        itemC[1]      = 8;
        itemW[1]      = 2;
    } else if(mOpenCLBackend->getOpenCLRuntime()->getCLTuneLevel() == Fast || mOpenCLBackend->getOpenCLRuntime()->getCLTuneLevel() == None) {
        actual_kernel = 1;
        kernelName[0] = "conv_2d_1x1_c4h1w1";
        itemC[0]      = 4;
        itemW[0]      = 1;
    }

    cl::Kernel kernel[total_kernel];
    std::vector<uint32_t> globalWorkSize[total_kernel];
    std::vector<uint32_t> localWorkSize[total_kernel];
    std::pair<int, int> min_cost(INT_MAX, 0);//(min_time, min_index)
    for(int knl_idx = 0; knl_idx < actual_kernel; knl_idx++) {
        std::set<std::string> buildOption = mResource->buildOptions;
        if(outputShape.at(3) % itemC[knl_idx] != 0){
            buildOption.emplace("-DCHANNEL_LEAVE");
        }
        if((outputShape.at(2) % itemW[knl_idx]) != 0){
            buildOption.emplace("-DBLOCK_LEAVE");
        }
        kernel[knl_idx]        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("conv_2d_buf", kernelName[knl_idx], buildOption);
        uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(kernel[knl_idx]));

        uint32_t idx            = 0;
        cl_int ret = CL_SUCCESS;
        globalWorkSize[knl_idx] = {static_cast<uint32_t>(UP_DIV(outputShape.at(3), itemC[knl_idx]) * UP_DIV(outputShape.at(2), itemW[knl_idx])), static_cast<uint32_t>(outputShape.at(0) * outputShape.at(1))};

        ret |= kernel[knl_idx].setArg(idx++, globalWorkSize[knl_idx][0]);
        ret |= kernel[knl_idx].setArg(idx++, globalWorkSize[knl_idx][1]);
        ret |= kernel[knl_idx].setArg(idx++, UP_DIV(width, itemW[knl_idx]));
        ret |= kernel[knl_idx].setArg(idx++, openCLBuffer(input));
        ret |= kernel[knl_idx].setArg(idx++, *mResource->kernelBuffer.get());
        ret |= kernel[knl_idx].setArg(idx++, openCLBuffer(mResource->dequantScale.get()));
        ret |= kernel[knl_idx].setArg(idx++, openCLBuffer(mResource->dequantOffset.get()));
        ret |= kernel[knl_idx].setArg(idx++, openCLBuffer(mResource->bias.get()));
        ret |= kernel[knl_idx].setArg(idx++, openCLBuffer(output));
        ret |= kernel[knl_idx].setArg(idx++, static_cast<int>(inputChannelBlocks));
        ret |= kernel[knl_idx].setArg(idx++, height);
        ret |= kernel[knl_idx].setArg(idx++, width);
        ret |= kernel[knl_idx].setArg(idx++, UP_DIV(outChannel, 4));
        MNN_CHECK_CL_SUCCESS(ret, "setArg Conv1x1BufLowMemory Kernel Select");
        std::pair<std::vector<uint32_t>, int> retTune;
        retTune = gws2dLwsTune(kernel[knl_idx], globalWorkSize[knl_idx], kernelName[knl_idx], maxWorkGroupSize);
        //printf("cov1x1 %d, %d\n", knl_idx, retTune.second);
        if(min_cost.first > retTune.second) {
            min_cost.first = retTune.second;
            min_cost.second = knl_idx;
            mLocalWorkSize = {retTune.first[0], retTune.first[1]};
        }
    }

    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
    int min_index  = min_cost.second;
    mGlobalWorkSize = {globalWorkSize[min_index][0], globalWorkSize[min_index][1]};
    std::set<std::string> buildOption = mResource->buildOptions;
    if(outputShape.at(3) % itemC[min_index] != 0){
        buildOption.emplace("-DCHANNEL_LEAVE");
    }
    if((outputShape.at(2) % itemW[min_index]) != 0){
        buildOption.emplace("-DBLOCK_LEAVE");
    }
    mKernel        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("conv_2d_buf", kernelName[min_index], buildOption);
    // MNN_PRINT("Kernel is %d.\n", min_index);
    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= mKernel.setArg(idx++, mGlobalWorkSize[0]);
    ret |= mKernel.setArg(idx++, mGlobalWorkSize[1]);
    ret |= mKernel.setArg(idx++, UP_DIV(width, itemW[min_index]));
    ret |= mKernel.setArg(idx++, openCLBuffer(input));
    ret |= mKernel.setArg(idx++, *mResource->kernelBuffer.get());
    ret |= mKernel.setArg(idx++, openCLBuffer(mResource->dequantScale.get()));
    ret |= mKernel.setArg(idx++, openCLBuffer(mResource->dequantOffset.get()));
    ret |= mKernel.setArg(idx++, openCLBuffer(mResource->bias.get()));
    ret |= mKernel.setArg(idx++, openCLBuffer(output));
    ret |= mKernel.setArg(idx++, static_cast<int>(inputChannelBlocks));
    ret |= mKernel.setArg(idx++, height);
    ret |= mKernel.setArg(idx++, width);
    ret |= mKernel.setArg(idx++, UP_DIV(outChannel, 4));
    MNN_CHECK_CL_SUCCESS(ret, "setArg Conv1x1BufLowMemory");
    mOpenCLBackend->recordKernel2d(mKernel, mGlobalWorkSize, mLocalWorkSize);
    mOpenCLBackend->endRecord(mRecording);
    return;
}
// select the fastest kernel for the general cases by tuning
void ConvBufLowMemoryExecution::tuneGeneralCaseLowMemory(Tensor * input, Tensor * output) {
    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);
    auto runTime     = ((OpenCLBackend *)backend())->getOpenCLRuntime();
    mOpenCLBackend->startRecord(mRecording);
    const int height             = outputShape.at(1);
    const int width              = outputShape.at(2);
    const int outChannel         = outputShape.at(3);
    const int inputHeight   = inputShape.at(1);
    const int inputWidth    = inputShape.at(2);
    const int inputChannels = inputShape.at(3);
    const int inputChannelBlocks = UP_DIV(inputChannels, 4);
    std::string info = std::to_string(inputChannels) + "_" + std::to_string(mResource->mKernelHeight) + "_" + std::to_string(mResource->mKernelWidth) + "_" + std::to_string(mResource->mStrides[0]) + "_" + std::to_string(mResource->mStrides[1]) + "_" + std::to_string(mResource->mDilations[0]) + "_" + std::to_string(mResource->mDilations[1]);
    int inputImageShape[2]  = {inputHeight, inputWidth};
    int outputImageShape[2] = {height, width};
    int kernelShape[2]      = {mResource->mKernelHeight, mResource->mKernelWidth};
    int strideShape[2]      = {mResource->mStrides[0], mResource->mStrides[1]};
    int paddingShape[2]     = {mPaddings[0], mPaddings[1]};
    int dilationShape[2]    = {mResource->mDilations[0], mResource->mDilations[1]};
    // {"conv_2d_c4h1w2", "conv_2d_c4h1w1", "conv_2d_c8h1w1", "conv_2d_c4h1w4", "conv_2d_c8h2w1", "conv_2d_c4h4w1"};
    const int total_kernel = 7;
    std::string kernelName[total_kernel] = {"conv_2d_c4h1w1", "conv_2d_c4h1w2", "conv_2d_c4h4w1", "conv_2d_c8h2w1", "conv_2d_c8h4w1", "conv_2d_c4h1w4", "conv_2d_c8h1w4"};
    int itemC[total_kernel] = {4, 4, 4, 8, 8, 4, 8};
    int itemH[total_kernel] = {1, 1, 4, 2, 4, 1, 1};
    int itemW[total_kernel] = {1, 2, 1, 1, 1, 4, 4};
    int actual_kernel = total_kernel;
    cl::Kernel kernel[total_kernel];
    std::vector<uint32_t> globalWorkSize[total_kernel];
    std::vector<uint32_t> localWorkSize[total_kernel];
    std::pair<int, int> min_cost(INT_MAX, 0);//(min_time, min_index)
    // MNN_PRINT("Checking kernel %d.\n", knlCheck);
    for (int knl_idx = 0; knl_idx < actual_kernel; knl_idx++) {
        std::set<std::string> buildOption = mResource->buildOptions;
        if(outputShape.at(3) % itemC[knl_idx] != 0){
            buildOption.emplace("-DCHANNEL_LEAVE");
        }
        if((outputShape.at(2) % itemW[knl_idx]) != 0 || (outputShape.at(1) % itemH[knl_idx]) != 0){
            buildOption.emplace("-DBLOCK_LEAVE");
        }
        kernel[knl_idx]        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("conv_2d_buf", kernelName[knl_idx], buildOption);
        uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(kernel[knl_idx]));

        globalWorkSize[knl_idx] = {static_cast<uint32_t>(UP_DIV(outputShape.at(3), itemC[knl_idx]) * UP_DIV(outputShape.at(2), itemW[knl_idx])), static_cast<uint32_t>(outputShape.at(0) * UP_DIV(outputShape.at(1), itemH[knl_idx]))};
        uint32_t idx            = 0;
        cl_int ret = CL_SUCCESS;
        ret |= kernel[knl_idx].setArg(idx++, globalWorkSize[knl_idx][0]);
        ret |= kernel[knl_idx].setArg(idx++, globalWorkSize[knl_idx][1]);
        ret |= kernel[knl_idx].setArg(idx++, openCLBuffer(input));
        ret |= kernel[knl_idx].setArg(idx++, openCLBuffer(mResource->filter.get()));
        ret |= kernel[knl_idx].setArg(idx++, openCLBuffer(mResource->dequantScale.get()));
        ret |= kernel[knl_idx].setArg(idx++, openCLBuffer(mResource->dequantOffset.get()));
        ret |= kernel[knl_idx].setArg(idx++, openCLBuffer(mResource->bias.get()));
        ret |= kernel[knl_idx].setArg(idx++, openCLBuffer(output));
        ret |= kernel[knl_idx].setArg(idx++, sizeof(inputImageShape), inputImageShape);
        ret |= kernel[knl_idx].setArg(idx++, inputChannels);
        ret |= kernel[knl_idx].setArg(idx++, inputChannelBlocks);
        ret |= kernel[knl_idx].setArg(idx++, sizeof(outputImageShape), outputImageShape);
        ret |= kernel[knl_idx].setArg(idx++, sizeof(kernelShape), kernelShape);
        ret |= kernel[knl_idx].setArg(idx++, sizeof(strideShape), strideShape);
        ret |= kernel[knl_idx].setArg(idx++, sizeof(paddingShape), paddingShape);
        ret |= kernel[knl_idx].setArg(idx++, sizeof(dilationShape), dilationShape);
        ret |= kernel[knl_idx].setArg(idx++, UP_DIV(width, itemW[knl_idx]));
        ret |= kernel[knl_idx].setArg(idx++, UP_DIV(outChannel, 4));
        ret |= kernel[knl_idx].setArg(idx++, UP_DIV(height, itemH[knl_idx]));
        MNN_CHECK_CL_SUCCESS(ret, "setArg ConvBufLowMemory Kernel Select");
        std::pair<std::vector<uint32_t>, int> retTune;
        retTune = gws2dLwsTune(kernel[knl_idx], globalWorkSize[knl_idx], kernelName[knl_idx] + info, maxWorkGroupSize);
        if(min_cost.first > retTune.second) {
            min_cost.first = retTune.second;
            min_cost.second = knl_idx;
            mLocalWorkSize = {retTune.first[0], retTune.first[1]};
        }
    }
    int min_index  = min_cost.second;
    mGlobalWorkSize = {globalWorkSize[min_index][0], globalWorkSize[min_index][1]};

    std::set<std::string> buildOption = mResource->buildOptions;
    if(outputShape.at(3) % itemC[min_index] != 0){
        buildOption.emplace("-DCHANNEL_LEAVE");
    }
    if((outputShape.at(2) % itemW[min_index]) != 0 || (outputShape.at(1) % itemH[min_index]) != 0){
        buildOption.emplace("-DBLOCK_LEAVE");
    }
    mKernel        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("conv_2d_buf", kernelName[min_index], buildOption);

    uint32_t idx            = 0;
    cl_int ret = CL_SUCCESS;
    ret |= mKernel.setArg(idx++, mGlobalWorkSize[0]);
    ret |= mKernel.setArg(idx++, mGlobalWorkSize[1]);
    ret |= mKernel.setArg(idx++, openCLBuffer(input));
    ret |= mKernel.setArg(idx++, openCLBuffer(mResource->filter.get()));
    ret |= mKernel.setArg(idx++, openCLBuffer(mResource->dequantScale.get()));
    ret |= mKernel.setArg(idx++, openCLBuffer(mResource->dequantOffset.get()));
    ret |= mKernel.setArg(idx++, openCLBuffer(mResource->bias.get()));
    ret |= mKernel.setArg(idx++, openCLBuffer(output));
    ret |= mKernel.setArg(idx++, sizeof(inputImageShape), inputImageShape);
    ret |= mKernel.setArg(idx++, inputChannels);
    ret |= mKernel.setArg(idx++, inputChannelBlocks);
    ret |= mKernel.setArg(idx++, sizeof(outputImageShape), outputImageShape);
    ret |= mKernel.setArg(idx++, sizeof(kernelShape), kernelShape);
    ret |= mKernel.setArg(idx++, sizeof(strideShape), strideShape);
    ret |= mKernel.setArg(idx++, sizeof(paddingShape), paddingShape);
    ret |= mKernel.setArg(idx++, sizeof(dilationShape), dilationShape);
    ret |= mKernel.setArg(idx++, UP_DIV(width, itemW[min_index]));
    ret |= mKernel.setArg(idx++, UP_DIV(outChannel, 4));
    ret |= mKernel.setArg(idx++, UP_DIV(height, itemH[min_index]));
    MNN_CHECK_CL_SUCCESS(ret, "setArg ConvBufLowMemory");
    mOpenCLBackend->recordKernel2d(mKernel, mGlobalWorkSize, mLocalWorkSize);
    mOpenCLBackend->endRecord(mRecording);
    return;
}
void ConvBufLowMemoryExecution::tuneGemmLowMemory(Tensor * input, Tensor * output) {
    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);
    auto runTime     = ((OpenCLBackend *)backend())->getOpenCLRuntime();
    mOpenCLBackend->startRecord(mRecording);
    const int outChannel = outputShape.at(3);
    const int inputChannels = inputShape.at(3);
    const int batch = outputShape.at(0);
    const int inputChannelBlocks = UP_DIV(inputChannels, 4);
    const int outputChannelBlocks = UP_DIV(outChannel, 4);
    std::string kernelname = "gemm_conv_buf";
    int global_x = outputChannelBlocks;
    int global_y = batch;
    if(batch > 1){
        kernelname = "gemm_conv_b2_buf";
        global_y = UP_DIV(batch, 2);
    }
    mKernel        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("gemm_buf", kernelname, mResource->buildOptions);
    uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(mKernel));
    mGlobalWorkSize = {static_cast<uint32_t>(global_x), static_cast<uint32_t>(global_y)};
    // MNN_PRINT("Kernel is %d.\n", min_index);
    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= mKernel.setArg(idx++, mGlobalWorkSize[0]);
    ret |= mKernel.setArg(idx++, mGlobalWorkSize[1]);
    ret |= mKernel.setArg(idx++, openCLBuffer(input));
    ret |= mKernel.setArg(idx++, *mResource->kernelBuffer.get());
    ret |= mKernel.setArg(idx++, openCLBuffer(mResource->dequantScale.get()));
    ret |= mKernel.setArg(idx++, openCLBuffer(mResource->dequantOffset.get()));
    ret |= mKernel.setArg(idx++, openCLBuffer(mResource->bias.get()));
    ret |= mKernel.setArg(idx++, openCLBuffer(output));
    ret |= mKernel.setArg(idx++, static_cast<int>(outputChannelBlocks));
    ret |= mKernel.setArg(idx++, static_cast<int>(inputChannelBlocks));
    ret |= mKernel.setArg(idx++, static_cast<int>(batch));
    MNN_CHECK_CL_SUCCESS(ret, "setArg gemm_conv_buf");

    mLocalWorkSize = gws2dLwsTune(mKernel, mGlobalWorkSize, kernelname, maxWorkGroupSize).first;
    mOpenCLBackend->recordKernel2d(mKernel, mGlobalWorkSize, mLocalWorkSize);
    mOpenCLBackend->endRecord(mRecording);
    return;
}
ConvBufLowMemoryExecution::ConvBufLowMemoryExecution(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, const MNN::Op *op, Backend *backend)
    : ConvBufCommonExecution(backend) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start ConvBufLowMemoryExecution init !\n");
#endif
    mResource.reset(new ConvBufResource);
    mOpenCLBackend                 = static_cast<OpenCLBackend *>(backend);
    const auto *conv2dParams       = op->main_as_Convolution2D();
    const auto *conv2dCommonParams = conv2dParams->common();
    mConv2dParams                  = conv2dParams;
    mResource->conv2dCommonParams  = conv2dCommonParams;
    mResource->mStrides                       = {conv2dCommonParams->strideY(), conv2dCommonParams->strideX()};
    mResource->mDilations                     = {conv2dCommonParams->dilateY(), conv2dCommonParams->dilateX()};
    auto padding = ConvolutionCommon::convolutionPad(inputs[0], outputs[0], conv2dCommonParams);
    mPaddings[0] = padding.second;//padY
    mPaddings[1] = padding.first;//padX

    mResource->mKernelWidth   = conv2dCommonParams->kernelX();
    mResource->mKernelHeight  = conv2dCommonParams->kernelY();
    mOutputChannel = conv2dCommonParams->outputCount();
    std::string kernelName = "conv_2d_c4h1w4";
    mInputChannel = conv2dCommonParams->inputCount();
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
    // set mDequantScale, mDequantOffset, mFilterDataPtr
    // prepare mDequantScale mDequantOffset mFilterDataPtr
    getInfoFromOpLowMemory(quanCommon);
    //select opt conv method
    //std::vector<int> inputShape  = tensorShapeFormat(inputs[0]);
    //const int inputChannels = inputShape.at(3);
    //const int batch = inputShape.at(0);
    //printf("mConv1x1Opt = %d  mKernelHeight = %d  mKernelWidth = %d  mPaddings[0] = %d mPaddings[1] = %d mStrides[0] = %d mStrides[1] = %d inputs[0]->width() = %d inputs[0]->height() = %d mOutputChannel = %d inputChannels = %d batch = %d\n", mConv1x1Opt, mKernelHeight, mKernelWidth,
            //mPaddings[0], mPaddings[1], mStrides[0], mStrides[1], inputs[0]->width(), inputs[0]->height(), mOutputChannel, inputChannels, batch);
    if (mResource->mKernelHeight == mResource->mKernelWidth && mResource->mKernelHeight == 1 && mResource->mStrides[0] == 1 && mResource->mStrides[1] == 1) {
        set1x1WeightLowMemory(4, 4, mFilterDataPtr, quanCommon);
    }else {
        // set mFilter for not 1x1 case
        setGeneralWeightLowMemory(mFilterDataPtr, quanCommon);
    }
    // Create Kernel
    if (conv2dCommonParams->relu()) {
        mResource->buildOptions.emplace("-DRELU");
    } else if (conv2dCommonParams->relu6()) {
        mResource->buildOptions.emplace("-DRELU6");
    }
    if (mNumQuantBit == 8) {
        // int8 case
        mResource->buildOptions.emplace("-DUSE_LOW_BIT_WEIGHT_INT8");
    } else if (mNumQuantBit == 4){
        // int4 case
        mResource->buildOptions.emplace("-DUSE_LOW_BIT_WEIGHT_INT4");
    } else {/* More types to be supported. */}
    mKernel           = mOpenCLBackend->getOpenCLRuntime()->buildKernel("conv_2d_buf", kernelName, mResource->buildOptions);
    mMaxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(mKernel));
#ifdef LOG_VERBOSE
    MNN_PRINT("end ConvExecution init !\n");
#endif
}

ConvBufLowMemoryExecution::ConvBufLowMemoryExecution(std::shared_ptr<ConvBufResource> resource, const Op* op, Backend *backend)
    : ConvBufCommonExecution(backend) {
    mResource = resource;
    const auto *conv2dParams       = op->main_as_Convolution2D();
    const auto *conv2dCommonParams = conv2dParams->common();
    mConv2dParams                  = conv2dParams;
    mResource->conv2dCommonParams  = conv2dCommonParams;
}

ConvBufLowMemoryExecution::~ConvBufLowMemoryExecution() {
    // Do nothing
}

bool ConvBufLowMemoryExecution::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    *dst = new ConvBufLowMemoryExecution(mResource, op, bn);
    return true;
}

ErrorCode ConvBufLowMemoryExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start ConvExecution onResize !\n");
#endif
    auto input  = inputs[0];
    auto output = outputs[0];
    auto padding = ConvolutionCommon::convolutionPad(input, output, mResource->conv2dCommonParams);
    mPaddings[0] = padding.second;//padY
    mPaddings[1] = padding.first;//padX
    // onclone default use conv1x1Opt, need reset
    mResource->gemmOpt = (mResource->mKernelHeight == mResource->mKernelWidth && mResource->mKernelHeight == 1 && mPaddings[0] == 0 && mPaddings[1] == 0 && mResource->mStrides[0] == 1 && mResource->mStrides[1] == 1 && inputs[0]->width() == 1 && inputs[0]->height() == 1);
    mResource->conv1x1Opt = (mResource->mKernelHeight == mResource->mKernelWidth && mResource->mKernelHeight == 1 && mPaddings[0] == 0 && mPaddings[1] == 0 && mResource->mStrides[0] == 1 && mResource->mStrides[1] == 1 && inputs[0]->width() >= 4);
    if (mResource->conv1x1Opt) {
        tune1x1CaseLowMemory(input, output);
    } else if(mResource->gemmOpt){
        tuneGemmLowMemory(input, output);
    } else {
        tuneGeneralCaseLowMemory(input, output);
    }
#ifdef LOG_VERBOSE
    MNN_PRINT("end ConvExecution onResize !\n");
#endif
    return NO_ERROR;
}
ErrorCode ConvBufLowMemoryExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start ConvExecution onExecute !\n");
#endif
#ifdef ENABLE_OPENCL_TIME_PROFILER
    cl::Event event;
    runKernel2D(mKernel, mGlobalWorkSize, mLocalWorkSize, mOpenCLBackend->getOpenCLRuntime(), &event);
    mOpenCLBackend->getOpenCLRuntime()->pushEvent({"ConvBuf2D", event});
#else
    if(mOpenCLBackend->isUseRecordQueue()){
        if(mOpenCLBackend->isDevideOpRecord())
            mOpenCLBackend->addRecord(mRecording);
#ifdef LOG_VERBOSE
        MNN_PRINT("End ConvExecution onExecute... \n");
#endif
        return NO_ERROR;
    }
    // gemm/gemv:
    // input : (batch, ic/4, 4)
    // weight: (ic/4, oc, 4)
    // output: (batch, oc, 4)
    runKernel2D(mKernel, mGlobalWorkSize, mLocalWorkSize, mOpenCLBackend->getOpenCLRuntime());
#endif

#ifdef LOG_VERBOSE
    MNN_PRINT("end ConvExecution onExecute !\n");
#endif
    return NO_ERROR;
}
} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
#endif /* MNN_LOW_MEMORY */
