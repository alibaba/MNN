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
    quanCommon = ConvolutionCommon::load(mResource->mConv2dParams, this->backend(), false, true);
    if (mResource->mConv2dParams->quanParameter() != nullptr) {
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
    int numAlpha = mResource->mOutputChannel;
    // set mDequantScale mDequantOffset
    int numAlphaPack = ROUND_UP(numAlpha, 16);
    mResource->dequantScale.reset(Tensor::createDevice<int32_t>({numAlphaPack}));
    mResource->dequantOffset.reset(Tensor::createDevice<int32_t>({numAlphaPack}));

    mOpenCLBackend->onAcquireBuffer(mResource->dequantScale.get(), Backend::STATIC);
    mOpenCLBackend->onAcquireBuffer(mResource->dequantOffset.get(), Backend::STATIC);
    cl::Buffer &dequantScaleBuffer = openCLBuffer(mResource->dequantScale.get());
    cl::Buffer &dequantOffsetBuffer = openCLBuffer(mResource->dequantOffset.get());
    // transfer data from src in cpu to dst in gpu
    int fpBytes = mOpenCLBackend->fpBytes();
    cl_int resBias, resScale, resOffset;

    
    void * dequantScaleBufferMap = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(dequantScaleBuffer, true, CL_MAP_WRITE, 0, numAlphaPack * sizeof(int32_t), nullptr, nullptr, &resScale);
    void * dequantOffsetBufferMap = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(dequantOffsetBuffer, true, CL_MAP_WRITE, 0, numAlphaPack * sizeof(int32_t), nullptr, nullptr, &resOffset);
    
    ::memset(dequantScaleBufferMap, -1, numAlphaPack * sizeof(int32_t));
    ::memset(dequantOffsetBufferMap, 0, numAlphaPack * sizeof(int32_t));

    if (dequantScaleBufferMap != nullptr && dequantOffsetBufferMap != nullptr && resScale == CL_SUCCESS && resOffset == CL_SUCCESS) {
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
    } else {
        MNN_ERROR("Map error dequantBufferMap == nullptr \n");
        MNN_ASSERT(false);
    }
    mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(dequantScaleBuffer, dequantScaleBufferMap);
    mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(dequantOffsetBuffer, dequantOffsetBufferMap);
    // set mFilterDataPtr
    mFilterDataPtr = (void *)quanCommon->weight.get();
}
// set mKernelBuffer for the 1x1 kernels
void ConvBufLowMemoryExecution::set1x1WeightLowMemory(int packCout, int packCin, void * filterDataPtr, std::shared_ptr<ConvolutionCommon::Int8Common> & quanCommon) {
    cl_int res;
    std::shared_ptr<Tensor> filterBuffer(Tensor::createDevice<float>({ROUND_UP(mResource->mOutputChannel, packCout), ROUND_UP(mResource->mInputChannel, packCin), mResource->mKernelWidth, mResource->mKernelHeight}));
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
    
    // Use Image load weights
    void *mapPtr = nullptr;
    size_t row_pitch;
    size_t slice_pitch;
    if(UP_DIV(mResource->mInputChannel, packCin) <= 16384 && ROUND_UP(mResource->mOutputChannel, packCout) <= 16384){
        mResource->mUseImage = true;
    }
    if(mResource->mUseImage) {
        if(mNumQuantBit == 4){
            packCin *= 2;
        }
        size_t w = ROUND_UP(mResource->mOutputChannel, packCout);
        size_t h = UP_DIV(mResource->mInputChannel, packCin);
        mResource->mKernelImage.reset(new cl::Image2D(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_FLOAT), w, h, 0, nullptr, &res));
        if (nullptr == mResource->mKernelImage.get() || res != CL_SUCCESS) {
            MNN_ERROR("Alloc Image %d x %d error, code:%d \n", w, h, res);
        }
        mapPtr = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapImage(*(mResource->mKernelImage.get()), true, CL_MAP_WRITE, {0, 0, 0}, {w, h, 1}, &row_pitch, &slice_pitch, nullptr, nullptr, &res);
        if(mNumQuantBit == 4){
            row_pitch *= 2;
        }
    } else{
        mResource->mKernelBuffer.reset(new cl::Buffer(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size));
        mapPtr = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(*(mResource->mKernelBuffer.get()), true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &res);
        row_pitch = ROUND_UP(mResource->mOutputChannel, packCout) * packCin;
    }
    if(mapPtr != nullptr && res == CL_SUCCESS){
        for(int o = 0; o < mResource->mOutputChannel; o++){
            float zero = 0;
            if(quanCommon->asymmetric){
                zero = (-dequantAlpha[2 * o + 1])/dequantAlpha[2 * o];
            }
            int i = 0;
            for(; i < mResource->mInputChannel ; i++){
                int bufferIdx = (i/packCin) * row_pitch + o*packCin + (i%packCin);//(Ci/packCin， Co/packCout * packCout * packCin)
                int filterIdx = o*mResource->mInputChannel + i;
                if (mNumQuantBit == 8) {
                    // int8 case
                    ((int8_t *)mapPtr)[bufferIdx] = (int8_t)(((int8_t *)filterDataPtr)[filterIdx]);
                } else if (mNumQuantBit == 4){
                    // int4 case
                    if (bufferIdx % 2 == 0) {
                        ((uint8_t *)mapPtr)[bufferIdx / 2] += (uint8_t)((((int8_t *)filterDataPtr)[filterIdx] + 8) * 16);
                    } else {
                        ((uint8_t *)mapPtr)[bufferIdx / 2] += (uint8_t)(((int8_t *)filterDataPtr)[filterIdx] + 8);
                    }
                } else {/* More types to be supported. */}
            }
            for(; i < ROUND_UP(mResource->mInputChannel, packCin); i++){
                int bufferIdx = (i/packCin) * row_pitch + o*packCin + (i%packCin);//(Ci/packCin， Co/packCout * packCout * packCin)
                if (mNumQuantBit == 8) {
                    // int8 case
                    ((int8_t *)mapPtr)[bufferIdx] = (int8_t)(zero);
                } else if (mNumQuantBit == 4){
                    // int4 case
                    if (bufferIdx % 2 == 0) {
                        ((uint8_t *)mapPtr)[bufferIdx / 2] += (uint8_t)((zero + 8) * 16);
                    } else {
                        ((uint8_t *)mapPtr)[bufferIdx / 2] += (uint8_t)(zero + 8);
                    }
                }
            }
        }
    } else {
        MNN_ERROR("set1x1WeightLowMemory: Map error ptrCL == nullptr \n");
        MNN_ASSERT(false);
    }
    if(mResource->mUseImage){
        mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(*(mResource->mKernelImage.get()), mapPtr);
    } else{
        mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(*(mResource->mKernelBuffer.get()), mapPtr);
    }
}
// set mFilter for the general kernels
void ConvBufLowMemoryExecution::setGeneralWeightLowMemory(void* filterDataPtr, std::shared_ptr<ConvolutionCommon::Int8Common> & quanCommon) {
    if (filterDataPtr != nullptr) {
        std::vector<int> filterImageShape{ROUND_UP(mResource->mInputChannel, 4), (UP_DIV(mResource->mOutputChannel, 4) * mResource->mKernelWidth * mResource->mKernelHeight)};
        std::shared_ptr<Tensor> filterBuffer(Tensor::createDevice<float>({mResource->mOutputChannel, ROUND_UP(mResource->mInputChannel, 4), mResource->mKernelWidth, mResource->mKernelHeight}));
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
            for(int oc=0; oc<mResource->mOutputChannel; oc++) {
                float zero = 0;
                if(quanCommon->asymmetric){
                    zero = (-dequantAlpha[2 * oc + 1])/dequantAlpha[2 * oc];
                }
                int ic = 0;
                for(; ic<mResource->mInputChannel; ic++) {
                    ::memcpy((int8_t *)ptrCL + (oc * ROUND_UP(mResource->mInputChannel, 4) + ic) * mResource->mKernelWidth * mResource->mKernelHeight, ((int8_t *)filterDataPtr) + (oc * mResource->mInputChannel + ic) * mResource->mKernelWidth * mResource->mKernelHeight, copy_size);
                }
                for(; ic<ROUND_UP(mResource->mInputChannel, 4); ic++) {
                    ((int8_t *)ptrCL)[(oc * ROUND_UP(mResource->mInputChannel, 4) + ic) * mResource->mKernelWidth * mResource->mKernelHeight] = (int8_t)(zero);
                }
            }
        } else {
            MNN_ERROR("setGeneralWeightLowMemory: Map error ptrCL == nullptr \n");
        }
        mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(filterBufferCL, ptrCL);
        // convert to NC4HW4
        if (mNumQuantBit == 8) {
            // ROUND_UP(IC, 4), UP_DIV(OC, 4) * mKernelWidth * mKernelHeight
            mResource->mFilter.reset(Tensor::createDevice<int8_t>({1, filterImageShape[1], 1, 4 * filterImageShape[0]}));
            mOpenCLBackend->onAcquireBuffer(mResource->mFilter.get(), Backend::STATIC);
            MNN::OpenCL::BufferConvertor bufferConvertor{mOpenCLBackend->getOpenCLRuntime()};
            // filterBuffer shape: {OC, ROUND_UP(IC, 4), mKernelWidth, mKernelHeight}
            bufferConvertor.convertToNC4HW4Buffer(filterBuffer.get(), MNN::OpenCL::CONV2D_FILTER, mResource->mFilter.get(), false, true, mLowMemoryFlag, mNumQuantBit);
        } else if (mNumQuantBit == 4){
            // ROUND_UP(IC, 4), UP_DIV(OC, 4) * mKernelWidth * mKernelHeight
            // For int4 case, data stored in mFilter should be uint8_t,
            // while "Tensor::createDevice<uint8_t>" occupies more memory than "Tensor::createDevice<int8_t>".
            // Therefore, we use "Tensor::createDevice<int8_t>" currently, leaving "Tensor::createDevice<uint8_t>" to be supported.
            mResource->mFilter.reset(Tensor::createDevice<int8_t>({1, filterImageShape[1], 1, 2 * filterImageShape[0]}));
            mOpenCLBackend->onAcquireBuffer(mResource->mFilter.get(), Backend::STATIC);
            MNN::OpenCL::BufferConvertor bufferConvertor{mOpenCLBackend->getOpenCLRuntime()};
            // filterBuffer shape: {OC, ROUND_UP(IC, 4), mKernelWidth, mKernelHeight}
            bufferConvertor.convertToNC4HW4Buffer(filterBuffer.get(), MNN::OpenCL::CONV2D_FILTER, mResource->mFilter.get(), false, true, mLowMemoryFlag, mNumQuantBit);
        } else {/* More types to be supported. */}
    } else {
        MNN_ERROR("GetConvParams Error: filterDataPtr == nullptr. \n");
        MNN_ASSERT(false);
    }
}
// select the fastest kernel for the general cases by tuning
void ConvBufLowMemoryExecution::tuneGeneralCaseLowMemory(Tensor * input, Tensor * output) {
    auto &unit = mUnits[0];
    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);
    const int height             = outputShape.at(1);
    const int width              = outputShape.at(2);
    const int outChannel         = outputShape.at(3);
    const int inputHeight   = inputShape.at(1);
    const int inputWidth    = inputShape.at(2);
    const int inputChannels = inputShape.at(3);
    const int inputChannelBlocks = UP_DIV(inputChannels, 4);
    std::string info = std::to_string(inputChannels) + "_" + std::to_string(outChannel) + "_" + std::to_string(mResource->mKernelHeight) + "_" + std::to_string(mResource->mKernelWidth) + "_" + std::to_string(mResource->mStrides[0]) + "_" + std::to_string(mResource->mStrides[1]) + "_" + std::to_string(mResource->mDilations[0]) + "_" + std::to_string(mResource->mDilations[1]);
    int inputImageShape[2]  = {inputHeight, inputWidth};
    int outputImageShape[2] = {height, width};
    int kernelShape[2]      = {mResource->mKernelHeight, mResource->mKernelWidth};
    int strideShape[2]      = {mResource->mStrides[0], mResource->mStrides[1]};
    int paddingShape[2]     = {mPaddings[0], mPaddings[1]};
    int dilationShape[2]    = {mResource->mDilations[0], mResource->mDilations[1]};
    // {"conv_2d_c4h1w2", "conv_2d_c4h1w1", "conv_2d_c8h1w1", "conv_2d_c4h1w4", "conv_2d_c8h2w1", "conv_2d_c4h4w1"};
    const int total_kernel = 7;
    std::string kernelName[total_kernel] = {"conv_2d_int_c4h1w1", "conv_2d_int_c4h1w2", "conv_2d_int_c4h4w1", "conv_2d_int_c8h2w1", "conv_2d_int_c8h4w1", "conv_2d_int_c4h1w4", "conv_2d_int_c8h1w4"};
    int itemC[total_kernel] = {4, 4, 4, 8, 8, 4, 8};
    int itemH[total_kernel] = {1, 1, 4, 2, 4, 1, 1};
    int itemW[total_kernel] = {1, 2, 1, 1, 1, 4, 4};
    int actual_kernel = total_kernel;
    std::shared_ptr<KernelWrap> kernel[total_kernel];
    std::vector<uint32_t> globalWorkSize[total_kernel];
    std::vector<uint32_t> localWorkSize[total_kernel];
    std::pair<int, int> min_cost(INT_MAX, 0);//(min_time, min_index)
    // MNN_PRINT("Checking kernel %d.\n", knlCheck);
    for (int knl_idx = 0; knl_idx < actual_kernel; knl_idx++) {
        std::set<std::string> buildOption = mResource->mBuildOptions;
        if(outputShape.at(3) % itemC[knl_idx] != 0){
            buildOption.emplace("-DCHANNEL_LEAVE");
        }
        if((outputShape.at(2) % itemW[knl_idx]) != 0 || (outputShape.at(1) % itemH[knl_idx]) != 0){
            buildOption.emplace("-DBLOCK_LEAVE");
        }
        kernel[knl_idx]        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("conv_2d_int_buf", kernelName[knl_idx], buildOption);
        uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(kernel[knl_idx]));

        globalWorkSize[knl_idx] = {static_cast<uint32_t>(UP_DIV(outputShape.at(3), itemC[knl_idx]) * UP_DIV(outputShape.at(2), itemW[knl_idx])), static_cast<uint32_t>(outputShape.at(0) * UP_DIV(outputShape.at(1), itemH[knl_idx]))};
        uint32_t idx            = 0;
        cl_int ret = CL_SUCCESS;
        ret |= kernel[knl_idx]->get().setArg(idx++, globalWorkSize[knl_idx][0]);
        ret |= kernel[knl_idx]->get().setArg(idx++, globalWorkSize[knl_idx][1]);
        ret |= kernel[knl_idx]->get().setArg(idx++, openCLBuffer(input));
        ret |= kernel[knl_idx]->get().setArg(idx++, openCLBuffer(mResource->mFilter.get()));
        ret |= kernel[knl_idx]->get().setArg(idx++, openCLBuffer(mResource->dequantScale.get()));
        ret |= kernel[knl_idx]->get().setArg(idx++, openCLBuffer(mResource->dequantOffset.get()));
        ret |= kernel[knl_idx]->get().setArg(idx++, openCLBuffer(mResource->mBias.get()));
        ret |= kernel[knl_idx]->get().setArg(idx++, openCLBuffer(output));
        ret |= kernel[knl_idx]->get().setArg(idx++, sizeof(inputImageShape), inputImageShape);
        ret |= kernel[knl_idx]->get().setArg(idx++, inputChannels);
        ret |= kernel[knl_idx]->get().setArg(idx++, inputChannelBlocks);
        ret |= kernel[knl_idx]->get().setArg(idx++, sizeof(outputImageShape), outputImageShape);
        ret |= kernel[knl_idx]->get().setArg(idx++, sizeof(kernelShape), kernelShape);
        ret |= kernel[knl_idx]->get().setArg(idx++, sizeof(strideShape), strideShape);
        ret |= kernel[knl_idx]->get().setArg(idx++, sizeof(paddingShape), paddingShape);
        ret |= kernel[knl_idx]->get().setArg(idx++, sizeof(dilationShape), dilationShape);
        ret |= kernel[knl_idx]->get().setArg(idx++, UP_DIV(width, itemW[knl_idx]));
        ret |= kernel[knl_idx]->get().setArg(idx++, UP_DIV(outChannel, 4));
        ret |= kernel[knl_idx]->get().setArg(idx++, UP_DIV(height, itemH[knl_idx]));
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

    std::set<std::string> buildOption = mResource->mBuildOptions;
    if(outputShape.at(3) % itemC[min_index] != 0){
        buildOption.emplace("-DCHANNEL_LEAVE");
    }
    if((outputShape.at(2) % itemW[min_index]) != 0 || (outputShape.at(1) % itemH[min_index]) != 0){
        buildOption.emplace("-DBLOCK_LEAVE");
    }
    unit.kernel        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("conv_2d_int_buf", kernelName[min_index], buildOption);

    uint32_t idx            = 0;
    cl_int ret = CL_SUCCESS;
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[0]);
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[1]);
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(input));
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mResource->mFilter.get()));
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mResource->dequantScale.get()));
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mResource->dequantOffset.get()));
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mResource->mBias.get()));
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(output));
    ret |= unit.kernel->get().setArg(idx++, sizeof(inputImageShape), inputImageShape);
    ret |= unit.kernel->get().setArg(idx++, inputChannels);
    ret |= unit.kernel->get().setArg(idx++, inputChannelBlocks);
    ret |= unit.kernel->get().setArg(idx++, sizeof(outputImageShape), outputImageShape);
    ret |= unit.kernel->get().setArg(idx++, sizeof(kernelShape), kernelShape);
    ret |= unit.kernel->get().setArg(idx++, sizeof(strideShape), strideShape);
    ret |= unit.kernel->get().setArg(idx++, sizeof(paddingShape), paddingShape);
    ret |= unit.kernel->get().setArg(idx++, sizeof(dilationShape), dilationShape);
    ret |= unit.kernel->get().setArg(idx++, UP_DIV(width, itemW[min_index]));
    ret |= unit.kernel->get().setArg(idx++, UP_DIV(outChannel, 4));
    ret |= unit.kernel->get().setArg(idx++, UP_DIV(height, itemH[min_index]));
    MNN_CHECK_CL_SUCCESS(ret, "setArg ConvBufLowMemory");
    mOpenCLBackend->recordKernel2d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
    unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1]};
    unit.localWorkSize = {mLocalWorkSize[0], mLocalWorkSize[1]};
    return;
}
void ConvBufLowMemoryExecution::tuneGemmLowMemory(Tensor * input, Tensor * output) {
    auto &unit = mUnits[0];
    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);
    const int outChannel = outputShape.at(3);
    const int inputChannels = inputShape.at(3);
    const int batch = outputShape.at(0);
    const int height = outputShape.at(1);
    const int width = outputShape.at(2);
    const int inputChannelBlocks = UP_DIV(inputChannels, 4);
    const int outputChannelBlocks = UP_DIV(outChannel, 4);

    int global_y = batch * height;
    const int total_kernel = 5;
    std::string kernelName[total_kernel] = {"gemm_conv_c1_buf", "gemm_conv_c2_buf", "gemm_conv_c4_buf", "gemm_conv_c1_image",  "gemm_conv_c2_image"};
    int itemC[total_kernel] = {1, 2, 4, 1, 2};
    int actual_kernel = total_kernel;
    std::shared_ptr<KernelWrap> kernel[total_kernel];
    std::vector<uint32_t> globalWorkSize[total_kernel];
    std::vector<uint32_t> localWorkSize[total_kernel];
    std::pair<int, int> min_cost(INT_MAX, 0);//(min_time, min_index)
    std::set<std::string> buildOption = mResource->mBuildOptions;
    if(width == 1 && height == 1){
        buildOption.emplace("-DWIDTH_HEIGHT_1");
    }
    
    if(inputChannels % 16 != 0){
        buildOption.emplace("-DINPUT_CHANNEL_LEAVE");
    } else if (mResource->mUseImage && mNumQuantBit == 4 && inputChannels % 32 != 0) {
        // Image weight-int4 use load32
        buildOption.emplace("-DINPUT_CHANNEL_LEAVE");
    }
    std::string info = std::to_string(inputChannels) + "_" + std::to_string(outChannel);
    if(batch > 1){
        global_y = UP_DIV(batch, 4) * height;
        buildOption.emplace("-DBACTH_BLOCK4");
        info += "_BATCH_BLOCK4";
    }
    int knl_idx = 0;
    actual_kernel = 3;
    if(mResource->mUseImage){
        knl_idx = 3;
        actual_kernel = total_kernel;
    }
    for (; knl_idx < actual_kernel; knl_idx++) {
        kernel[knl_idx]        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("gemv_conv1x1_buf", kernelName[knl_idx], buildOption);
        uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(kernel[knl_idx]));
        
        globalWorkSize[knl_idx] = {static_cast<uint32_t>(UP_DIV(outChannel, itemC[knl_idx]) * width), static_cast<uint32_t>(global_y)};
        uint32_t idx            = 0;
        cl_int ret = CL_SUCCESS;
        ret |= kernel[knl_idx]->get().setArg(idx++, globalWorkSize[knl_idx][0]);
        ret |= kernel[knl_idx]->get().setArg(idx++, globalWorkSize[knl_idx][1]);
        ret |= kernel[knl_idx]->get().setArg(idx++, openCLBuffer(input));
        if(mResource->mUseImage){
            ret |= kernel[knl_idx]->get().setArg(idx++, *mResource->mKernelImage.get());
        }else{
            ret |= kernel[knl_idx]->get().setArg(idx++, *mResource->mKernelBuffer.get());
        }
        ret |= kernel[knl_idx]->get().setArg(idx++, openCLBuffer(mResource->dequantScale.get()));
        ret |= kernel[knl_idx]->get().setArg(idx++, openCLBuffer(mResource->dequantOffset.get()));
        ret |= kernel[knl_idx]->get().setArg(idx++, openCLBuffer(mResource->mBias.get()));
        ret |= kernel[knl_idx]->get().setArg(idx++, openCLBuffer(output));
        ret |= kernel[knl_idx]->get().setArg(idx++, static_cast<int>(outputChannelBlocks));
        ret |= kernel[knl_idx]->get().setArg(idx++, static_cast<int>(inputChannelBlocks));
        ret |= kernel[knl_idx]->get().setArg(idx++, static_cast<int>(batch));
        ret |= kernel[knl_idx]->get().setArg(idx++, static_cast<int>(height));
        ret |= kernel[knl_idx]->get().setArg(idx++, static_cast<int>(width));
        MNN_CHECK_CL_SUCCESS(ret, "setArg gemv_conv1x1_buf Kernel Select");
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
    
    
    unit.kernel        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("gemv_conv1x1_buf", kernelName[min_index], buildOption);
    //MNN_PRINT("Kernel is %d.\n", min_index);
    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[0]);
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[1]);
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(input));
    if(mResource->mUseImage){
        ret |= unit.kernel->get().setArg(idx++, *mResource->mKernelImage.get());
    }else{
        ret |= unit.kernel->get().setArg(idx++, *mResource->mKernelBuffer.get());
    }
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mResource->dequantScale.get()));
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mResource->dequantOffset.get()));
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mResource->mBias.get()));
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(output));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(outputChannelBlocks));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(inputChannelBlocks));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(batch));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(height));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(width));
    MNN_CHECK_CL_SUCCESS(ret, "setArg gemv_conv1x1_buf");
    mOpenCLBackend->recordKernel2d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
    unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1]};
    unit.localWorkSize = {mLocalWorkSize[0], mLocalWorkSize[1]};
    return;
}
ConvBufLowMemoryExecution::ConvBufLowMemoryExecution(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, const MNN::Op *op, Backend *backend)
    : ConvBufCommonExecution(op->main_as_Convolution2D(), backend), CommonExecution(backend, op) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start ConvBufLowMemoryExecution init !\n");
#endif
    mOpenCLBackend                 = static_cast<OpenCLBackend *>(backend);
    const auto *conv2dParams       = op->main_as_Convolution2D();
    const auto *conv2dCommonParams = conv2dParams->common();
    mResource->mConv2dParams       = conv2dParams;
    mResource->mConv2dCommonParams  = conv2dCommonParams;
    mResource->mStrides                       = {conv2dCommonParams->strideY(), conv2dCommonParams->strideX()};
    mResource->mDilations                     = {conv2dCommonParams->dilateY(), conv2dCommonParams->dilateX()};
    auto padding = ConvolutionCommon::convolutionPad(inputs[0], outputs[0], conv2dCommonParams);
    mPaddings[0] = padding.second;//padY
    mPaddings[1] = padding.first;//padX

    mResource->mKernelWidth   = conv2dCommonParams->kernelX();
    mResource->mKernelHeight  = conv2dCommonParams->kernelY();
    mResource->mOutputChannel = conv2dCommonParams->outputCount();
    mResource->mInputChannel = conv2dCommonParams->inputCount();
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
    // set mDequantScale, mDequantOffset, mFilterDataPtr
    // prepare mDequantScale mDequantOffset mFilterDataPtr
    getInfoFromOpLowMemory(quanCommon);
    //select opt conv method
    if (mResource->mKernelHeight == mResource->mKernelWidth && mResource->mKernelHeight == 1 && mResource->mStrides[0] == 1 && mResource->mStrides[1] == 1 && conv2dCommonParams->padX() == 0 && conv2dCommonParams->padY() == 0 && conv2dCommonParams->dilateX() == 1 && conv2dCommonParams->dilateY() == 1) {
        set1x1WeightLowMemory(4, 16, mFilterDataPtr, quanCommon);
        mResource->mConv1x1Opt = true;
    }else {
        // set mFilter for not 1x1 case
        setGeneralWeightLowMemory(mFilterDataPtr, quanCommon);
    }
    // Create Kernel
    if (conv2dCommonParams->relu()) {
        mResource->mBuildOptions.emplace("-DRELU");
    } else if (conv2dCommonParams->relu6()) {
        mResource->mBuildOptions.emplace("-DRELU6");
    }
    if (mNumQuantBit == 8) {
        // int8 case
        mResource->mBuildOptions.emplace("-DUSE_LOW_BIT_WEIGHT_INT8");
    } else if (mNumQuantBit == 4){
        // int4 case
        mResource->mBuildOptions.emplace("-DUSE_LOW_BIT_WEIGHT_INT4");
    } else {/* More types to be supported. */}
#ifdef LOG_VERBOSE
    MNN_PRINT("end ConvExecution init !\n");
#endif
}

ConvBufLowMemoryExecution::ConvBufLowMemoryExecution(std::shared_ptr<ConvBufResource> resource, const MNN::Op* op, Backend *backend)
    : ConvBufCommonExecution(backend), CommonExecution(backend, op) {
    mResource = resource;
    const auto *conv2dParams       = op->main_as_Convolution2D();
    const auto *conv2dCommonParams = conv2dParams->common();
    mResource->mConv2dParams       = conv2dParams;
    mResource->mConv2dCommonParams  = conv2dCommonParams;
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

ErrorCode ConvBufLowMemoryExecution::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start ConvExecution onResize !\n");
#endif
    mUnits.resize(1);
    auto input  = inputs[0];
    auto output = outputs[0];
    auto padding = ConvolutionCommon::convolutionPad(input, output, mResource->mConv2dCommonParams);
    mPaddings[0] = padding.second;//padY
    mPaddings[1] = padding.first;//padX
    // onclone default use conv1x1Opt, need reset
    if (mResource->mConv1x1Opt) {
        tuneGemmLowMemory(input, output);
    } else {
        tuneGeneralCaseLowMemory(input, output);
    }
#ifdef LOG_VERBOSE
    MNN_PRINT("end ConvExecution onResize !\n");
#endif
    return NO_ERROR;
}
} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
#endif /* MNN_LOW_MEMORY */
