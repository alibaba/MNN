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
    mResource->mInputChannel = quanCommon->weight.size() / (mResource->mKernelWidth * mResource->mKernelHeight * mResource->mOutputChannel);
    // set mResource->mNumQuantBit
    if(quanCommon->canUseInt4){
        mResource->mNumQuantBit = 4;
    }else{
        mResource->mNumQuantBit = 8;
    }
    // src of alpha in CPU
    float * dequantAlpha = quanCommon->alpha.get();
    int totalCount = quanCommon->alpha.size();
    if (quanCommon->asymmetric) {
        totalCount /= 2;
    }
    int numAlpha = mResource->mOutputChannel;
    mResource->mBlockSize = totalCount / numAlpha;
    // set mDequantScale mDequantOffset
    int numAlphaPack = ROUND_UP(numAlpha, 4);

    mResource->dequantScaleOffset.reset(Tensor::createDevice<int32_t>({mResource->mBlockSize, numAlphaPack, 2}));
    mOpenCLBackend->onAcquireBuffer(mResource->dequantScaleOffset.get(), Backend::STATIC);
    cl::Buffer &dequantScaleOffsetBuffer = openCLBuffer(mResource->dequantScaleOffset.get());
    // transfer data from src in cpu to dst in gpu
    int fpBytes = mOpenCLBackend->fpBytes();
    cl_int resBias, resScaleOffset;

    int mapSize = mResource->mBlockSize * numAlphaPack * sizeof(int32_t) * 2;
    void * dequantScaleOffsetBufferMap = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(dequantScaleOffsetBuffer, true, CL_MAP_WRITE, 0, mapSize, nullptr, nullptr, &resScaleOffset);
    // mBlockSize % 4 need equal 0
    if (dequantScaleOffsetBufferMap != nullptr && resScaleOffset == CL_SUCCESS) {
        if (quanCommon->asymmetric) {
            for (int i = 0; i < numAlpha; ++i) {
                auto srcZ = dequantAlpha + i * mResource->mBlockSize * 2;
                for(int j = 0; j < mResource->mBlockSize; ++j){
                    float o = srcZ[2*j+0];
                    float s = srcZ[2*j+1];
                    ((float *)dequantScaleOffsetBufferMap)[(j * numAlphaPack + i) * 2] = s;
                    ((float *)dequantScaleOffsetBufferMap)[(j * numAlphaPack + i) * 2 + 1] = o;
                }
            }
        } else {
            for (int i = 0; i < numAlpha; ++i) {
                auto srcZ = dequantAlpha + i * mResource->mBlockSize;
                for(int j = 0; j < mResource->mBlockSize; ++j){
                    ((float *)dequantScaleOffsetBufferMap)[(j * numAlphaPack + i) * 2] = srcZ[j];
                    ((float *)dequantScaleOffsetBufferMap)[(j * numAlphaPack + i) * 2 + 1] = 0.0f;
                }
            }
        }
    } else {
        MNN_ERROR("Map error dequantBufferMap == nullptr \n");
        MNN_ASSERT(false);
    }
    mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(dequantScaleOffsetBuffer, dequantScaleOffsetBufferMap);
    // set mFilterDataPtr
    mFilterDataPtr = (void *)quanCommon->weight.get();
}

bool ConvBufLowMemoryExecution::convertToQuantWeight1x1Buffer(cl::Buffer input, int pack) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start convertToQuantWeight1x1Buffer !\n");
#endif
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    std::string kernelName = "conv2d_1x1_weight_quant_buffer";
    if(mResource->mUseImage){
        kernelName = "conv2d_1x1_weight_quant_image";
    }
    std::set<std::string> buildOptions;
    if (mResource->mNumQuantBit == 8) {
        buildOptions.emplace("-DUSE_LOW_BIT_WEIGHT_INT8");
    } else if (mResource->mNumQuantBit == 4){
        // int4 case
        buildOptions.emplace("-DUSE_LOW_BIT_WEIGHT_INT4");
    } else {/* More types to be supported. */}
    if(mResource->mInputChannel % pack != 0){
        buildOptions.emplace("-DINPUT_CHANNEL_LEAVE");
    }
    
    mBufferToConv1x1Kernel = runtime->buildKernelWithCache("buffer_convert_quant", kernelName, buildOptions);
    auto kernel = mBufferToConv1x1Kernel->get();
    uint32_t gws[2] = {static_cast<uint32_t>(UP_DIV(mResource->mInputChannel, pack)), static_cast<uint32_t>(mResource->mOutputChannel)};

    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= kernel.setArg(idx++, gws[0]);
    ret |= kernel.setArg(idx++, gws[1]);
    ret |= kernel.setArg(idx++, input);
    if(mResource->mUseImage){
        ret |= kernel.setArg(idx++, *mResource->mKernelImage.get());
    }else{
        ret |= kernel.setArg(idx++, *mResource->mKernelBuffer.get());
    }
    ret |= kernel.setArg(idx++, mResource->mInputChannel);
    ret |= kernel.setArg(idx++, mResource->mOutputChannel);
    MNN_CHECK_CL_SUCCESS(ret, "setArg convertToQuantWeight1x1Buffer");

    const uint32_t maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mBufferToConv1x1Kernel));
    const std::vector<uint32_t> lws = {16, std::max((uint32_t)1, maxWorkGroupSize / 16)};

    cl::Event event;
    cl_int res;

    std::vector<uint32_t> roundUpGroupWorkSize(lws.size());
    for (size_t i = 0; i < lws.size(); ++i) {
        roundUpGroupWorkSize[i] = ROUND_UP(gws[i], lws[i]);
    }

    res = runtime->commandQueue().enqueueNDRangeKernel(kernel, cl::NullRange,
                                                         cl::NDRange(roundUpGroupWorkSize[0], roundUpGroupWorkSize[1]),
                                                         cl::NDRange(lws[0], lws[1]), nullptr, &event);
    
    event.wait();
    MNN_CHECK_CL_SUCCESS(res, "convertToQuantWeight1x1Buffer");

#ifdef LOG_VERBOSE
    MNN_PRINT("end convertToQuantWeight1x1Buffer !\n");
#endif
    return true;
}

// set mKernelBuffer for the 1x1 kernels
void ConvBufLowMemoryExecution::set1x1WeightLowMemory(int packCout, int packCin, void * filterDataPtr, std::shared_ptr<ConvolutionCommon::Int8Common> & quanCommon) {
    cl_int res;
    std::shared_ptr<Tensor> filterBuffer(Tensor::createDevice<float>({ROUND_UP(mResource->mOutputChannel, packCout), ROUND_UP(mResource->mInputChannel, packCin), mResource->mKernelWidth, mResource->mKernelHeight}));
    size_t buffer_size = filterBuffer->usize() / sizeof(float);
    size_t cpy_size = mResource->mOutputChannel * mResource->mInputChannel * mResource->mKernelWidth * mResource->mKernelHeight * sizeof(char);
    float *dequantAlpha = quanCommon->alpha.get();
    cl::Buffer filterBufferCL(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size);
    void *mapPtr = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(filterBufferCL, true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &res);
    if(mapPtr != nullptr && res == CL_SUCCESS){
        ::memcpy(mapPtr, filterDataPtr, cpy_size);
    } else {
        MNN_ERROR("set1x1WeightLowMemory: Map error ptrCL == nullptr \n");
        MNN_ASSERT(false);
    }
    mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(filterBufferCL, mapPtr);
    // shared part for all cases
    if (mResource->mNumQuantBit == 8) {
        // int8 case
        buffer_size *= sizeof(int8_t);
    } else if (mResource->mNumQuantBit == 4){
        // int4 case
        buffer_size /= 2;
    } else {/* More types to be supported. */}
    
    // Use Image load weights
    if(UP_DIV(mResource->mInputChannel, packCin) <= 16384 && ROUND_UP(mResource->mOutputChannel, packCout) <= 16384){
        mResource->mUseImage = true;
    }
    if(mResource->mUseImage) {
        if(mResource->mNumQuantBit == 4){
            packCin *= 2;
        }
        size_t w = ROUND_UP(mResource->mOutputChannel, packCout);
        size_t h = UP_DIV(mResource->mInputChannel, packCin);
        mResource->mKernelImage.reset(new cl::Image2D(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_SIGNED_INT32), w, h, 0, nullptr, &res));
        if (nullptr == mResource->mKernelImage.get() || res != CL_SUCCESS) {
            MNN_ERROR("Alloc Image %d x %d error, code:%d \n", (int)w, (int)h, (int)res);
        }
    } else{
        mResource->mKernelBuffer.reset(new cl::Buffer(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size));
    }
    convertToQuantWeight1x1Buffer(filterBufferCL, packCin);
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
                int ic = 0;
                for(; ic<mResource->mInputChannel; ic++) {
                    ::memcpy((int8_t *)ptrCL + (oc * ROUND_UP(mResource->mInputChannel, 4) + ic) * mResource->mKernelWidth * mResource->mKernelHeight, ((int8_t *)filterDataPtr) + (oc * mResource->mInputChannel + ic) * mResource->mKernelWidth * mResource->mKernelHeight, copy_size);
                }
            }
        } else {
            MNN_ERROR("setGeneralWeightLowMemory: Map error ptrCL == nullptr \n");
        }
        mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(filterBufferCL, ptrCL);
        // convert to NC4HW4
        if (mResource->mNumQuantBit == 8) {
            // ROUND_UP(IC, 4), UP_DIV(OC, 4) * mKernelWidth * mKernelHeight
            mResource->mFilter.reset(Tensor::createDevice<int8_t>({1, filterImageShape[1], 1, 4 * filterImageShape[0]}));
            mOpenCLBackend->onAcquireBuffer(mResource->mFilter.get(), Backend::STATIC);
            MNN::OpenCL::BufferConvertor bufferConvertor{mOpenCLBackend->getOpenCLRuntime()};
            // filterBuffer shape: {OC, ROUND_UP(IC, 4), mKernelWidth, mKernelHeight}
            bufferConvertor.convertToNC4HW4Buffer(filterBuffer.get(), MNN::OpenCL::CONV2D_FILTER, mResource->mFilter.get(), false, true, mLowMemoryFlag, mResource->mNumQuantBit);
        } else if (mResource->mNumQuantBit == 4){
            // ROUND_UP(IC, 4), UP_DIV(OC, 4) * mKernelWidth * mKernelHeight
            // For int4 case, data stored in mFilter should be uint8_t,
            // while "Tensor::createDevice<uint8_t>" occupies more memory than "Tensor::createDevice<int8_t>".
            // Therefore, we use "Tensor::createDevice<int8_t>" currently, leaving "Tensor::createDevice<uint8_t>" to be supported.
            mResource->mFilter.reset(Tensor::createDevice<int8_t>({1, filterImageShape[1], 1, 2 * filterImageShape[0]}));
            mOpenCLBackend->onAcquireBuffer(mResource->mFilter.get(), Backend::STATIC);
            MNN::OpenCL::BufferConvertor bufferConvertor{mOpenCLBackend->getOpenCLRuntime()};
            // filterBuffer shape: {OC, ROUND_UP(IC, 4), mKernelWidth, mKernelHeight}
            bufferConvertor.convertToNC4HW4Buffer(filterBuffer.get(), MNN::OpenCL::CONV2D_FILTER, mResource->mFilter.get(), false, true, mLowMemoryFlag, mResource->mNumQuantBit);
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
    const int blockDim = mResource->mInputChannel / mResource->mBlockSize;
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
        if(inputChannels % 4 != 0){
            buildOption.emplace("-DINPUT_CHANNEL_LEAVE");
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
        ret |= kernel[knl_idx]->get().setArg(idx++, openCLBuffer(mResource->dequantScaleOffset.get()));
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
        ret |= kernel[knl_idx]->get().setArg(idx++, blockDim);
        MNN_CHECK_CL_SUCCESS(ret, "setArg ConvBufLowMemory Kernel Select");
        std::pair<std::vector<uint32_t>, int> retTune;
        retTune = localWS2DDefault(globalWorkSize[knl_idx], maxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), kernelName[knl_idx] + info, kernel[knl_idx]);
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
    if(inputChannels % 4 != 0){
        buildOption.emplace("-DINPUT_CHANNEL_LEAVE");
    }
    unit.kernel        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("conv_2d_int_buf", kernelName[min_index], buildOption);

    uint32_t idx            = 0;
    cl_int ret = CL_SUCCESS;
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[0]);
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[1]);
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(input));
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mResource->mFilter.get()));
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mResource->dequantScaleOffset.get()));
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
    ret |= unit.kernel->get().setArg(idx++, blockDim);
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
    const int blockNum = mResource->mBlockSize;
    const int blockDim = mResource->mInputChannel / mResource->mBlockSize;

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
    
    if(blockDim % 16 != 0){
        buildOption.emplace("-DINPUT_CHANNEL_LEAVE");
    } else if (mResource->mUseImage && mResource->mNumQuantBit == 4 && blockDim % 32 != 0) {
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
        ret |= kernel[knl_idx]->get().setArg(idx++, openCLBuffer(mResource->dequantScaleOffset.get()));
        ret |= kernel[knl_idx]->get().setArg(idx++, openCLBuffer(mResource->mBias.get()));
        ret |= kernel[knl_idx]->get().setArg(idx++, openCLBuffer(output));
        ret |= kernel[knl_idx]->get().setArg(idx++, static_cast<int>(outputChannelBlocks));
        ret |= kernel[knl_idx]->get().setArg(idx++, static_cast<int>(inputChannelBlocks));
        ret |= kernel[knl_idx]->get().setArg(idx++, inputChannels);
        ret |= kernel[knl_idx]->get().setArg(idx++, static_cast<int>(batch));
        ret |= kernel[knl_idx]->get().setArg(idx++, static_cast<int>(height));
        ret |= kernel[knl_idx]->get().setArg(idx++, static_cast<int>(width));
        ret |= kernel[knl_idx]->get().setArg(idx++, static_cast<int>(blockNum));
        ret |= kernel[knl_idx]->get().setArg(idx++, static_cast<int>(blockDim));
        MNN_CHECK_CL_SUCCESS(ret, "setArg gemv_conv1x1_buf Kernel Select");
        std::pair<std::vector<uint32_t>, int> retTune;
        retTune = localWS2DDefault(globalWorkSize[knl_idx], maxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), kernelName[knl_idx] + info, kernel[knl_idx]);
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
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mResource->dequantScaleOffset.get()));
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mResource->mBias.get()));
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(output));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(outputChannelBlocks));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(inputChannelBlocks));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(inputChannels));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(batch));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(height));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(width));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(blockNum));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(blockDim));
    MNN_CHECK_CL_SUCCESS(ret, "setArg gemv_conv1x1_buf");
    mOpenCLBackend->recordKernel2d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
    unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1]};
    unit.localWorkSize = {mLocalWorkSize[0], mLocalWorkSize[1]};
    return;
}
void ConvBufLowMemoryExecution::tuneGemvBatchLowMemory(Tensor * input, Tensor * output) {
    mUnits.resize(3);
    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);
    const int outChannel = outputShape.at(3);
    const int inputChannels = inputShape.at(3);
    const int batch = outputShape.at(0);
    const int width_height = outputShape.at(1) * outputShape.at(2);
    const int inputChannelBlocks = UP_DIV(inputChannels, 4);
    const int outputChannelBlocks = UP_DIV(outChannel, 4);
    const int blockNum = mResource->mBlockSize;
    const int blockDim = mResource->mInputChannel / mResource->mBlockSize;
    
    int global_y = UP_DIV(batch, 4) * width_height;
    const int total_kernel = 5;
    std::string kernelName[total_kernel] = {"gemm_b4_c1_buf", "gemm_b4_c2_buf", "gemm_b4_c4_buf", "gemm_b4_c1_image",  "gemm_b4_c2_image"};
    int itemC[total_kernel] = {1, 2, 4, 1, 2};
    int actual_kernel = total_kernel;
    std::shared_ptr<KernelWrap> kernel[total_kernel];
    std::vector<uint32_t> globalWorkSize[total_kernel];
    std::vector<uint32_t> localWorkSize[total_kernel];
    std::pair<int, int> min_cost(INT_MAX, 0);//(min_time, min_index)
    std::set<std::string> buildOption = mResource->mBuildOptions;
    if(blockDim % 16 != 0){
        buildOption.emplace("-DINPUT_CHANNEL_LEAVE");
    } else if (mResource->mUseImage && mResource->mNumQuantBit == 4 && blockDim % 32 != 0) {
        // Image weight-int4 use load32
        buildOption.emplace("-DINPUT_CHANNEL_LEAVE");
    }
    std::string info = std::to_string(inputChannels) + "_" + std::to_string(outChannel);
    // mResource->mInputChannel ROUND_UP to blockDim, avoid gemm overstep
    mConvGemmInpTensor.reset(Tensor::createDevice<float>({ROUND_UP(batch, 4) * ROUND_UP(ROUND_UP(mResource->mInputChannel, 4), blockDim) * width_height}));
    mConvGemmOutTensor.reset(Tensor::createDevice<float>({ROUND_UP(batch, 4) * ROUND_UP(mResource->mOutputChannel, 4) * width_height}));
    mOpenCLBackend->onAcquireBuffer(mConvGemmInpTensor.get(), Backend::DYNAMIC);
    mOpenCLBackend->onAcquireBuffer(mConvGemmOutTensor.get(), Backend::DYNAMIC);
    mOpenCLBackend->onReleaseBuffer(mConvGemmOutTensor.get(), Backend::DYNAMIC);
    mOpenCLBackend->onReleaseBuffer(mConvGemmInpTensor.get(), Backend::DYNAMIC);

    // reshape n*c/4*4*hw -> n/4*hw*c*4
    {
        auto &unit = mUnits[0];
        mGlobalWorkSize = {static_cast<uint32_t>(UP_DIV(mResource->mInputChannel, 4)), static_cast<uint32_t>(UP_DIV(batch, 4)), static_cast<uint32_t>(width_height)};
        unit.kernel        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("gemm_quant_batch_buf", "reshape_nchw4_nhwc4", buildOption);
        uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(unit.kernel));
        
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[0]);
        ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[1]);
        ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[2]);
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(input));
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mConvGemmInpTensor.get()));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(width_height));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(batch));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(inputChannels));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(inputChannelBlocks));
        MNN_CHECK_CL_SUCCESS(ret, "setArg reshape_nc4_cn4");
        mLocalWorkSize = localWS3DDefault(mGlobalWorkSize, maxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), "reshape_nchw4_nhwc4", unit.kernel).first;
        mOpenCLBackend->recordKernel3d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
        unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
        unit.localWorkSize = {mLocalWorkSize[0], mLocalWorkSize[1], mLocalWorkSize[2]};
    }
    // gemm
    {
        auto &unit = mUnits[1];
        int knl_idx = 0;
        actual_kernel = 3;
        if(mResource->mUseImage){
            knl_idx = 3;
            actual_kernel = total_kernel;
        }
        for (; knl_idx < actual_kernel; knl_idx++) {
            kernel[knl_idx]        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("gemm_quant_batch_buf", kernelName[knl_idx], buildOption);
            uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(kernel[knl_idx]));
            
            globalWorkSize[knl_idx] = {static_cast<uint32_t>(UP_DIV(outChannel, itemC[knl_idx])), static_cast<uint32_t>(global_y)};
            uint32_t idx            = 0;
            cl_int ret = CL_SUCCESS;
            ret |= kernel[knl_idx]->get().setArg(idx++, globalWorkSize[knl_idx][0]);
            ret |= kernel[knl_idx]->get().setArg(idx++, globalWorkSize[knl_idx][1]);
            ret |= kernel[knl_idx]->get().setArg(idx++, openCLBuffer(mConvGemmInpTensor.get()));
            if(mResource->mUseImage){
                ret |= kernel[knl_idx]->get().setArg(idx++, *mResource->mKernelImage.get());
            }else{
                ret |= kernel[knl_idx]->get().setArg(idx++, *mResource->mKernelBuffer.get());
            }
            ret |= kernel[knl_idx]->get().setArg(idx++, openCLBuffer(mResource->dequantScaleOffset.get()));
            ret |= kernel[knl_idx]->get().setArg(idx++, openCLBuffer(mResource->mBias.get()));
            ret |= kernel[knl_idx]->get().setArg(idx++, openCLBuffer(mConvGemmOutTensor.get()));
            ret |= kernel[knl_idx]->get().setArg(idx++, static_cast<int>(outputChannelBlocks));
            ret |= kernel[knl_idx]->get().setArg(idx++, static_cast<int>(inputChannelBlocks));
            ret |= kernel[knl_idx]->get().setArg(idx++, static_cast<int>(blockNum));
            ret |= kernel[knl_idx]->get().setArg(idx++, static_cast<int>(blockDim));
            MNN_CHECK_CL_SUCCESS(ret, "setArg gemv_conv1x1_buf Kernel Select");
            std::pair<std::vector<uint32_t>, int> retTune;
            retTune = localWS2DDefault(globalWorkSize[knl_idx], maxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), kernelName[knl_idx] + info, kernel[knl_idx]);
	    if(min_cost.first > retTune.second) {
                min_cost.first = retTune.second;
                min_cost.second = knl_idx;
                mLocalWorkSize = {retTune.first[0], retTune.first[1]};
            }
        }
        int min_index  = min_cost.second;
        mGlobalWorkSize = {globalWorkSize[min_index][0], globalWorkSize[min_index][1]};
        
        
        unit.kernel        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("gemm_quant_batch_buf", kernelName[min_index], buildOption);
        //MNN_PRINT("Kernel is %d.\n", min_index);
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[0]);
        ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[1]);
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mConvGemmInpTensor.get()));
        if(mResource->mUseImage){
            ret |= unit.kernel->get().setArg(idx++, *mResource->mKernelImage.get());
        }else{
            ret |= unit.kernel->get().setArg(idx++, *mResource->mKernelBuffer.get());
        }
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mResource->dequantScaleOffset.get()));
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mResource->mBias.get()));
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mConvGemmOutTensor.get()));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(outputChannelBlocks));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(inputChannelBlocks));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(blockNum));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(blockDim));
        MNN_CHECK_CL_SUCCESS(ret, "setArg gemv_conv1x1_buf");
        mOpenCLBackend->recordKernel2d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
        unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1]};
        unit.localWorkSize = {mLocalWorkSize[0], mLocalWorkSize[1]};
    }
    // reshape n/4*hw*c*4 -> n*c/4*4*hw
    {
        auto &unit = mUnits[2];
        mGlobalWorkSize = {static_cast<uint32_t>(UP_DIV(mResource->mOutputChannel, 4)), static_cast<uint32_t>(UP_DIV(batch, 4)), static_cast<uint32_t>(width_height)};
        unit.kernel        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("gemm_quant_batch_buf", "reshape_nhwc4_nchw4", buildOption);
        uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(unit.kernel));
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[0]);
        ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[1]);
        ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[2]);
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mConvGemmOutTensor.get()));
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(output));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(width_height));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(batch));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(outputChannelBlocks));
        MNN_CHECK_CL_SUCCESS(ret, "setArg reshape_cn4_nc4");
        mLocalWorkSize = localWS3DDefault(mGlobalWorkSize, maxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), "reshape_nhwc4_nchw4", unit.kernel).first;
        mOpenCLBackend->recordKernel3d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
        unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
        unit.localWorkSize = {mLocalWorkSize[0], mLocalWorkSize[1], mLocalWorkSize[2]};
    }
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
    if (mResource->mNumQuantBit == 8) {
        // int8 case
        mResource->mBuildOptions.emplace("-DUSE_LOW_BIT_WEIGHT_INT8");
    } else if (mResource->mNumQuantBit == 4){
        // int4 case
        mResource->mBuildOptions.emplace("-DUSE_LOW_BIT_WEIGHT_INT4");
    } else {/* More types to be supported. */}
#ifdef LOG_VERBOSE
    MNN_PRINT("end ConvBufLowMemoryExecution init !\n");
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
    MNN_PRINT("Start ConvBufLowMemoryExecution onResize !\n");
#endif
    mUnits.resize(1);
    auto input  = inputs[0];
    auto output = outputs[0];
    auto padding = ConvolutionCommon::convolutionPad(input, output, mResource->mConv2dCommonParams);
    mPaddings[0] = padding.second;//padY
    mPaddings[1] = padding.first;//padX
    // onclone default use conv1x1Opt, need reset
    std::vector<int> outputShape = tensorShapeFormat(output);
    const int batch = outputShape.at(0);
    bool isMali = mOpenCLBackend->getOpenCLRuntime()->getGpuType() == MALI;
    if (mResource->mConv1x1Opt) {
        if(batch > 1 && isMali){
            tuneGemvBatchLowMemory(input, output);
        }else{
            tuneGemmLowMemory(input, output);
        }
    } else {
        tuneGeneralCaseLowMemory(input, output);
    }
#ifdef LOG_VERBOSE
    MNN_PRINT("end ConvBufLowMemoryExecution onResize !\n");
#endif
    return NO_ERROR;
}
} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
#endif /* MNN_LOW_MEMORY */
