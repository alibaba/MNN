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
    quanCommon = ConvolutionCommon::load(mOp, this->backend(), false, true);
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
        mResource->mInputChannel = (quanCommon->weight.size() * 2) / (mResource->mKernelWidth * mResource->mKernelHeight * mResource->mOutputChannel);
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
        buildOptions.emplace("-DCHANNEL_LEAVE");
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
    std::shared_ptr<Tensor> filterBuffer(Tensor::createDevice<float>({ROUND_UP(mResource->mOutputChannel, packCout), ROUND_UP(mResource->mInputChannel, packCin), 1, 1}));
    size_t buffer_size = filterBuffer->usize() / sizeof(float);
    size_t cpy_size = mResource->mOutputChannel * mResource->mInputChannel;
    // shared part for all cases
    if (mResource->mNumQuantBit == 4){
        // int4 case
        buffer_size /= 2;
        cpy_size = UP_DIV(cpy_size, 2);
    } else {/* More types to be supported. */}
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
        std::shared_ptr<Tensor> filterBuffer(Tensor::createDevice<float>({ROUND_UP(mResource->mOutputChannel, 4), mResource->mInputChannel, mResource->mKernelWidth, mResource->mKernelHeight}));
        size_t buffer_size = filterBuffer->usize() / sizeof(float);
        size_t cpy_size = mResource->mOutputChannel * mResource->mInputChannel * mResource->mKernelWidth * mResource->mKernelHeight;
        if (mResource->mNumQuantBit == 4){
            buffer_size /= 2;
            cpy_size = UP_DIV(cpy_size, 2);
        }
        cl::Buffer filterBufferCL(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size);
        filterBuffer->buffer().device = (uint64_t)(&filterBufferCL);
        float *dequantAlpha = quanCommon->alpha.get();
        // map and pack data from filterDataPtr
        cl_int res;
        auto ptrCL = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(filterBufferCL, true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &res);
        if(ptrCL != nullptr && res == CL_SUCCESS) {
                ::memcpy(ptrCL, filterDataPtr, cpy_size);
        } else {
            MNN_ERROR("setGeneralWeightLowMemory: Map error ptrCL == nullptr \n");
        }
        mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(filterBufferCL, ptrCL);
        // convert to NC4HW4
        if (mResource->mNumQuantBit == 8) {
            // ROUND_UP(IC, 4), UP_DIV(OC, 4) * mKernelWidth * mKernelHeight
            mResource->mFilter.reset(Tensor::createDevice<int8_t>({1, UP_DIV(mResource->mOutputChannel, 4) * mResource->mKernelWidth * mResource->mKernelHeight, 1, 4 * ROUND_UP(mResource->mInputChannel, 4)}));
            mOpenCLBackend->onAcquireBuffer(mResource->mFilter.get(), Backend::STATIC);
            MNN::OpenCL::BufferConvertor bufferConvertor{mOpenCLBackend->getOpenCLRuntime()};
            // filterBuffer shape: {OC, ROUND_UP(IC, 4), mKernelWidth, mKernelHeight}
            bufferConvertor.convertToNC4HW4Buffer(filterBuffer.get(), MNN::OpenCL::CONV2D_FILTER, mResource->mFilter.get(), false, true, mLowMemoryFlag, mResource->mNumQuantBit);
        } else if (mResource->mNumQuantBit == 4){
            // ROUND_UP(IC, 4), UP_DIV(OC, 4) * mKernelWidth * mKernelHeight
            // For int4 case, data stored in mFilter should be uint8_t,
            // while "Tensor::createDevice<uint8_t>" occupies more memory than "Tensor::createDevice<int8_t>".
            // Therefore, we use "Tensor::createDevice<int8_t>" currently, leaving "Tensor::createDevice<uint8_t>" to be supported.
            mResource->mFilter.reset(Tensor::createDevice<int8_t>({1, UP_DIV(mResource->mOutputChannel, 4) * mResource->mKernelWidth * mResource->mKernelHeight, 1, 2 * ROUND_UP(mResource->mInputChannel, 4)}));
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
    mUnits.resize(1);
    auto &unit = mUnits[0];
    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);
    const int batch              = outputShape.at(0);
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
        ret |= kernel[knl_idx]->get().setArg(idx++, batch);
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
    ret |= unit.kernel->get().setArg(idx++, batch);
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

// weight inverse quantization, use xgemm opt
void ConvBufLowMemoryExecution::useFPWeightGemmLowMemory(Tensor * input, Tensor * output) {
    mUnits.resize(3);
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);
    int channelPack = 16;
    if(mResource->mUseImage && mResource->mNumQuantBit == 4){
        channelPack = 32;
    }
    int area = inputShape.at(1) * inputShape.at(2);
    int M = outputShape.at(0) * area;
    int N = mResource->mOutputChannel;
    int K = mResource->mInputChannel;
    int mAlignK = 4;
    int mAlignN = 16;
    int mAlignM = 64;
    
    // set M Align and N Align
    if(mResource->mOutputChannel > 1024) {
        mAlignN = 128;
    } else if(mResource->mOutputChannel > 512) {
        mAlignN = 64;
    } else if(mResource->mOutputChannel > 96) {
        mAlignN = 32;
    }
    float ratio = 1.0 * M / 1024.0 * N / 1024.0 * K / 1024.0;
    if(M > 1024 && ratio >= 1.0) {
        mAlignM = 128;
    } else if(M > 512 && ratio >= 0.1) {
        mAlignM = 64;
    } else if(M > 96){
        mAlignM = 32;
    } else {
        mAlignM = 16;
    }
    int alignM = ROUND_UP(M, mAlignM);
    int alignN = ROUND_UP(N, mAlignN);
    int alignK = ROUND_UP(K, mAlignK);
    int blockDim = mResource->mInputChannel / mResource->mBlockSize;
    
    // alloc temp bufer
    mConvGemmWeightTensor.reset(Tensor::createDevice<float>({ROUND_UP(mResource->mOutputChannel, mAlignN) * ROUND_UP(mResource->mInputChannel, std::max(mAlignK, channelPack))}));
    mConvGemmInpTensor.reset(Tensor::createDevice<float>({alignK * alignM}));
    mConvGemmOutTensor.reset(Tensor::createDevice<float>({alignN * alignM}));
    mOpenCLBackend->onAcquireBuffer(mConvGemmWeightTensor.get(), Backend::DYNAMIC);
    mOpenCLBackend->onAcquireBuffer(mConvGemmOutTensor.get(), Backend::DYNAMIC);
    mOpenCLBackend->onAcquireBuffer(mConvGemmInpTensor.get(), Backend::DYNAMIC);
    mOpenCLBackend->onReleaseBuffer(mConvGemmWeightTensor.get(), Backend::DYNAMIC);
    mOpenCLBackend->onReleaseBuffer(mConvGemmInpTensor.get(), Backend::DYNAMIC);
    mOpenCLBackend->onReleaseBuffer(mConvGemmOutTensor.get(), Backend::DYNAMIC);
    
    //weight inverse quantization and rearrange
    {
        auto &unit = mUnits[0];
        int outputChannelAlign = ROUND_UP(mResource->mOutputChannel, alignN);
        int outputChannel4Align = ROUND_UP(mResource->mOutputChannel, 4);
        std::set<std::string> buildOption = mResource->mBuildOptions;
        if(mResource->mUseImage){
            buildOption.emplace("-DUSE_IMAGE");
        }
        mGlobalWorkSize = {static_cast<uint32_t>(UP_DIV(mResource->mInputChannel, channelPack)), static_cast<uint32_t>(UP_DIV(mResource->mOutputChannel, 4))};
        unit.kernel = runtime->buildKernel("gemm_conv1x1_buf", "inverse_quant_weight", buildOption);
        uint32_t maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[0]);
        ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[1]);
        if(mResource->mUseImage){
            ret |= unit.kernel->get().setArg(idx++, *mResource->mKernelImage.get());
        }else{
            ret |= unit.kernel->get().setArg(idx++, *mResource->mKernelBuffer.get());
        }
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mResource->dequantScaleOffset.get()));
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mConvGemmWeightTensor.get()));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(outputChannelAlign));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(outputChannel4Align));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(blockDim));
        MNN_CHECK_CL_SUCCESS(ret, "setArg inverse_quant_weight");
        
        mLocalWorkSize = localWS2DDefault(mGlobalWorkSize, maxWorkGroupSize, runtime, "inverse_quant_weight", unit.kernel).first;
        mOpenCLBackend->recordKernel2d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
        unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1]};
        unit.localWorkSize = {mLocalWorkSize[0], mLocalWorkSize[1]};
    }
    
    // rearange input
    {
        auto &unit = mUnits[1];
        std::set<std::string> buildOptions = mResource->mBuildOptions;
        
        int m_pack = 4;
        mGlobalWorkSize = {static_cast<uint32_t>(alignM/m_pack), static_cast<uint32_t>(alignK/4)};
        unit.kernel = mOpenCLBackend->getOpenCLRuntime()->buildKernel("gemm_buf", "transpose_pad", buildOptions);
        uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(unit.kernel));

        int offset = 0;
        int idx            = 0;
        cl_int ret = CL_SUCCESS;
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(mGlobalWorkSize[0]));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(mGlobalWorkSize[1]));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(alignM));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(alignK));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(M));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(K));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(area));
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(input));
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mConvGemmInpTensor.get()));
        MNN_CHECK_CL_SUCCESS(ret, "setArg transpose_pad");
        mLocalWorkSize = localWS2DDefault(mGlobalWorkSize, maxWorkGroupSize, runtime, "transpose_pad", unit.kernel).first;

        mOpenCLBackend->recordKernel2d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
        unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1]};
        unit.localWorkSize = {mLocalWorkSize[0], mLocalWorkSize[1]};
    }
    
    // call gemm strassen
    {
        mStrassenComputor.reset(new StrassenMatrixComputor(backend(), 3));
        mStrassenComputor->onEncode(alignM, alignK, alignN, alignM, alignN, alignN, openCLBuffer(mConvGemmInpTensor.get()), openCLBuffer(mConvGemmWeightTensor.get()), openCLBuffer(mConvGemmOutTensor.get()), false, openCLBuffer(mResource->mBias.get()));
    }
        
    // call output transpose
    {
        auto &unit = mUnits[2];
        std::set<std::string> buildOptions = mResource->mBuildOptions;
        int pack_m = 1;
        if(M % 8 == 0) {
            pack_m = 8;
        } else if(M % 4 == 0) {
            pack_m = 4;
        }
        buildOptions.emplace("-DM_VEC=" + std::to_string(pack_m));
        unit.kernel = runtime->buildKernel("gemm_buf", "transpose_bias", buildOptions);
        uint32_t maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));

        mGlobalWorkSize = {static_cast<uint32_t>(UP_DIV(M, pack_m)), static_cast<uint32_t>(UP_DIV(N, 4))};

        int offset = 0;
        int idx            = 0;
        cl_int ret = CL_SUCCESS;
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(mGlobalWorkSize[0]));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(mGlobalWorkSize[1]));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(alignM));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(alignN));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(M));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(N));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(area));
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mConvGemmOutTensor.get()));
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mResource->mBias.get()));
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(output));

        MNN_CHECK_CL_SUCCESS(ret, "setArg transpose_bias");
        mLocalWorkSize = localWS2DDefault(mGlobalWorkSize, maxWorkGroupSize, runtime, "transpose_bias", unit.kernel).first;
        mOpenCLBackend->recordKernel2d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
        unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1]};
        unit.localWorkSize = {mLocalWorkSize[0], mLocalWorkSize[1]};
    }
    
    return;
}
void ConvBufLowMemoryExecution::tuneGemvLowMemory(Tensor * input, Tensor * output) {
    mUnits.resize(1);
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

    int global_y = batch * height * width;
    const int total_kernel = 3;
    std::string kernelName[total_kernel] = {"gemv_conv_c1_buf", "gemv_conv_c2_buf", "gemv_conv_c4_buf"};
    int itemC[total_kernel] = {1, 2, 4};
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
    if(mResource->mUseImage){
        buildOption.emplace("-DUSE_IMAGE");
    }
    for (int knl_idx = 0; knl_idx < actual_kernel; knl_idx++) {
        auto option = buildOption;
        option.emplace("-DTILE_N=" + std::to_string(itemC[knl_idx]));
        kernel[knl_idx]        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("gemv_conv1x1_buf", kernelName[knl_idx], option);
        uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(kernel[knl_idx]));
        globalWorkSize[knl_idx] = {static_cast<uint32_t>(UP_DIV(outChannel, itemC[knl_idx])), static_cast<uint32_t>(global_y)};
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
        ret |= kernel[knl_idx]->get().setArg(idx++, static_cast<int>(global_y));
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
    
    buildOption.emplace("-DTILE_N=" + std::to_string(itemC[min_index]));
    unit.kernel        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("gemv_conv1x1_buf", kernelName[min_index], buildOption);
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
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(global_y));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(blockNum));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(blockDim));
    MNN_CHECK_CL_SUCCESS(ret, "setArg gemv_conv1x1_buf");
    mOpenCLBackend->recordKernel2d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
    unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1]};
    unit.localWorkSize = {mLocalWorkSize[0], mLocalWorkSize[1]};
    return;
}
unsigned int ConvBufLowMemoryExecution::tuneGemmLowMemory(Tensor * input, Tensor * output, std::string option, bool onlyGetTime) {
    mUnits.resize(3);
    unsigned int total_time = 0;
    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);
    int channelPack = 16;
    if(mResource->mUseImage && mResource->mNumQuantBit == 4){
        channelPack = 32;
    }
    const int outChannel = outputShape.at(3);
    const int inputChannels = inputShape.at(3);
    const int batch = outputShape.at(0);
    const int width_height = outputShape.at(1) * outputShape.at(2);
    const int inputChannelAlign = ROUND_UP(inputChannels, channelPack);
    const int outputChannelAlign = ROUND_UP(outChannel, 4);
    const int blockNum = mResource->mBlockSize;
    const int blockDim = mResource->mInputChannel / mResource->mBlockSize;
    
    int global_y = batch * width_height;
    const int total_kernel = 3;
    std::string kernelName[total_kernel] = {"gemm_b4_c1_buf", "gemm_b4_c2_buf", "gemm_b4_c4_buf"};
    int itemC[total_kernel] = {1, 2, 4};
    int actual_kernel = total_kernel;
    std::shared_ptr<KernelWrap> kernel[total_kernel];
    std::vector<uint32_t> globalWorkSize[total_kernel];
    std::vector<uint32_t> localWorkSize[total_kernel];
    std::pair<unsigned int, int> min_cost(INT_MAX, 0);//(min_time, min_index)
    std::set<std::string> buildOption = mResource->mBuildOptions;
    if(blockDim % 16 != 0){
        buildOption.emplace("-DINPUT_CHANNEL_LEAVE");
    } else if (mResource->mUseImage && mResource->mNumQuantBit == 4 && blockDim % 32 != 0) {
        // Image weight-int4 use load32
        buildOption.emplace("-DINPUT_CHANNEL_LEAVE");
    }
    buildOption.emplace(option);
    if(mResource->mUseImage){
        buildOption.emplace("-DUSE_IMAGE");
    }
    std::string info = std::to_string(inputChannels) + "_" + std::to_string(outChannel) + option;
    // mResource->mInputChannel ROUND_UP to blockDim, avoid gemm overstep
    mConvGemmInpTensor.reset(Tensor::createDevice<float>({ROUND_UP(batch, 4) * inputChannelAlign * width_height}));
    mConvGemmOutTensor.reset(Tensor::createDevice<float>({ROUND_UP(batch, 4) * ROUND_UP(mResource->mOutputChannel, 4) * width_height}));
    mOpenCLBackend->onAcquireBuffer(mConvGemmInpTensor.get(), Backend::DYNAMIC);
    mOpenCLBackend->onAcquireBuffer(mConvGemmOutTensor.get(), Backend::DYNAMIC);
    mOpenCLBackend->onReleaseBuffer(mConvGemmOutTensor.get(), Backend::DYNAMIC);
    mOpenCLBackend->onReleaseBuffer(mConvGemmInpTensor.get(), Backend::DYNAMIC);

    // reshape n*c/4*4*hw -> n/4*hw*c*4
    {
        auto &unit = mUnits[0];
        mGlobalWorkSize = {static_cast<uint32_t>(UP_DIV(inputChannelAlign, 4)), static_cast<uint32_t>(UP_DIV(global_y, 4))};
        unit.kernel        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("gemm_conv1x1_buf", "reshape_nchw4_nhwc4", buildOption);
        uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(unit.kernel));

        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[0]);
        ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[1]);
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(input));
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mConvGemmInpTensor.get()));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(global_y));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(inputChannels));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(inputChannelAlign));
        MNN_CHECK_CL_SUCCESS(ret, "setArg reshape_nc4_cn4");
        std::pair<std::vector<uint32_t>, unsigned int> retTune = localWS2DDefault(mGlobalWorkSize, maxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), "reshape_nchw4_nhwc4", unit.kernel);
        total_time += retTune.second;
        mLocalWorkSize = retTune.first;
        if(false == onlyGetTime){
            mOpenCLBackend->recordKernel2d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
        }
        unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1]};
        unit.localWorkSize = {mLocalWorkSize[0], mLocalWorkSize[1]};
    }
    // gemm
    {
        auto &unit = mUnits[1];
        for (int knl_idx = 0; knl_idx < actual_kernel; knl_idx++) {
            kernel[knl_idx]        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("gemm_conv1x1_buf", kernelName[knl_idx], buildOption);
            uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(kernel[knl_idx]));
            
            globalWorkSize[knl_idx] = {static_cast<uint32_t>(UP_DIV(outChannel, itemC[knl_idx])), static_cast<uint32_t>(UP_DIV(global_y, 4))};
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
            ret |= kernel[knl_idx]->get().setArg(idx++, static_cast<int>(UP_DIV(global_y, 4) * 4));
            ret |= kernel[knl_idx]->get().setArg(idx++, static_cast<int>(outputChannelAlign));
            ret |= kernel[knl_idx]->get().setArg(idx++, static_cast<int>(inputChannelAlign));
            ret |= kernel[knl_idx]->get().setArg(idx++, static_cast<int>(blockNum));
            ret |= kernel[knl_idx]->get().setArg(idx++, static_cast<int>(blockDim));
            MNN_CHECK_CL_SUCCESS(ret, "setArg gemv_conv1x1_buf Kernel Select");
            std::pair<std::vector<uint32_t>, unsigned int> retTune;
            retTune = localWS2DDefault(globalWorkSize[knl_idx], maxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), kernelName[knl_idx] + info, kernel[knl_idx]);
            if(min_cost.first > retTune.second) {
                min_cost.first = retTune.second;
                min_cost.second = knl_idx;
                mLocalWorkSize = {retTune.first[0], retTune.first[1]};
            }
        }
        total_time += min_cost.first;
        int min_index  = min_cost.second;
        mGlobalWorkSize = {globalWorkSize[min_index][0], globalWorkSize[min_index][1]};


        unit.kernel        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("gemm_conv1x1_buf", kernelName[min_index], buildOption);
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
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(UP_DIV(global_y, 4) * 4));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(outputChannelAlign));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(inputChannelAlign));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(blockNum));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(blockDim));
        MNN_CHECK_CL_SUCCESS(ret, "setArg gemm_conv1x1_buf");
        if(false == onlyGetTime){
            mOpenCLBackend->recordKernel2d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
        }
        unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1]};
        unit.localWorkSize = {mLocalWorkSize[0], mLocalWorkSize[1]};
    }
    // reshape n/4*hw*c*4 -> n*c/4*4*hw
    {
        auto &unit = mUnits[2];
        mGlobalWorkSize = {static_cast<uint32_t>(UP_DIV(mResource->mOutputChannel, 4)), static_cast<uint32_t>(UP_DIV(global_y, 4))};
        unit.kernel        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("gemm_conv1x1_buf", "reshape_nhwc4_nchw4", buildOption);
        uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(unit.kernel));
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[0]);
        ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[1]);
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mConvGemmOutTensor.get()));
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(output));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(global_y));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(outputChannelAlign));
        MNN_CHECK_CL_SUCCESS(ret, "setArg reshape_cn4_nc4");
        std::pair<std::vector<uint32_t>, unsigned int> retTune = localWS2DDefault(mGlobalWorkSize, maxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), "reshape_nhwc4_nchw4", unit.kernel);
        mLocalWorkSize = retTune.first;
        total_time += retTune.second;
        if(false == onlyGetTime){
            mOpenCLBackend->recordKernel2d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
        }
        unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1]};
        unit.localWorkSize = {mLocalWorkSize[0], mLocalWorkSize[1]};
    }
    return total_time;
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

ErrorCode ConvBufLowMemoryExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start ConvBufLowMemoryExecution onResize !\n");
#endif
    auto runTime = mOpenCLBackend->getOpenCLRuntime();
    mOpenCLBackend->startRecord(mRecording);
    mUnits.resize(1);
    auto input  = inputs[0];
    auto output = outputs[0];
    auto padding = ConvolutionCommon::convolutionPad(input, output, mResource->mConv2dCommonParams);
    mPaddings[0] = padding.second;//padY
    mPaddings[1] = padding.first;//padX
    // onclone default use conv1x1Opt, need reset
    std::vector<int> outputShape = tensorShapeFormat(output);
    const int batch = outputShape.at(0) * outputShape.at(1) * outputShape.at(2);
    mUseFPWeight = false;
    if (mResource->mConv1x1Opt) {
        if(batch == 1){
            tuneGemvLowMemory(input, output);
        } else {
            if(batch > 512){
                useFPWeightGemmLowMemory(input, output);
                mUseFPWeight = true;
            }
            else if(false == getPreParamInfo("ConvBufLowMemoryPreArrangeMode", &batchConvMode, runTime)){
                if(tuneGemmLowMemory(input, output, "-DFORMAT_CNHW", true) < tuneGemmLowMemory(input, output, "", true)){
                    batchConvMode = 1;
                } else{
                    batchConvMode = 2;
                }
                setPreParamInfo("ConvBufLowMemoryPreArrangeMode", batchConvMode, runTime);
            } else {
                std::string option = "";
                if(1 == batchConvMode){
                    option = "-DFORMAT_CNHW";
                }
                tuneGemmLowMemory(input, output, option);
            }
        }
    } else {
        tuneGeneralCaseLowMemory(input, output);
    }
    for (auto &unit : mUnits) {
        bool lws_null = true;
        for (size_t i = 0; i < unit.globalWorkSize.dimensions(); ++i) {
            unit.globalWorkSize.get()[i] = ROUND_UP(unit.globalWorkSize.get()[i], std::max((size_t)1, unit.localWorkSize.get()[i]));
            if(unit.localWorkSize.get()[i] != 0) {
                lws_null = false;
            }
        }
        if(lws_null){
            unit.localWorkSize = cl::NullRange;
        }
    }
    mOpenCLBackend->endRecord(mRecording);
#ifdef LOG_VERBOSE
    MNN_PRINT("end ConvBufLowMemoryExecution onResize !\n");
#endif
    return NO_ERROR;
}

ErrorCode ConvBufLowMemoryExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start ConvBufLowMemoryExecution onExecute !\n");
#endif
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
#ifdef ENABLE_OPENCL_TIME_PROFILER
    int idx = 0;
#else
    if(mOpenCLBackend->isUseRecordQueue()){
        mOpenCLBackend->addRecord(mRecording, mOpRecordUpdateInfo);
        return NO_ERROR;
    }
#endif
    auto res = CL_SUCCESS;
    if(mUseFPWeight){
        // arrange input and weight
        int i = 0;
        for (; i < 2; ++i){
            auto unit = mUnits[i];
            #ifdef ENABLE_OPENCL_TIME_PROFILER
            cl::Event event;
            res = runtime->commandQueue().enqueueNDRangeKernel(unit.kernel->get(),
                                                   cl::NullRange,
                                                   unit.globalWorkSize,
                                                   unit.localWorkSize,
                                                   nullptr,
                                                   &event);
            runtime->pushEvent({EnumNameOpType(mOpType) + std::to_string(idx++), event});
            #else
            res = runtime->commandQueue().enqueueNDRangeKernel(unit.kernel->get(),
                                                   cl::NullRange,
                                                   unit.globalWorkSize,
                                                   unit.localWorkSize);
            #endif
            MNN_CHECK_CL_SUCCESS(res, EnumNameOpType(mOp->type()));
        }
        // call gemm execute
        mStrassenComputor->onExecute();
        
        // rearrange output
        for (; i < mUnits.size(); ++i){
            auto unit = mUnits[i];
            #ifdef ENABLE_OPENCL_TIME_PROFILER
            cl::Event event;
            res = runtime->commandQueue().enqueueNDRangeKernel(unit.kernel->get(),
                                                   cl::NullRange,
                                                   unit.globalWorkSize,
                                                   unit.localWorkSize,
                                                   nullptr,
                                                   &event);
            runtime->pushEvent({EnumNameOpType(mOpType) + std::to_string(idx++), event});
            #else
            res = runtime->commandQueue().enqueueNDRangeKernel(unit.kernel->get(),
                                                   cl::NullRange,
                                                   unit.globalWorkSize,
                                                   unit.localWorkSize);
            #endif
            MNN_CHECK_CL_SUCCESS(res, EnumNameOpType(mOp->type()));
        }
    }else{
        for (auto &unit : mUnits) {
            #ifdef ENABLE_OPENCL_TIME_PROFILER
            cl::Event event;
            res = runtime->commandQueue().enqueueNDRangeKernel(unit.kernel->get(),
                                                               cl::NullRange,
                                                               unit.globalWorkSize,
                                                               unit.localWorkSize,
                                                               nullptr,
                                                               &event);
            runtime->pushEvent({EnumNameOpType(mOpType) + std::to_string(idx++), event});
            #else
            res = runtime->commandQueue().enqueueNDRangeKernel(unit.kernel->get(),
                                                               cl::NullRange,
                                                               unit.globalWorkSize,
                                                               unit.localWorkSize);
            #endif
            MNN_CHECK_CL_SUCCESS(res, EnumNameOpType(mOp->type()));
        }
    }
#ifdef LOG_VERBOSE
    MNN_PRINT("end ConvBufLowMemoryExecution onExecute !\n");
#endif
    return NO_ERROR;
}

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
#endif /* MNN_LOW_MEMORY */
