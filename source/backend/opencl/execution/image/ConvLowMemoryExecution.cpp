//  ConvLowMemoryExecution.cpp
//
//  Created by MNN on 2023/12/1.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef MNN_LOW_MEMORY
#ifndef MNN_OPENCL_BUFFER_CLOSED
#include "ConvLowMemoryExecution.hpp"
// #define LOG_VERBOSE
namespace MNN {
namespace OpenCL {

// set mDequantScale mDequantOffset mNumQuantBit mFilterDataPtr from mConv2dParams
void ConvLowMemoryExecution::getInfoFromOpLowMemory(std::shared_ptr<ConvolutionCommon::Int8Common> & quanCommon) {
    quanCommon = ConvolutionCommon::load(mResource->mConv2dParams, this->backend(), false, true);
    if (mResource->mConv2dParams->quanParameter() != nullptr) {
        mLowMemoryFlag = true;
    } else {
        MNN_ERROR("Conv buf low memory init error.\n");
        MNN_ASSERT(false);
    }
    
    mResource->mInputChannel = quanCommon->weight.size() / (mResource->mKernelWidth * mResource->mKernelHeight * mResource->mOutputChannel);
    // set mNumQuantBit
    if(quanCommon->canUseInt4){
        mNumQuantBit = 4;
    }else{
        mNumQuantBit = 8;
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
    int numAlphaPack = ROUND_UP(numAlpha, 16);
    int mapSize = mResource->mBlockSize * numAlphaPack * sizeof(int32_t) * 2;
    mResource->dequantScaleOffset.reset(new cl::Buffer(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, mapSize));
    // transfer data from src in cpu to dst in gpu
    cl_int resScaleOffset;
    void * dequantScaleOffsetBufferMap = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(*mResource->dequantScaleOffset.get(), true, CL_MAP_WRITE, 0, mapSize, nullptr, nullptr, &resScaleOffset);
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
    mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(*mResource->dequantScaleOffset.get(), dequantScaleOffsetBufferMap);
    // set mFilterDataPtr
    mFilterDataPtr = (void *)quanCommon->weight.get();
}
// set mKernelBuffer for the 1x1 kernels
void ConvLowMemoryExecution::set1x1WeightLowMemory(int packCout, int packCin, void * filterDataPtr, std::shared_ptr<ConvolutionCommon::Int8Common> & quanCommon) {
    cl_int res;
    std::shared_ptr<Tensor> filterBuffer(Tensor::createDevice<float>({ROUND_UP(mResource->mOutputChannel, 8)/*Cout pack set to max 8*/, ROUND_UP(mResource->mInputChannel, packCin), mResource->mKernelWidth, mResource->mKernelHeight}));
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
    mResource->mKernelBuffer.reset(new cl::Buffer(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size));
    auto kernelBufferPtr = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(*(mResource->mKernelBuffer.get()), true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &res);
    if(kernelBufferPtr != nullptr && res == CL_SUCCESS){
        ::memset(kernelBufferPtr, 0, buffer_size);

        
        for(int o = 0; o < mResource->mOutputChannel; o++){
            int i = 0;
            for(; i < mResource->mInputChannel; i++){
                int bufferIdx = (o/packCout) * packCin*packCout + (i/packCin)*packCin*ROUND_UP(mResource->mOutputChannel, packCout) + (i%packCin)*packCout + (o%packCout);//(Ci/packCin， Co/packCout, packCin， packCout)
                int filterIdx = o*mResource->mInputChannel + i;
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
        }
    } else {
        MNN_ERROR("set1x1WeightLowMemory: Map error ptrCL == nullptr \n");
        MNN_ASSERT(false);
    }
    mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(*(mResource->mKernelBuffer.get()), kernelBufferPtr);
}
// set mFilter for the general kernels
void ConvLowMemoryExecution::setGeneralWeightLowMemory(void* filterDataPtr, std::shared_ptr<ConvolutionCommon::Int8Common> & quanCommon) {
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
        if (mNumQuantBit == 8) {
            // ROUND_UP(IC, 4), UP_DIV(OC, 4) * mKernelWidth * mKernelHeight
            mResource->mFilter.reset(Tensor::createDevice<int8_t>({1, filterImageShape[1], 1, 4 * filterImageShape[0]}));
            mResource->mKernelBuffer.reset(new cl::Buffer(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size));
            mResource->mFilter->buffer().device = (uint64_t)(mResource->mKernelBuffer.get());
            MNN::OpenCL::BufferConvertor bufferConvertor{mOpenCLBackend->getOpenCLRuntime()};
            // filterBuffer shape: {OC, ROUND_UP(IC, 4), mKernelWidth, mKernelHeight}
            bufferConvertor.convertToNC4HW4Buffer(filterBuffer.get(), MNN::OpenCL::CONV2D_FILTER, mResource->mFilter.get(), false, true, mLowMemoryFlag, mNumQuantBit);
        } else if (mNumQuantBit == 4){
            // ROUND_UP(IC, 4), UP_DIV(OC, 4) * mKernelWidth * mKernelHeight
            // For int4 case, data stored in mFilter should be uint8_t
            // while "Tensor::createDevice<uint8_t>" occupies more memory than "Tensor::createDevice<int8_t>".
            // Therefore, we use "Tensor::createDevice<int8_t>" currently, leaving "Tensor::createDevice<uint8_t>" to be supported.
            mResource->mFilter.reset(Tensor::createDevice<int8_t>({1, filterImageShape[1], 1, 2 * filterImageShape[0]}));
            mResource->mKernelBuffer.reset(new cl::Buffer(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size/2));
            mResource->mFilter->buffer().device = (uint64_t)(mResource->mKernelBuffer.get());
            MNN::OpenCL::BufferConvertor bufferConvertor{mOpenCLBackend->getOpenCLRuntime()};
            // filterBuffer shape: {OC, ROUND_UP(IC, 4), mKernelWidth, mKernelHeight}
            bufferConvertor.convertToNC4HW4Buffer(filterBuffer.get(), MNN::OpenCL::CONV2D_FILTER, mResource->mFilter.get(), false, true, mLowMemoryFlag, mNumQuantBit);
        } else {/* More types to be supported. */}
    } else {
        MNN_ERROR("GetConvParams Error: filterDataPtr == nullptr. \n");
        MNN_ASSERT(false);
    }
}
// select the fastest kernel for the 1x1 cases by tuning
void ConvLowMemoryExecution::tune1x1CaseLowMemory(Tensor * input, Tensor * output) {
    auto &unit = mUnits[0];
    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);
    auto runTime     = ((OpenCLBackend *)backend())->getOpenCLRuntime();
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
    cl_int ret = CL_SUCCESS;
    for(int knl_idx = 0; knl_idx < actual_kernel; knl_idx++) {
        std::set<std::string> buildOption = mResource->mBuildOptions;
        if(inputChannels % 4 != 0){
            buildOption.emplace("-DINPUT_CHANNEL_LEAVE");
        }
        kernel[knl_idx]        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("conv_2d", kernelName[knl_idx], buildOption);
        uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(kernel[knl_idx]));
        
        globalWorkSize[knl_idx] = {static_cast<uint32_t>(UP_DIV(outputShape.at(3), itemC[knl_idx]) * UP_DIV(outputShape.at(2), itemW[knl_idx])), static_cast<uint32_t>(outputShape.at(0) * UP_DIV(outputShape.at(1), itemH[knl_idx]))};
        uint32_t idx            = 0;
        ret |= kernel[knl_idx]->get().setArg(idx++, globalWorkSize[knl_idx][0]);
        ret |= kernel[knl_idx]->get().setArg(idx++, globalWorkSize[knl_idx][1]);
        ret |= kernel[knl_idx]->get().setArg(idx++, openCLImage(input));
        ret |= kernel[knl_idx]->get().setArg(idx++, *mResource->mKernelBuffer.get());
        ret |= kernel[knl_idx]->get().setArg(idx++, *mResource->dequantScaleOffset.get());
        ret |= kernel[knl_idx]->get().setArg(idx++, openCLImage(mResource->mBias.get()));
        ret |= kernel[knl_idx]->get().setArg(idx++, openCLImage(output));
        ret |= kernel[knl_idx]->get().setArg(idx++, sizeof(inputImageShape), inputImageShape);
        ret |= kernel[knl_idx]->get().setArg(idx++, static_cast<int>(inputChannelBlocks));
        ret |= kernel[knl_idx]->get().setArg(idx++, sizeof(outputImageShape), outputImageShape);
        ret |= kernel[knl_idx]->get().setArg(idx++, sizeof(stideShape), stideShape);
        ret |= kernel[knl_idx]->get().setArg(idx++, UP_DIV(width, 4));
        ret |= kernel[knl_idx]->get().setArg(idx++, UP_DIV(outputShape.at(3), 4));
        ret |= kernel[knl_idx]->get().setArg(idx++, blockDim);
        ret |= kernel[knl_idx]->get().setArg(idx++, inputChannels);
        
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
    mGlobalWorkSize = {globalWorkSize[min_index][0], globalWorkSize[min_index][1]};
    std::set<std::string> buildOption = mResource->mBuildOptions;
    if(inputChannels % 4 != 0){
        buildOption.emplace("-DINPUT_CHANNEL_LEAVE");
    }
    unit.kernel        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("conv_2d", kernelName[min_index], buildOption);
    uint32_t idx = 0;
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[0]);
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[1]);
    ret |= unit.kernel->get().setArg(idx++, openCLImage(input));
    ret |= unit.kernel->get().setArg(idx++, *mResource->mKernelBuffer.get());
    ret |= unit.kernel->get().setArg(idx++, *mResource->dequantScaleOffset.get());
    ret |= unit.kernel->get().setArg(idx++, openCLImage(mResource->mBias.get()));
    ret |= unit.kernel->get().setArg(idx++, openCLImage(output));
    ret |= unit.kernel->get().setArg(idx++, sizeof(inputImageShape), inputImageShape);
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(inputChannelBlocks));
    ret |= unit.kernel->get().setArg(idx++, sizeof(outputImageShape), outputImageShape);
    ret |= unit.kernel->get().setArg(idx++, sizeof(stideShape), stideShape);
    ret |= unit.kernel->get().setArg(idx++, UP_DIV(width, 4));
    ret |= unit.kernel->get().setArg(idx++, UP_DIV(outputShape.at(3), 4));
    ret |= unit.kernel->get().setArg(idx++, blockDim);
    ret |= unit.kernel->get().setArg(idx++, inputChannels);
    MNN_CHECK_CL_SUCCESS(ret, "setArg Conv1x1LowMemory");
    mOpenCLBackend->recordKernel2d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
    
    unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1]};
    unit.localWorkSize = {mLocalWorkSize[0], mLocalWorkSize[1]};
    return;
}
// select the fastest kernel for the general cases by tuning
void ConvLowMemoryExecution::tuneGeneralCaseLowMemory(Tensor * input, Tensor * output) {
    auto &unit = mUnits[0];
    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);
    auto runTime     = ((OpenCLBackend *)backend())->getOpenCLRuntime();
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
    // MNN_PRINT("Checking kernel %d.\n", knlCheck);
    for (int knl_idx = 0; knl_idx < actual_kernel; knl_idx++) {
        std::set<std::string> buildOption = mResource->mBuildOptions;
        if(inputChannels % 4 != 0){
            buildOption.emplace("-DINPUT_CHANNEL_LEAVE");
        }
        kernel[knl_idx]        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("conv_2d", kernelName[knl_idx], buildOption);
        uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(kernel[knl_idx]));

        globalWorkSize[knl_idx] = {static_cast<uint32_t>(UP_DIV(outputShape.at(3), itemC[knl_idx]) * UP_DIV(outputShape.at(2), itemW[knl_idx])), static_cast<uint32_t>(outputShape.at(0) * UP_DIV(outputShape.at(1), itemH[knl_idx]))};
        uint32_t idx            = 0;
        cl_int ret = CL_SUCCESS;
        ret |= kernel[knl_idx]->get().setArg(idx++, globalWorkSize[knl_idx][0]);
        ret |= kernel[knl_idx]->get().setArg(idx++, globalWorkSize[knl_idx][1]);
        ret |= kernel[knl_idx]->get().setArg(idx++, openCLImage(input));
        ret |= kernel[knl_idx]->get().setArg(idx++, openCLBuffer(mResource->mFilter.get()));
        ret |= kernel[knl_idx]->get().setArg(idx++, *mResource->dequantScaleOffset.get());
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
        ret |= kernel[knl_idx]->get().setArg(idx++, blockDim);
        ret |= kernel[knl_idx]->get().setArg(idx++, inputChannels);
        MNN_CHECK_CL_SUCCESS(ret, "setArg ConvLowMemory Kernel Select");
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
    if(inputChannels % 4 != 0){
        buildOption.emplace("-DINPUT_CHANNEL_LEAVE");
    }
    unit.kernel        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("conv_2d", kernelName[min_index], buildOption);

    uint32_t idx            = 0;
    cl_int ret = CL_SUCCESS;
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[0]);
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[1]);
    ret |= unit.kernel->get().setArg(idx++, openCLImage(input));
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mResource->mFilter.get()));
    ret |= unit.kernel->get().setArg(idx++, *mResource->dequantScaleOffset.get());
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
    ret |= unit.kernel->get().setArg(idx++, blockDim);
    ret |= unit.kernel->get().setArg(idx++, inputChannels);
    MNN_CHECK_CL_SUCCESS(ret, "setArg ConvLowMemory");
    mOpenCLBackend->recordKernel2d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);

    unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1]};
    unit.localWorkSize = {mLocalWorkSize[0], mLocalWorkSize[1]};
    return;
}
void ConvLowMemoryExecution::tuneGemmLowMemory(Tensor * input, Tensor * output) {
    auto &unit = mUnits[0];
    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);
    auto runTime     = ((OpenCLBackend *)backend())->getOpenCLRuntime();
    const int outChannel = outputShape.at(3);
    const int inputChannels = inputShape.at(3);
    const int batch = outputShape.at(0);
    const int inputChannelBlocks = UP_DIV(inputChannels, 4);
    const int outputChannelBlocks = UP_DIV(outChannel, 4);
    const int blockDim = mResource->mInputChannel / mResource->mBlockSize;
    std::string kernelname = "gemm_conv";
    int global_x = outputChannelBlocks;
    int global_y = batch;
    if(batch > 1)
    {
        kernelname = "gemm_conv_b2";
        global_y = UP_DIV(batch, 2);
    }
    std::string info = std::to_string(inputChannels) + "_" + std::to_string(outChannel);
    std::set<std::string> buildOption = mResource->mBuildOptions;
    if(inputChannels % 4 != 0){
        buildOption.emplace("-DINPUT_CHANNEL_LEAVE");
    }
    unit.kernel        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("gemm", kernelname, buildOption);
    uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(unit.kernel));
    mGlobalWorkSize = {static_cast<uint32_t>(global_x), static_cast<uint32_t>(global_y)};
    // MNN_PRINT("Kernel is %d.\n", min_index);
    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[0]);
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[1]);
    ret |= unit.kernel->get().setArg(idx++, openCLImage(input));
    ret |= unit.kernel->get().setArg(idx++, *mResource->mKernelBuffer.get());
    ret |= unit.kernel->get().setArg(idx++, *mResource->dequantScaleOffset.get());
    ret |= unit.kernel->get().setArg(idx++, openCLImage(mResource->mBias.get()));
    ret |= unit.kernel->get().setArg(idx++, openCLImage(output));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(outputChannelBlocks));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(inputChannelBlocks));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(batch));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(blockDim));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(inputChannels));
    MNN_CHECK_CL_SUCCESS(ret, "setArg gemm_conv");
    
    mLocalWorkSize = localWS2DDefault(mGlobalWorkSize, maxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), kernelname + info, unit.kernel).first;
    mOpenCLBackend->recordKernel2d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);

    unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1]};
    unit.localWorkSize = {mLocalWorkSize[0], mLocalWorkSize[1]};
    return;
}
ConvLowMemoryExecution::ConvLowMemoryExecution(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, const MNN::Op *op, Backend *backend)
    : ConvCommonExecution(op->main_as_Convolution2D(), backend), CommonExecution(backend, op) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start ConvLowMemoryExecution init !\n");
#endif
    auto &unit = mUnits[0];
    mOpenCLBackend                 = static_cast<OpenCLBackend *>(backend);
    const auto *conv2dParams       = op->main_as_Convolution2D();
    const auto *conv2dCommonParams = conv2dParams->common();
    mResource->mConv2dParams       = conv2dParams;
    mResource->mConv2dCommonParams = conv2dCommonParams;
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
    if (mResource->mKernelHeight == mResource->mKernelWidth && mResource->mKernelHeight == 1 && mResource->mStrides[0] == 1 && mResource->mStrides[1] == 1 && mPaddings[0] == 0 && mPaddings[1] == 0) {
        // set mKernelBuffer for 1x1 case
        // At first, set packCout equal to 4
        set1x1WeightLowMemory(4, 4, mFilterDataPtr, quanCommon);
        mResource->mConv1x1Opt = true;
    }else {
        // set mFilter for not 1x1 case
        setGeneralWeightLowMemory(mFilterDataPtr, quanCommon);
    }
    // Create Kernel
    mResource->mBuildOptions.emplace("-DBIAS");
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

ConvLowMemoryExecution::ConvLowMemoryExecution(std::shared_ptr<ConvResource> resource, const MNN::Op* op, Backend *backend)
    : ConvCommonExecution(backend), CommonExecution(backend, op) {
    mResource = resource;
    const auto *conv2dParams       = op->main_as_Convolution2D();
    const auto *conv2dCommonParams = conv2dParams->common();
    mResource->mConv2dParams       = conv2dParams;
    mResource->mConv2dCommonParams  = conv2dCommonParams;
}

ConvLowMemoryExecution::~ConvLowMemoryExecution() {
    // Do nothing
}

bool ConvLowMemoryExecution::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    *dst = new ConvLowMemoryExecution(mResource, op, bn);
    return true;
}

ErrorCode ConvLowMemoryExecution::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start ConvExecution onResize !\n");
#endif
    mUnits.resize(1);
    auto input  = inputs[0];
    auto output = outputs[0];
    auto padding = ConvolutionCommon::convolutionPad(input, output, mResource->mConv2dCommonParams);
    mPaddings[0] = padding.second;//padY
    mPaddings[1] = padding.first;//padX
    mResource->gemmOpt = (mResource->mConv1x1Opt && inputs[0]->width() == 1 && inputs[0]->height() == 1);
    if (mResource->gemmOpt) {
        tuneGemmLowMemory(input, output);
    } else if(mResource->mConv1x1Opt){
        tune1x1CaseLowMemory(input, output);
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
