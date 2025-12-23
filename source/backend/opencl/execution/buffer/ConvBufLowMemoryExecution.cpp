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
#define PACK_COUT 8
#define PACK_CIN 4

// set mDequantScale mDequantOffset mNumQuantBit mFilterDataPtr from mConv2dParams
void ConvBufLowMemoryExecution::getInfoFromOpLowMemory(void *weight_ptr) {
    auto quanCommon = ConvolutionCommon::load(mOp, this->backend(), false, true, weight_ptr);
    // set mResource->mNumQuantBit
    if(quanCommon->canUseInt4){
        mResource->mNumQuantBit = 4;
    }else{
        mResource->mNumQuantBit = 8;
    }
    if (mOp->main_as_Convolution2D()->common()->inputCount() > 0) {
        mResource->mInputChannel = mOp->main_as_Convolution2D()->common()->inputCount();
    } else {
        mResource->mInputChannel = quanCommon->weight.size() / (mResource->mKernelWidth * mResource->mKernelHeight * mResource->mOutputChannel);
    }
    // src of alpha in CPU
    float * dequantAlpha = quanCommon->alpha.get();
    int totalCount = quanCommon->alphaSize;
    int soSize = 1;
    if (quanCommon->asymmetric) {
        soSize = 2;
        totalCount /= 2;
        mResource->mBuildOptions.emplace("-DASYMMETRIC");
    }
    int numAlpha = mResource->mOutputChannel;
    mResource->mBlockSize = totalCount / numAlpha;
    // set mDequantScale mDequantOffset
    int numAlphaPack = ROUND_UP(numAlpha, 4);
    int fpBytes = mOpenCLBackend->fpBytes();
    int buffer_size = mResource->mBlockSize * numAlphaPack * fpBytes * soSize + sizeof(float);
    
    auto staticMapAlloc = mOpenCLBackend->getStaticAllocatorMMap();
    if(mOpenCLBackend->getRuntime()->hint().useCachedMmap && staticMapAlloc != nullptr){
        mResource->mDequantScaleOffsetBuffer = staticMapAlloc.get()->allocBuffer(buffer_size);
    }else{
        mResource->mDequantScaleOffsetBuffer.reset(new cl::Buffer(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size));
    }
    // transfer data from src in cpu to dst in gpu
    cl_int resBias, resScaleOffset;
    float coef = 1.0;
    
    void * dequantScaleOffsetBufferMap = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(*mResource->mDequantScaleOffsetBuffer.get(), true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &resScaleOffset);
    if(mOpenCLBackend->getRuntime()->hint().useCachedMmap > 1){
        if(fpBytes == 2){
            float* coefMapPtr = (float*)(((half_float::half*)dequantScaleOffsetBufferMap) + (numAlphaPack * mResource->mBlockSize * soSize));
            coef = coefMapPtr[0];
        }else{
            coef = ((float *)dequantScaleOffsetBufferMap)[(numAlphaPack * mResource->mBlockSize * soSize)];
        }
    }else{
        if(fpBytes == 2) {
            float max_data = 0.0f;
            if (quanCommon->asymmetric){
                for (int i = 0; i < numAlpha; ++i) {
                    auto srcZ = dequantAlpha + i * mResource->mBlockSize * 2;
                    for(int j = 0; j < mResource->mBlockSize; ++j){
                        float s = fabsf(srcZ[2*j+0]);
                        float b = fabsf(srcZ[2*j+1]);
                        float temp = ALIMAX(s, b);
                        if(temp > max_data) {
                            max_data = temp;
                        }
                    }
                }
            }else{
                for (int i = 0; i < numAlpha; ++i) {
                    auto srcZ = dequantAlpha + i * mResource->mBlockSize;
                    for(int j = 0; j < mResource->mBlockSize; ++j){
                        float s = fabsf(srcZ[j]);
                        if(s > max_data) {
                            max_data = s;
                        }
                    }
                }
            }
            if(abs(max_data) >= 0.000001f){
                coef = 1000.0f / max_data;
            }
            if (dequantScaleOffsetBufferMap != nullptr && resScaleOffset == CL_SUCCESS) {
                if (quanCommon->asymmetric) {
                    for (int i = 0; i < numAlpha; ++i) {
                        auto srcZ = dequantAlpha + i * mResource->mBlockSize * 2;
                        for(int j = 0; j < mResource->mBlockSize; ++j){
                            float o = srcZ[2*j+0];
                            float s = srcZ[2*j+1];
                            ((half_float::half*)dequantScaleOffsetBufferMap)[(j * numAlphaPack + i) * 2] = (half_float::half)(s * coef);
                            ((half_float::half*)dequantScaleOffsetBufferMap)[(j * numAlphaPack + i) * 2 + 1] = (half_float::half)(o * coef);
                        }
                    }
                } else {
                    for (int i = 0; i < numAlpha; ++i) {
                        auto srcZ = dequantAlpha + i * mResource->mBlockSize;
                        for(int j = 0; j < mResource->mBlockSize; ++j){
                            ((half_float::half*)dequantScaleOffsetBufferMap)[(j * numAlphaPack + i)] = (half_float::half)(srcZ[j] * coef);
                        }
                    }
                }
                float* coefMapPtr = (float*)(((half_float::half*)dequantScaleOffsetBufferMap) + (numAlphaPack * mResource->mBlockSize * soSize));
                coefMapPtr[0] = coef;
            } else {
                MNN_ERROR("Map error dequantBufferMap == nullptr \n");
                MNN_ASSERT(false);
            }
        } else{
            if (dequantScaleOffsetBufferMap != nullptr && resScaleOffset == CL_SUCCESS) {
                if (quanCommon->asymmetric) {
                    for (int i = 0; i < numAlpha; ++i) {
                        auto srcZ = dequantAlpha + i * mResource->mBlockSize * 2;
                        for(int j = 0; j < mResource->mBlockSize; ++j){
                            float o = srcZ[2*j+0];
                            float s = srcZ[2*j+1];
                            ((float *)dequantScaleOffsetBufferMap)[(j * numAlphaPack + i) * 2] = s * coef;
                            ((float *)dequantScaleOffsetBufferMap)[(j * numAlphaPack + i) * 2 + 1] = o * coef;
                        }
                    }
                } else {
                    for (int i = 0; i < numAlpha; ++i) {
                        auto srcZ = dequantAlpha + i * mResource->mBlockSize;
                        for(int j = 0; j < mResource->mBlockSize; ++j){
                            ((float *)dequantScaleOffsetBufferMap)[(j * numAlphaPack + i)] = srcZ[j] * coef;
                        }
                    }
                }
                ((float *)dequantScaleOffsetBufferMap)[(numAlphaPack * mResource->mBlockSize * soSize)] = coef;
            } else {
                MNN_ERROR("Map error dequantBufferMap == nullptr \n");
                MNN_ASSERT(false);
            }
        }
    }
    mResource->mCoef = coef;
    mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(*mResource->mDequantScaleOffsetBuffer.get(), dequantScaleOffsetBufferMap);
    // set mFilterDataPtr
    mFilterDataPtr = (void *)quanCommon->weight.get();
}

bool ConvBufLowMemoryExecution::convertToQuantWeight1x1Buffer(cl::Buffer input) {
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

    mBufferToConv1x1Kernel = runtime->buildKernelWithCache("buffer_convert_quant", kernelName, buildOptions, mOpenCLBackend->getPrecision());
    auto kernel = mBufferToConv1x1Kernel->get();
    uint32_t gws[2] = {static_cast<uint32_t>(UP_DIV(mResource->mInputChannel, PACK_CIN)), static_cast<uint32_t>(UP_DIV(mResource->mOutputChannel, PACK_COUT))};

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
void ConvBufLowMemoryExecution::set1x1WeightLowMemory() {
    bool preAllocGpuMem = mResource->mInputChannel != 0 && mResource->mConv2dParams->quanParameter();
    if(preAllocGpuMem){
        mResource->mNumQuantBit = mResource->mConv2dParams->quanParameter()->aMaxOrBits();
        if(mResource->mNumQuantBit == 0){
            // support old model for external weight file with int4/int8 quant
            mResource->mNumQuantBit = ConvolutionCommon::getQuantBitFromExternalFile(mOp);
        }
    } else{
        getInfoFromOpLowMemory(nullptr);
    }
    cl_int res = CL_SUCCESS;
    std::shared_ptr<Tensor> filterBuffer(Tensor::createDevice<float>({ROUND_UP(mResource->mOutputChannel, PACK_COUT), ROUND_UP(mResource->mInputChannel, PACK_CIN), 1, 1}));
    size_t buffer_size = filterBuffer->usize() / sizeof(float);
    size_t cpy_size = mResource->mOutputChannel * mResource->mInputChannel;
    int actual_packCin = PACK_CIN;
    // shared part for all cases
    if (mResource->mNumQuantBit == 4){
        // int4 case
        buffer_size /= 2;
        cpy_size = UP_DIV(cpy_size, 2);
    } else if(mResource->mNumQuantBit == 8){
        actual_packCin /= 2;
    } else {/* More types to be supported. */}
    if(mOpenCLBackend->getRuntime()->hint().useCachedMmap <= 1){
        cl::Buffer filterBufferCL(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size);
        void *mapPtr = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(filterBufferCL, true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &res);
        if(mapPtr != nullptr && res == CL_SUCCESS){
            if(preAllocGpuMem){
                getInfoFromOpLowMemory(mapPtr);
            } else{
                ::memcpy(mapPtr, mFilterDataPtr, cpy_size);
            }
        } else {
            MNN_ERROR("set1x1WeightLowMemory: Map error ptrCL == nullptr \n");
            MNN_ASSERT(false);
        }
        mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(filterBufferCL, mapPtr);
        // Use Image load weights
        if(UP_DIV(mResource->mInputChannel, actual_packCin) <= 16384 && ROUND_UP(mResource->mOutputChannel, PACK_COUT) <= 16384){
            mResource->mUseImage = true;
        }
        auto staticMapAlloc = mOpenCLBackend->getStaticAllocatorMMap();
        if(mResource->mUseImage){
            size_t w = UP_DIV(mResource->mInputChannel, actual_packCin);
            size_t h = UP_DIV(mResource->mOutputChannel, PACK_COUT);
            if(mOpenCLBackend->getRuntime()->hint().useCachedMmap && staticMapAlloc != nullptr){
                mResource->mKernelImage = staticMapAlloc.get()->allocImage(w, h, CL_SIGNED_INT32);
            }else{
                mResource->mKernelImage.reset(new cl::Image2D(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_SIGNED_INT32), w, h, 0, nullptr, &res));
            }
            if (nullptr == mResource->mKernelImage.get() || res != CL_SUCCESS) {
                MNN_ERROR("Alloc Image %d x %d error, code:%d \n", (int)w, (int)h, (int)res);
            }
        }else{
            if(mOpenCLBackend->getRuntime()->hint().useCachedMmap && staticMapAlloc != nullptr){
                mResource->mKernelBuffer = staticMapAlloc.get()->allocBuffer(buffer_size);
            }else{
                mResource->mKernelBuffer.reset(new cl::Buffer(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size));
            }
        }
        convertToQuantWeight1x1Buffer(filterBufferCL);
    }else {
        if(preAllocGpuMem){
            getInfoFromOpLowMemory(nullptr);
        }
        // Use Image load weights
        if(UP_DIV(mResource->mInputChannel, actual_packCin) <= 16384 && ROUND_UP(mResource->mOutputChannel, PACK_COUT) <= 16384){
            mResource->mUseImage = true;
        }
        auto staticMapAlloc = mOpenCLBackend->getStaticAllocatorMMap();
        if(mResource->mUseImage){
            size_t w = UP_DIV(mResource->mInputChannel, actual_packCin);
            size_t h = UP_DIV(mResource->mOutputChannel, PACK_COUT);
            if(mOpenCLBackend->getRuntime()->hint().useCachedMmap && staticMapAlloc != nullptr){
                mResource->mKernelImage = staticMapAlloc.get()->allocImage(w, h, CL_SIGNED_INT32);
            }else{
                mResource->mKernelImage.reset(new cl::Image2D(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_SIGNED_INT32), w, h, 0, nullptr, &res));
            }
            if (nullptr == mResource->mKernelImage.get() || res != CL_SUCCESS) {
                MNN_ERROR("Alloc Image %d x %d error, code:%d \n", (int)w, (int)h, (int)res);
            }
        }else{
            if(mOpenCLBackend->getRuntime()->hint().useCachedMmap && staticMapAlloc != nullptr){
                mResource->mKernelBuffer = staticMapAlloc.get()->allocBuffer(buffer_size);
            }else{
                mResource->mKernelBuffer.reset(new cl::Buffer(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size));
            }
        }
    }
}
// set mFilter for the general kernels
void ConvBufLowMemoryExecution::setGeneralWeightLowMemory() {
    bool preAllocGpuMem = mResource->mInputChannel != 0 && mResource->mConv2dParams->quanParameter();
    if(preAllocGpuMem){
        mResource->mNumQuantBit = mResource->mConv2dParams->quanParameter()->aMaxOrBits();
        if(mResource->mNumQuantBit == 0){
            // support old model for external weight file with int4/int8 quant
            mResource->mNumQuantBit = ConvolutionCommon::getQuantBitFromExternalFile(mOp);
        }
    } else{
        getInfoFromOpLowMemory(nullptr);
    }
    
    if(mOpenCLBackend->getRuntime()->hint().useCachedMmap <= 1){
        std::shared_ptr<Tensor> filterBuffer(Tensor::createDevice<float>({ROUND_UP(mResource->mOutputChannel, 4), mResource->mInputChannel, mResource->mKernelWidth, mResource->mKernelHeight}));
        size_t buffer_size = filterBuffer->usize() / sizeof(float);
        size_t cpy_size = mResource->mOutputChannel * mResource->mInputChannel * mResource->mKernelWidth * mResource->mKernelHeight;
        if (mResource->mNumQuantBit == 4){
            buffer_size /= 2;
            cpy_size = UP_DIV(cpy_size, 2);
        }
        cl::Buffer filterBufferCL(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size);
        filterBuffer->buffer().device = (uint64_t)(&filterBufferCL);
        // map and pack data from filterDataPtr
        cl_int res;
        auto ptrCL = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(filterBufferCL, true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &res);
        if(ptrCL != nullptr && res == CL_SUCCESS) {
            if(preAllocGpuMem){
                getInfoFromOpLowMemory(ptrCL);
            } else{
                ::memcpy(ptrCL, mFilterDataPtr, cpy_size);
            }
        } else {
            MNN_ERROR("setGeneralWeightLowMemory: Map error ptrCL == nullptr \n");
        }
        mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(filterBufferCL, ptrCL);
        if (mResource->mNumQuantBit == 8) {
            // ROUND_UP(IC, 4), UP_DIV(OC, 4) * mKernelWidth * mKernelHeight
            mResource->mFilter.reset(Tensor::createDevice<int8_t>({1, UP_DIV(mResource->mOutputChannel, 4) * mResource->mKernelWidth * mResource->mKernelHeight, 1, 4 * ROUND_UP(mResource->mInputChannel, 4)}));
            mOpenCLBackend->onAcquireBuffer(mResource->mFilter.get(), Backend::STATIC);
        } else if (mResource->mNumQuantBit == 4){
            // ROUND_UP(IC, 4), UP_DIV(OC, 4) * mKernelWidth * mKernelHeight
            // For int4 case, data stored in mFilter should be uint8_t,
            // while "Tensor::createDevice<uint8_t>" occupies more memory than "Tensor::createDevice<int8_t>".
            // Therefore, we use "Tensor::createDevice<int8_t>" currently, leaving "Tensor::createDevice<uint8_t>" to be supported.
            mResource->mFilter.reset(Tensor::createDevice<int8_t>({1, UP_DIV(mResource->mOutputChannel, 4) * mResource->mKernelWidth * mResource->mKernelHeight, 1, 2 * ROUND_UP(mResource->mInputChannel, 4)}));
            mOpenCLBackend->onAcquireBuffer(mResource->mFilter.get(), Backend::STATIC);
        }
        // convert to NC4HW4
        MNN::OpenCL::BufferConvertor bufferConvertor{mOpenCLBackend->getOpenCLRuntime()};
        bufferConvertor.convertToNC4HW4Buffer(filterBuffer.get(), MNN::OpenCL::CONV2D_FILTER, mResource->mFilter.get(), mOpenCLBackend->getPrecision(), false, true, true, mResource->mNumQuantBit);
    }else{
        if(preAllocGpuMem){
            getInfoFromOpLowMemory(nullptr);
        }
        if (mResource->mNumQuantBit == 8) {
            // ROUND_UP(IC, 4), UP_DIV(OC, 4) * mKernelWidth * mKernelHeight
            mResource->mFilter.reset(Tensor::createDevice<int8_t>({1, UP_DIV(mResource->mOutputChannel, 4) * mResource->mKernelWidth * mResource->mKernelHeight, 1, 4 * ROUND_UP(mResource->mInputChannel, 4)}));
            mOpenCLBackend->onAcquireBuffer(mResource->mFilter.get(), Backend::STATIC);
        } else if (mResource->mNumQuantBit == 4){
            // ROUND_UP(IC, 4), UP_DIV(OC, 4) * mKernelWidth * mKernelHeight
            // For int4 case, data stored in mFilter should be uint8_t,
            // while "Tensor::createDevice<uint8_t>" occupies more memory than "Tensor::createDevice<int8_t>".
            // Therefore, we use "Tensor::createDevice<int8_t>" currently, leaving "Tensor::createDevice<uint8_t>" to be supported.
            mResource->mFilter.reset(Tensor::createDevice<int8_t>({1, UP_DIV(mResource->mOutputChannel, 4) * mResource->mKernelWidth * mResource->mKernelHeight, 1, 2 * ROUND_UP(mResource->mInputChannel, 4)}));
            mOpenCLBackend->onAcquireBuffer(mResource->mFilter.get(), Backend::STATIC);
        }
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
    const int total_kernel = 4;
    std::string kernelName[total_kernel] = {"conv_2d_int_c4h1w1", "conv_2d_int_c4h1w2", "conv_2d_int_c4h1w4", "conv_2d_int_c8h1w4"};
    int itemC[total_kernel] = {4, 4, 4, 8};
    int itemH[total_kernel] = {1, 1, 1, 1};
    int itemW[total_kernel] = {1, 2, 4, 4};
    int actual_kernel = total_kernel;
    std::shared_ptr<KernelWrap> kernel[total_kernel];
    std::vector<uint32_t> globalWorkSize[total_kernel];
    std::vector<uint32_t> localWorkSize[total_kernel];
    std::pair<int, int> min_cost(INT_MAX, 0);//(min_time, min_index)
    // MNN_PRINT("Checking kernel %d.\n", knlCheck);
    for (int knl_idx = 0; knl_idx < actual_kernel; knl_idx++) {
        std::set<std::string> buildOption = mResource->mBuildOptions;
        if(itemC[knl_idx] == 8 && outputShape.at(3) % itemC[knl_idx] > 0 && outputShape.at(3) % itemC[knl_idx] <= 4){
            buildOption.emplace("-DCHANNEL_BOUNDARY_PROTECT");
        }
        if((outputShape.at(2) % itemW[knl_idx]) != 0 || (outputShape.at(1) % itemH[knl_idx]) != 0){
            buildOption.emplace("-DBLOCK_LEAVE");
        }
        if(inputChannels % 4 != 0){
            buildOption.emplace("-DINPUT_CHANNEL_BOUNDARY_PROTECT");
        }
        kernel[knl_idx]        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("conv_2d_int_buf", kernelName[knl_idx], buildOption, mOpenCLBackend->getPrecision());
        uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(kernel[knl_idx]));

        globalWorkSize[knl_idx] = {static_cast<uint32_t>(UP_DIV(outputShape.at(3), itemC[knl_idx]) * UP_DIV(outputShape.at(2), itemW[knl_idx])), static_cast<uint32_t>(outputShape.at(0) * UP_DIV(outputShape.at(1), itemH[knl_idx]))};
        uint32_t idx            = 0;
        cl_int ret = CL_SUCCESS;
        ret |= kernel[knl_idx]->get().setArg(idx++, globalWorkSize[knl_idx][0]);
        ret |= kernel[knl_idx]->get().setArg(idx++, globalWorkSize[knl_idx][1]);
        ret |= kernel[knl_idx]->get().setArg(idx++, openCLBuffer(input));
        ret |= kernel[knl_idx]->get().setArg(idx++, openCLBuffer(mResource->mFilter.get()));
        ret |= kernel[knl_idx]->get().setArg(idx++, *mResource->mDequantScaleOffsetBuffer.get());
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
        ret |= kernel[knl_idx]->get().setArg(idx++, static_cast<float>(mResource->mCoef));
        MNN_CHECK_CL_SUCCESS(ret, "setArg ConvBufLowMemory Kernel Select");
        std::pair<std::vector<uint32_t>, int> retTune;
        retTune = localWS2DDefault(globalWorkSize[knl_idx], maxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), kernelName[knl_idx] + info, kernel[knl_idx], mOpenCLBackend->getCLTuneLevel(), "conv_2d_int_buf");
        if(min_cost.first > retTune.second) {
            min_cost.first = retTune.second;
            min_cost.second = knl_idx;
            mLocalWorkSize = {retTune.first[0], retTune.first[1]};
        }
    }
    int min_index  = min_cost.second;
    mGlobalWorkSize = {globalWorkSize[min_index][0], globalWorkSize[min_index][1]};

    std::set<std::string> buildOption = mResource->mBuildOptions;
    if(itemC[min_index] == 8 && outputShape.at(3) % itemC[min_index] > 0 && outputShape.at(3) % itemC[min_index] <= 4){
        buildOption.emplace("-DCHANNEL_BOUNDARY_PROTECT");
    }
    if((outputShape.at(2) % itemW[min_index]) != 0 || (outputShape.at(1) % itemH[min_index]) != 0){
        buildOption.emplace("-DBLOCK_LEAVE");
    }
    if(inputChannels % 4 != 0){
        buildOption.emplace("-DINPUT_CHANNEL_BOUNDARY_PROTECT");
    }
    unit.kernel        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("conv_2d_int_buf", kernelName[min_index], buildOption, mOpenCLBackend->getPrecision());

    uint32_t idx            = 0;
    cl_int ret = CL_SUCCESS;
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[0]);
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[1]);
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(input));
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mResource->mFilter.get()));
    ret |= unit.kernel->get().setArg(idx++, *mResource->mDequantScaleOffsetBuffer.get());
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
    ret |= unit.kernel->get().setArg(idx++, static_cast<float>(mResource->mCoef));
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
    int channelPack = 2;
    if(mResource->mNumQuantBit == 4){
        channelPack = 4;
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
    
    //weight inverse quantization and rearrange
    {
        auto &unit = mUnits[0];
        int outputChannelAlign = ROUND_UP(mResource->mOutputChannel, alignN);
        int outputChannel4Align = ROUND_UP(mResource->mOutputChannel, 4);
        int inputChannel4Align = ROUND_UP(mResource->mInputChannel, 4);
        std::set<std::string> buildOption = mResource->mBuildOptions;
        if(mResource->mUseImage){
            buildOption.emplace("-DUSE_IMAGE");
        }
        mGlobalWorkSize = {static_cast<uint32_t>(UP_DIV(mResource->mInputChannel, channelPack)), static_cast<uint32_t>(UP_DIV(mResource->mOutputChannel, 8))};
        unit.kernel = runtime->buildKernel("gemm_conv1x1_buf", "inverse_quant_weight", buildOption, mOpenCLBackend->getPrecision());
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
        ret |= unit.kernel->get().setArg(idx++, *mResource->mDequantScaleOffsetBuffer.get());
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mConvGemmWeightTensor.get()));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(mResource->mInputChannel));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(inputChannel4Align));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(outputChannelAlign));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(outputChannel4Align));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(blockDim));
        ret |= unit.kernel->get().setArg(idx++, static_cast<float>(mResource->mCoef));
        MNN_CHECK_CL_SUCCESS(ret, "setArg inverse_quant_weight");
        
        mLocalWorkSize = localWS2DDefault(mGlobalWorkSize, maxWorkGroupSize, runtime, "inverse_quant_weight", unit.kernel, mOpenCLBackend->getCLTuneLevel(), "gemm_conv1x1_buf").first;
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
        unit.kernel = mOpenCLBackend->getOpenCLRuntime()->buildKernel("gemm_buf", "transpose_pad", buildOptions, mOpenCLBackend->getPrecision());
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
        mLocalWorkSize = localWS2DDefault(mGlobalWorkSize, maxWorkGroupSize, runtime, "transpose_pad", unit.kernel, mOpenCLBackend->getCLTuneLevel(), "gemm_buf").first;

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
        // generate cache for every option
        std::vector<int> pack_m_vec = {1, 4, 8};
        for (auto p : pack_m_vec) {
            auto option = mResource->mBuildOptions;
            option.emplace("-DM_VEC=" + std::to_string(p));
            auto kernel = runtime->buildKernel("gemm_buf", "transpose_bias", option, mOpenCLBackend->getPrecision());
        }
        unit.kernel = runtime->buildKernel("gemm_buf", "transpose_bias", buildOptions, mOpenCLBackend->getPrecision());
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
        mLocalWorkSize = localWS2DDefault(mGlobalWorkSize, maxWorkGroupSize, runtime, "transpose_bias", unit.kernel, mOpenCLBackend->getCLTuneLevel(), "gemm_buf").first;
        mOpenCLBackend->recordKernel2d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
        unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1]};
        unit.localWorkSize = {mLocalWorkSize[0], mLocalWorkSize[1]};
    }
    mOpenCLBackend->onReleaseBuffer(mConvGemmWeightTensor.get(), Backend::DYNAMIC);
    mOpenCLBackend->onReleaseBuffer(mConvGemmInpTensor.get(), Backend::DYNAMIC);
    mOpenCLBackend->onReleaseBuffer(mConvGemmOutTensor.get(), Backend::DYNAMIC);
    
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
    bool useLocalMem = inputChannels >= 32;
    std::string info = std::to_string(inputChannels) + "_" + std::to_string(outChannel);
    std::set<std::string> buildOption = mResource->mBuildOptions;
    int inputChannelLeaves = 0;
    if(mResource->mNumQuantBit == 4){
        inputChannelLeaves = useLocalMem ? (inputChannels % 4) : (blockDim % 4);
    } else {
        inputChannelLeaves = useLocalMem ? (inputChannels % 2) : (blockDim % 2);
    }
    if(outChannel % 8 != 0){
        buildOption.emplace("-DOUTPUT_CHANNEL_LEAVES");
    }
    buildOption.emplace("-DINPUT_CHANNEL_LEAVES_NUM=" + std::to_string(inputChannelLeaves));
    if(mResource->mUseImage){
        buildOption.emplace("-DUSE_IMAGE");
    }
    
    int local_size = useLocalMem ? 128 : 1;
    if(useLocalMem && mOpenCLBackend->getCLTuneLevel() != None && mOpenCLBackend->getCLTuneLevel() != Fast){
        int min_time = INT_MAX;
        for (int ksize = 8; ksize <= 256; ksize*=2) {
            auto option = buildOption;
            option.emplace("-DWGS=" + std::to_string(ksize));
            auto kernel = mOpenCLBackend->getOpenCLRuntime()->buildKernel("gemv_conv1x1_buf", "gemv_conv_c8_buf", option, mOpenCLBackend->getPrecision());
            uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(kernel));
            std::vector<uint32_t> gws = {static_cast<uint32_t>(ksize), static_cast<uint32_t>(UP_DIV(outChannel, 8))};
            std::vector<uint32_t> lws = {static_cast<uint32_t>(ksize), 1};
            uint32_t idx = 0;
            cl_int ret = CL_SUCCESS;
            ret |= kernel->get().setArg(idx++, static_cast<int>(gws[0]));
            ret |= kernel->get().setArg(idx++, static_cast<int>(gws[1]));
            ret |= kernel->get().setArg(idx++, static_cast<int>(gws[1]));
            ret |= kernel->get().setArg(idx++, openCLBuffer(input));
            if(mResource->mUseImage){
                ret |= kernel->get().setArg(idx++, *mResource->mKernelImage.get());
            }else{
                ret |= kernel->get().setArg(idx++, *mResource->mKernelBuffer.get());
            }
            ret |= kernel->get().setArg(idx++, *mResource->mDequantScaleOffsetBuffer.get());
            ret |= kernel->get().setArg(idx++, openCLBuffer(mResource->mBias.get()));
            ret |= kernel->get().setArg(idx++, openCLBuffer(output));
            ret |= kernel->get().setArg(idx++, static_cast<int>(outputChannelBlocks));
            ret |= kernel->get().setArg(idx++, static_cast<int>(inputChannelBlocks));
            ret |= kernel->get().setArg(idx++, static_cast<int>(outputChannelBlocks));
            ret |= kernel->get().setArg(idx++, static_cast<int>(inputChannelBlocks));
            ret |= kernel->get().setArg(idx++, inputChannels);
            ret |= kernel->get().setArg(idx++, static_cast<int>(blockNum));
            ret |= kernel->get().setArg(idx++, static_cast<int>(blockDim));
            ret |= kernel->get().setArg(idx++, static_cast<float>(mResource->mCoef));
            MNN_CHECK_CL_SUCCESS(ret, "setArg gemv_conv_c8_buf Kernel Select");
            std::pair<std::vector<uint32_t>, int> retTune;
            int cost_time = get2DUseLocalMemTime(gws, lws, mOpenCLBackend->getOpenCLRuntime(), "gemv_conv_c8_buf" + info, kernel, "gemv_conv1x1_buf");
            if(min_time > cost_time) {
                local_size = ksize;
                min_time = cost_time;
            }
        }
    }
    
    buildOption.emplace("-DWGS=" + std::to_string(local_size));
    mGlobalWorkSize = {static_cast<uint32_t>(local_size), static_cast<uint32_t>(UP_DIV(outChannel, 8))};
    unit.kernel        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("gemv_conv1x1_buf", "gemv_conv_c8_buf", buildOption, mOpenCLBackend->getPrecision());
    uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(unit.kernel));
    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(mGlobalWorkSize[0]));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(mGlobalWorkSize[1]));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(mGlobalWorkSize[1]));
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(input));
    if(mResource->mUseImage){
        ret |= unit.kernel->get().setArg(idx++, *mResource->mKernelImage.get());
    }else{
        ret |= unit.kernel->get().setArg(idx++, *mResource->mKernelBuffer.get());
    }
    ret |= unit.kernel->get().setArg(idx++, *mResource->mDequantScaleOffsetBuffer.get());
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mResource->mBias.get()));
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(output));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(outputChannelBlocks));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(inputChannelBlocks));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(outputChannelBlocks));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(inputChannelBlocks));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(inputChannels));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(blockNum));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(blockDim));
    ret |= unit.kernel->get().setArg(idx++, static_cast<float>(mResource->mCoef));
    MNN_CHECK_CL_SUCCESS(ret, "setArg gemv_conv_c8_buf");
    if(useLocalMem){
        mLocalWorkSize = {static_cast<uint32_t>(local_size), 1};
    }else{
        mLocalWorkSize = localWS2DDefault(mGlobalWorkSize, maxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), "gemv_conv_c8_buf" + info, unit.kernel, mOpenCLBackend->getCLTuneLevel(), "gemv_conv1x1_buf").first;
    }
    mOpenCLBackend->recordKernel2d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
    unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1]};
    unit.localWorkSize = {mLocalWorkSize[0], mLocalWorkSize[1]};
    return;
}
void ConvBufLowMemoryExecution::tuneGemmLowMemory(Tensor * input, Tensor * output) {
    mUnits.resize(1);
    auto &unit = mUnits[0];
    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);
    const int outChannel = outputShape.at(3);
    const int inputChannels = inputShape.at(3);
    const int batch = outputShape.at(0);
    const int width_height = outputShape.at(1) * outputShape.at(2);
    const int inputChannelAlign = ROUND_UP(inputChannels, 4);
    const int outputChannelAlign = ROUND_UP(outChannel, 4);
    const int blockNum = mResource->mBlockSize;
    const int blockDim = mResource->mInputChannel / mResource->mBlockSize;
    
    int global_y = batch * width_height;
    std::string kernelName = "gemm_b4_c8";
    std::set<std::string> buildOption = mResource->mBuildOptions;
    int inputChannelLeaves = 0;
    int inputBatchLeaves = global_y % 4;
    if(mResource->mNumQuantBit == 4){
        inputChannelLeaves = blockDim % 4;
        kernelName += "_int4_buf";
    } else {
        inputChannelLeaves = blockDim % 4;
        kernelName += "_int8_buf";
    }
    buildOption.emplace("-DINPUT_CHANNEL_LEAVES_NUM=" + std::to_string(inputChannelLeaves));
    buildOption.emplace("-DINPUT_BATCH_LEAVES_NUM=" + std::to_string(inputBatchLeaves));
    if(mResource->mUseImage){
        buildOption.emplace("-DUSE_IMAGE");
    }
    // generate cache for every option
    for (int i = 0; i < 4; i++) {
        std::set<std::string> option = mResource->mBuildOptions;
        if(mResource->mUseImage){
            option.emplace("-DUSE_IMAGE");
        }
        option.emplace("-DINPUT_CHANNEL_LEAVES_NUM=" + std::to_string(inputChannelLeaves));
        option.emplace("-DINPUT_BATCH_LEAVES_NUM=" + std::to_string(i));
        auto kernel = mOpenCLBackend->getOpenCLRuntime()->buildKernel("gemm_conv1x1_buf", kernelName, option, mOpenCLBackend->getPrecision());
    }
    std::string info = std::to_string(inputChannels) + "_" + std::to_string(outChannel);
    if(global_y <= 16) {
        mUnits.resize(3);
        int outputChannelAlign8 = ROUND_UP(outChannel, 8);
        mConvGemmInpTensor.reset(Tensor::createDevice<float>({inputChannelAlign * ROUND_UP(global_y, 4)}));
        mConvGemmOutTensor.reset(Tensor::createDevice<float>({outputChannelAlign8 * ROUND_UP(global_y, 4)}));
        mOpenCLBackend->onAcquireBuffer(mConvGemmInpTensor.get(), Backend::DYNAMIC);
        mOpenCLBackend->onAcquireBuffer(mConvGemmOutTensor.get(), Backend::DYNAMIC);
        mOpenCLBackend->onReleaseBuffer(mConvGemmInpTensor.get(), Backend::DYNAMIC);
        mOpenCLBackend->onReleaseBuffer(mConvGemmOutTensor.get(), Backend::DYNAMIC);
        
        {
            //c4nhw4 -> nhwc
            auto &unit = mUnits[0];
            unit.kernel = mOpenCLBackend->getOpenCLRuntime()->buildKernel("gemm_conv1x1_buf", "gemm_c4nhw4_to_nhwc", buildOption, mOpenCLBackend->getPrecision());
            uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(unit.kernel));
            
            mGlobalWorkSize = {static_cast<uint32_t>(UP_DIV(global_y, 4)), static_cast<uint32_t>(UP_DIV(inputChannels, 4))};
            uint32_t idx = 0;
            cl_int ret = CL_SUCCESS;
            ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[0]);
            ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[1]);
            ret |= unit.kernel->get().setArg(idx++, openCLBuffer(input));
            ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mConvGemmInpTensor.get()));
            ret |= unit.kernel->get().setArg(idx++, static_cast<int>(global_y));
            ret |= unit.kernel->get().setArg(idx++, static_cast<int>(inputChannels));
            ret |= unit.kernel->get().setArg(idx++, static_cast<int>(inputChannelAlign));
            MNN_CHECK_CL_SUCCESS(ret, "setArg gemm_c4nhw4_to_nhwc");
            mLocalWorkSize = localWS2DDefault(mGlobalWorkSize, maxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), "gemm_c4nhw4_to_nhwc", unit.kernel, mOpenCLBackend->getCLTuneLevel(), "gemm_conv1x1_buf").first;
            mOpenCLBackend->recordKernel2d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
            unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1]};
            unit.localWorkSize = {mLocalWorkSize[0], mLocalWorkSize[1]};
        }
        {
            const int inputChannelBlocks = UP_DIV(inputChannels, 4);
            const int outputChannelBlocks = UP_DIV(outChannel, 4);
            auto &unit = mUnits[1];
            std::set<std::string> buildOption = mResource->mBuildOptions;
            if(mResource->mUseImage){
                buildOption.emplace("-DUSE_IMAGE");
            }
            buildOption.emplace("-DCOMPUTE_BATCH");
            
            int local_size = 64;
            if(mOpenCLBackend->getCLTuneLevel() != None && mOpenCLBackend->getCLTuneLevel() != Fast){
                int min_time = INT_MAX;
                for (int ksize = 16; ksize <= 256; ksize*=2) {
                    auto option = buildOption;
                    option.emplace("-DWGS=" + std::to_string(ksize));
                    auto kernel = mOpenCLBackend->getOpenCLRuntime()->buildKernel("gemv_conv1x1_buf", "gemv_conv_c8_buf", option, mOpenCLBackend->getPrecision());
                    uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(kernel));
                    std::vector<uint32_t> gws = {static_cast<uint32_t>(ksize), static_cast<uint32_t>(UP_DIV(outChannel, 8)), static_cast<uint32_t>(UP_DIV(global_y, 4))};
                    std::vector<uint32_t> lws = {static_cast<uint32_t>(ksize), 1, 1};
                    uint32_t idx = 0;
                    cl_int ret = CL_SUCCESS;
                    ret |= kernel->get().setArg(idx++, static_cast<int>(gws[0]));
                    ret |= kernel->get().setArg(idx++, static_cast<int>(gws[1]));
                    ret |= kernel->get().setArg(idx++, static_cast<int>(gws[2]));
                    ret |= kernel->get().setArg(idx++, openCLBuffer(mConvGemmInpTensor.get()));
                    if(mResource->mUseImage){
                        ret |= kernel->get().setArg(idx++, *mResource->mKernelImage.get());
                    }else{
                        ret |= kernel->get().setArg(idx++, *mResource->mKernelBuffer.get());
                    }
                    ret |= kernel->get().setArg(idx++, *mResource->mDequantScaleOffsetBuffer.get());
                    ret |= kernel->get().setArg(idx++, openCLBuffer(mResource->mBias.get()));
                    ret |= kernel->get().setArg(idx++, openCLBuffer(mConvGemmOutTensor.get()));
                    ret |= kernel->get().setArg(idx++, static_cast<int>(outputChannelAlign8));
                    ret |= kernel->get().setArg(idx++, static_cast<int>(inputChannelAlign));
                    ret |= kernel->get().setArg(idx++, static_cast<int>(outputChannelBlocks));
                    ret |= kernel->get().setArg(idx++, static_cast<int>(inputChannelBlocks));
                    ret |= kernel->get().setArg(idx++, inputChannels);
                    ret |= kernel->get().setArg(idx++, static_cast<int>(blockNum));
                    ret |= kernel->get().setArg(idx++, static_cast<int>(blockDim));
                    ret |= kernel->get().setArg(idx++, static_cast<float>(mResource->mCoef));
                    MNN_CHECK_CL_SUCCESS(ret, "setArg gemv_conv_c8_buf Kernel Select");
                    std::pair<std::vector<uint32_t>, int> retTune;
                    int cost_time = get2DUseLocalMemTime(gws, lws, mOpenCLBackend->getOpenCLRuntime(), "gemv_conv_c8_buf" + info + "_batch", kernel, "gemv_conv1x1_buf");
                    if(min_time > cost_time) {
                        local_size = ksize;
                        min_time = cost_time;
                    }
                }
            }
            buildOption.emplace("-DWGS=" + std::to_string(local_size));
            mGlobalWorkSize = {static_cast<uint32_t>(local_size), static_cast<uint32_t>(UP_DIV(outChannel, 8)), static_cast<uint32_t>(UP_DIV(global_y, 4))};
            unit.kernel        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("gemv_conv1x1_buf", "gemv_conv_c8_buf", buildOption, mOpenCLBackend->getPrecision());
            uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(unit.kernel));
            uint32_t idx = 0;
            cl_int ret = CL_SUCCESS;
            ret |= unit.kernel->get().setArg(idx++, static_cast<int>(mGlobalWorkSize[0]));
            ret |= unit.kernel->get().setArg(idx++, static_cast<int>(mGlobalWorkSize[1]));
            ret |= unit.kernel->get().setArg(idx++, static_cast<int>(mGlobalWorkSize[2]));
            ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mConvGemmInpTensor.get()));
            if(mResource->mUseImage){
                ret |= unit.kernel->get().setArg(idx++, *mResource->mKernelImage.get());
            }else{
                ret |= unit.kernel->get().setArg(idx++, *mResource->mKernelBuffer.get());
            }
            ret |= unit.kernel->get().setArg(idx++, *mResource->mDequantScaleOffsetBuffer.get());
            ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mResource->mBias.get()));
            ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mConvGemmOutTensor.get()));
            ret |= unit.kernel->get().setArg(idx++, static_cast<int>(outputChannelAlign8));
            ret |= unit.kernel->get().setArg(idx++, static_cast<int>(inputChannelAlign));
            ret |= unit.kernel->get().setArg(idx++, static_cast<int>(outputChannelBlocks));
            ret |= unit.kernel->get().setArg(idx++, static_cast<int>(inputChannelBlocks));
            ret |= unit.kernel->get().setArg(idx++, static_cast<int>(inputChannels));
            ret |= unit.kernel->get().setArg(idx++, static_cast<int>(blockNum));
            ret |= unit.kernel->get().setArg(idx++, static_cast<int>(blockDim));
            ret |= unit.kernel->get().setArg(idx++, static_cast<float>(mResource->mCoef));
            MNN_CHECK_CL_SUCCESS(ret, "setArg gemv_conv_c8_buf");
            mLocalWorkSize = {static_cast<uint32_t>(local_size), 1, 1};
            mOpenCLBackend->recordKernel3d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
            unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
            unit.localWorkSize = {mLocalWorkSize[0], mLocalWorkSize[1], mLocalWorkSize[2]};
        }
        {
            auto &unit = mUnits[2];
            unit.kernel = mOpenCLBackend->getOpenCLRuntime()->buildKernel("gemm_conv1x1_buf", "gemm_nhwc_to_c4nhw4", buildOption, mOpenCLBackend->getPrecision());
            uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(unit.kernel));
            mGlobalWorkSize = {static_cast<uint32_t>(UP_DIV(global_y, 4)), static_cast<uint32_t>(UP_DIV(outChannel, 4))};
            uint32_t idx = 0;
            cl_int ret = CL_SUCCESS;
            ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[0]);
            ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[1]);
            ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mConvGemmOutTensor.get()));
            ret |= unit.kernel->get().setArg(idx++, openCLBuffer(output));
            ret |= unit.kernel->get().setArg(idx++, static_cast<int>(global_y));
            ret |= unit.kernel->get().setArg(idx++, static_cast<int>(outputChannelAlign8));
            MNN_CHECK_CL_SUCCESS(ret, "setArg gemm_nhwc_to_c4nhw4");
            mLocalWorkSize = localWS2DDefault(mGlobalWorkSize, maxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), "gemm_nhwc_to_c4nhw4", unit.kernel, mOpenCLBackend->getCLTuneLevel(), "gemm_conv1x1_buf").first;
            mOpenCLBackend->recordKernel2d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
            unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1]};
            unit.localWorkSize = {mLocalWorkSize[0], mLocalWorkSize[1]};
        }
        return;
    }
    unit.kernel = mOpenCLBackend->getOpenCLRuntime()->buildKernel("gemm_conv1x1_buf", kernelName, buildOption, mOpenCLBackend->getPrecision());
    uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(unit.kernel));
    
    mGlobalWorkSize = {static_cast<uint32_t>(UP_DIV(global_y, 4)), static_cast<uint32_t>(UP_DIV(outChannel, 8))};
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
    ret |= unit.kernel->get().setArg(idx++, *mResource->mDequantScaleOffsetBuffer.get());
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mResource->mBias.get()));
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(output));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(global_y));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(outputChannelAlign));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(inputChannelAlign));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(blockNum));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(blockDim));
    ret |= unit.kernel->get().setArg(idx++, mResource->mCoef);
    MNN_CHECK_CL_SUCCESS(ret, "setArg gemm_conv1x1_buf");
    mLocalWorkSize = localWS2DDefault(mGlobalWorkSize, maxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), kernelName + info, unit.kernel, mOpenCLBackend->getCLTuneLevel(), "gemm_conv1x1_buf").first;
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
    mResource->mInputChannel = conv2dCommonParams->inputCount();
    mResource->mOutputChannel = conv2dCommonParams->outputCount();
        
    //select opt conv method
    if (mResource->mKernelHeight == mResource->mKernelWidth && mResource->mKernelHeight == 1 && mResource->mStrides[0] == 1 && mResource->mStrides[1] == 1 && conv2dCommonParams->padX() == 0 && conv2dCommonParams->padY() == 0 && conv2dCommonParams->dilateX() == 1 && conv2dCommonParams->dilateY() == 1) {
        set1x1WeightLowMemory();
        mResource->mConv1x1Opt = true;
    }else {
        // set mFilter for not 1x1 case
        setGeneralWeightLowMemory();
    }
    // Create Kernel
    if (conv2dCommonParams->relu()) {
        mResource->mBuildOptions.emplace("-DRELU");
    } else if (conv2dCommonParams->relu6()) {
        mResource->mBuildOptions.emplace("-DRELU6");
    }
    mResource->mBuildOptions.emplace("-DQUANT_BIT=" + std::to_string(mResource->mNumQuantBit));
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
            std::pair<std::vector<uint32_t>, uint32_t> tuneInfo;
            std::string info = "convBufLowMemory_" + std::to_string(mResource->mInputChannel) + "_" + std::to_string(mResource->mOutputChannel);
            if(batch > 16){
                if(getTunedInfo(info, {static_cast<unsigned int>(batch)}, tuneInfo, mOpenCLBackend->getOpenCLRuntime())){
                    mUseFPWeight = tuneInfo.first[0];
                } else{
                    if((mOpenCLBackend->getCLTuneLevel() == Heavy || mOpenCLBackend->getCLTuneLevel() == Wide)){
                        setRecordClose closeRecord(mOpenCLBackend);
                        tuneGemmLowMemory(input, output);
                        auto shortBatchTime = getExecuteTime();
                        mUseFPWeight = true;
                        useFPWeightGemmLowMemory(input, output);
                        auto longBatchTime = getExecuteTime();
                        mUseFPWeight = false;
                        if(longBatchTime < shortBatchTime){
                            mUseFPWeight = true;
                        }
                        std::pair<std::vector<uint32_t>, uint32_t> tuneInfoTmp = std::make_pair<std::vector<uint32_t>, uint32_t>({mUseFPWeight}, 0);
                        setTunedInfo(info, {static_cast<unsigned int>(batch)}, tuneInfoTmp, mOpenCLBackend->getOpenCLRuntime(), "gemm_conv1x1_buf");
                    } else{
                        if(batch > 512){
                            mUseFPWeight = true;
                        }
                    }
                }
            }
            if(mUseFPWeight){
                useFPWeightGemmLowMemory(input, output);
            }else{
                tuneGemmLowMemory(input, output);
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

int ConvBufLowMemoryExecution::getExecuteTime(){
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
    int executeTime = 0;
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    auto res = CL_SUCCESS;
    if(mUseFPWeight){
        // arrange input and weight
        int i = 0;
        for (; i < 2; ++i){
            auto unit = mUnits[i];
            cl::Event event;
            res = runtime->commandQueue().enqueueNDRangeKernel(unit.kernel->get(),
                                                   cl::NullRange,
                                                   unit.globalWorkSize,
                                                   unit.localWorkSize,
                                                   nullptr,
                                                   &event);
            executeTime += runtime->getEventTime(event);
        }
        // call gemm execute
        executeTime += mStrassenComputor->getExecuteTime();
        
        // rearrange output
        for (; i < mUnits.size(); ++i){
            auto unit = mUnits[i];
            cl::Event event;
            res = runtime->commandQueue().enqueueNDRangeKernel(unit.kernel->get(),
                                                   cl::NullRange,
                                                   unit.globalWorkSize,
                                                   unit.localWorkSize,
                                                   nullptr,
                                                   &event);
            executeTime += runtime->getEventTime(event);
        }
    }else{
        for (auto &unit : mUnits) {
            cl::Event event;
            res = runtime->commandQueue().enqueueNDRangeKernel(unit.kernel->get(),
                                                               cl::NullRange,
                                                               unit.globalWorkSize,
                                                               unit.localWorkSize,
                                                               nullptr,
                                                               &event);
            executeTime += runtime->getEventTime(event);
        }
    }
    return executeTime;
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
