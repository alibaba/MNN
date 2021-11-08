//
//  ConvBufExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED

#include "ConvBufExecution.hpp"
#include "ConvBufWinograd.hpp"
#include "core/ConvolutionCommon.hpp"
#include "core/Backend.hpp"
#include "RasterBufExecution.hpp"

namespace MNN {
namespace OpenCL {

std::pair<std::vector<uint32_t>,  uint32_t> ConvBufCommonExecution::gws2dLwsTune(const cl::Kernel &kernel, const std::vector<uint32_t> &gws, const std::string &kernelName, const uint32_t maxWorkGroupSize) {
    MNN_ASSERT(gws.size() == 2);
    
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    auto maxWorkItemSizes = runtime->getMaxWorkItemSizes();
    MNN_ASSERT(maxWorkItemSizes.size() >= 2);
    
    auto& tunedLws = runtime->tunedLwsMap();
    std::pair<std::string, std::vector<uint32_t>> info = std::make_pair(kernelName, gws);
    if (tunedLws.find(info) != tunedLws.end()) {
        //printf("ConvBuf2dGeneralLocalWS Found! gws:%d %d lws:%d %d\n", gws[0], gws[1], tunedLws[info][0], tunedLws[info][1]);
        return tunedLws[info];
    }
    
    std::vector<uint32_t> lws(3, 1);
    std::vector<uint32_t> lws_prefer(3, 1);
    uint32_t min_cost = UINT_MAX;
    
    if(runtime->getCLTuneLevel() == Heavy) {
        while(lws[1] <= gws[1] || lws[1] <= 6) {
            lws[0] = 1;
            while(lws[0] <= gws[0] || lws[0] <= 6) {
                if(lws[0] <= maxWorkItemSizes[0] && lws[1] <= maxWorkItemSizes[1] && lws[0]*lws[1] <= maxWorkGroupSize) {
                    cl::Event event;
                    std::vector<uint32_t> internalGlobalWS(2, 1);
                    for (size_t i = 0; i < gws.size(); ++i) {
                        internalGlobalWS[i] = ROUND_UP(gws[i], std::max((uint32_t)1, lws[i]));
                    }
                    cl_int res = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueNDRangeKernel(
                                    kernel, cl::NullRange,
                                    cl::NDRange(internalGlobalWS[0], internalGlobalWS[1]),
                                    cl::NDRange(lws[0], lws[1]),
                                    nullptr, &event);
                    MNN_CHECK_CL_SUCCESS(res, kernelName.c_str());

                    int cost_time = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
                    if(cost_time < min_cost) {
                        min_cost = cost_time;
                        lws_prefer[0] = lws[0];
                        lws_prefer[1] = lws[1];
                    }
                }
                lws[0]++;
            }
            lws[1]++;
        }
    } else if(runtime->getCLTuneLevel() == Wide) {
        while(lws[1] <= gws[1] || lws[1] <= 6) {
            lws[0] = 1;
            while(lws[0] <= gws[0] || lws[0] <= 6) {
                if(lws[0] <= maxWorkItemSizes[0] && lws[1] <= maxWorkItemSizes[1] && lws[0]*lws[1] <= maxWorkGroupSize) {
                    cl::Event event;
                    std::vector<uint32_t> internalGlobalWS(2, 1);
                    for (size_t i = 0; i < gws.size(); ++i) {
                        internalGlobalWS[i] = ROUND_UP(gws[i], std::max((uint32_t)1, lws[i]));
                    }
                    cl_int res = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueNDRangeKernel(
                                    kernel, cl::NullRange,
                                    cl::NDRange(internalGlobalWS[0], internalGlobalWS[1]),
                                    cl::NDRange(lws[0], lws[1]),
                                    nullptr, &event);
                    MNN_CHECK_CL_SUCCESS(res, kernelName.c_str());

                    int cost_time = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
                    if(cost_time < min_cost) {
                        min_cost = cost_time;
                        lws_prefer[0] = lws[0];
                        lws_prefer[1] = lws[1];
                    }
                }
                do {
                    lws[0]++;
                }
                while(((2*gws[0])%lws[0] > 1) && (lws[0] & (lws[0] - 1)) != 0 && (lws[0] <= gws[0]) && (lws[0] > 6));//divisible powOfTwo lessThanSix
            }
            do {
                lws[1]++;
            }
            while(((2*gws[1])%lws[1] > 1) && (lws[1] & (lws[1] - 1)) != 0 && (lws[1] <= gws[1]) && (lws[1] > 6));//divisible powOfTwo lessThanSix
        }
    } else if(runtime->getCLTuneLevel() == Normal) {
        while(lws[1] <= gws[1] && lws[1] <= 6) {
            lws[0] = 1;
            while(lws[0] <= gws[0] || lws[0] <= 6) {
                if(lws[0] <= maxWorkItemSizes[0] && lws[1] <= maxWorkItemSizes[1] && lws[0]*lws[1] <= maxWorkGroupSize) {
                    cl::Event event;
                    std::vector<uint32_t> internalGlobalWS(2, 1);
                    for (size_t i = 0; i < gws.size(); ++i) {
                        internalGlobalWS[i] = ROUND_UP(gws[i], std::max((uint32_t)1, lws[i]));
                    }
                    cl_int res = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueNDRangeKernel(
                                    kernel, cl::NullRange,
                                    cl::NDRange(internalGlobalWS[0], internalGlobalWS[1]),
                                    cl::NDRange(lws[0], lws[1]),
                                    nullptr, &event);
                    MNN_CHECK_CL_SUCCESS(res, kernelName.c_str());

                    int cost_time = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
                    if(cost_time < min_cost) {
                        min_cost = cost_time;
                        lws_prefer[0] = lws[0];
                        lws_prefer[1] = lws[1];
                    }
                }
                do {
                    lws[0]++;
                }
                while(((2*gws[0])%lws[0] > 1) && (lws[0] & (lws[0] - 1)) != 0 && (lws[0] <= gws[0]) && (lws[0] > 6));//divisible powOfTwo lessThanSix
            }
            do {
                lws[1]++;
            }
            while(((2*gws[1])%lws[1] > 1) && (lws[1] & (lws[1] - 1)) != 0 && (lws[1] <= gws[1]) && (lws[1] <= 6));//divisible powOfTwo lessThanSix
        }
    } else if(runtime->getCLTuneLevel() == Fast) {
        while(lws[1] <= gws[1] && lws[1] <= 6) {
            lws[0] = 1;
            while(lws[0] <= gws[0] && lws[0] <= 6) {
                if(lws[0] <= maxWorkItemSizes[0] && lws[1] <= maxWorkItemSizes[1] && lws[0]*lws[1] <= maxWorkGroupSize) {
                    cl::Event event;
                    std::vector<uint32_t> internalGlobalWS(2, 1);
                    for (size_t i = 0; i < gws.size(); ++i) {
                        internalGlobalWS[i] = ROUND_UP(gws[i], std::max((uint32_t)1, lws[i]));
                    }
                    cl_int res = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueNDRangeKernel(
                                    kernel, cl::NullRange,
                                    cl::NDRange(internalGlobalWS[0], internalGlobalWS[1]),
                                    cl::NDRange(lws[0], lws[1]),
                                    nullptr, &event);
                    MNN_CHECK_CL_SUCCESS(res, kernelName.c_str());

                    int cost_time = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
                    if(cost_time < min_cost) {
                        min_cost = cost_time;
                        lws_prefer[0] = lws[0];
                        lws_prefer[1] = lws[1];
                    }
                }
                do {
                    lws[0]++;
                }
                while(((2*gws[0])%lws[0] > 1) && (lws[0] & (lws[0] - 1)) != 0 && (lws[0] <= gws[0]) && (lws[0] <= 6));//divisible powOfTwo lessThanSix
            }
            do {
                lws[1]++;
            }
            while(((2*gws[1])%lws[1] > 1) && (lws[1] & (lws[1] - 1)) != 0 && (lws[1] <= gws[1]) && (lws[1] <= 6));//divisible powOfTwo lessThanSix
        }
    } else if(runtime->getCLTuneLevel() == None) {
        // define not tune method to choose lws
        if(runtime->getGpuMemType() == GpuMemObject::IMAGE) {
            lws_prefer[0] = 8;
            lws_prefer[1] = 4;
        } else {
            lws_prefer[0] = 0;
            lws_prefer[1] = 0;
        }
        min_cost = 0;
    }
    
    if (tunedLws.find(info) == tunedLws.end()) {
        //printf("ConvBuf2dGeneralLocalWS %d Insert! gws:%d %d, lws:%d %d, time:%dus\n", (int)tunedLws.size(), gws[0], gws[1], lws_prefer[0], lws_prefer[1], min_cost);
        tunedLws.insert(std::make_pair(info, std::make_pair(lws_prefer, min_cost)));
    }
    return std::make_pair(lws_prefer, min_cost);
}

ConvBufCommonExecution::ConvBufCommonExecution(const Convolution2D *conv2dParams, Backend *backend) : Execution(backend) {
    auto openclBackend       = (OpenCLBackend *)backend;
    int biasSize             = conv2dParams->common()->outputCount();
    int buffer_size = ALIGN_UP8(biasSize);//pack to eight
    if(openclBackend->getOpenCLRuntime()->isSupportedFP16()) {
        buffer_size *= sizeof(half_float::half);
    } else {
        buffer_size *= sizeof(float);
    }

    mBias.reset(Tensor::createDevice<float>({1, 1, 1,  ALIGN_UP8(biasSize)}));
    backend->onAcquireBuffer(mBias.get(), Backend::STATIC);
    cl::Buffer &biasBuffer = openCLBuffer(mBias.get());
    
    cl_int res;
    auto biasPtrCL = openclBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(
        biasBuffer, true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &res);
    if(biasPtrCL != nullptr && res == CL_SUCCESS){
        ::memset(biasPtrCL, 0, buffer_size);
        if (nullptr != conv2dParams->bias()) {
            const float *biasDataPtr = conv2dParams->bias()->data();
            if(openclBackend->getOpenCLRuntime()->isSupportedFP16()){
                for(int i=0; i<biasSize; i++) {
                    ((half_float::half*)biasPtrCL)[i] = (half_float::half)(biasDataPtr[i]);
                }
            }else{
                ::memcpy(biasPtrCL, biasDataPtr, biasSize * sizeof(float));
            }
        }
    }else{
        MNN_ERROR("Map error biasPtrCL == nullptr \n");
    }
    openclBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(biasBuffer, biasPtrCL);
}

ConvBufCommonExecution::~ConvBufCommonExecution() {
    MNN_ASSERT(nullptr != mBias);
    backend()->onReleaseBuffer(mBias.get(), Backend::STATIC);
}

void ConvBufExecution::setConv1x1WeightBuffer(int packCout, int packCin, const float* filterDataPtr) {
    cl_int res;
    std::shared_ptr<Tensor> filterBuffer(Tensor::createDevice<float>({ROUND_UP(mOutputChannel, 8)/*Cout pack set to max 8*/, ROUND_UP(mInputChannel, packCin), mKernelWidth, mKernelHeight}));
    
    int buffer_size = filterBuffer->elementSize();
    if(mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16()) {
        buffer_size *= sizeof(half_float::half);
    } else {
        buffer_size *= sizeof(float);
    }
    mKernelBuffer.reset(new cl::Buffer(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size));
    auto kernelBufferPtr = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(*(mKernelBuffer.get()), true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &res);
    if(kernelBufferPtr != nullptr && res == CL_SUCCESS){
        ::memset(kernelBufferPtr, 0, buffer_size);
        for(int o = 0; o < mOutputChannel; o++){
            for(int i = 0 ; i < mInputChannel; i++){
                int bufferIdx = (o/packCout) * ROUND_UP(mInputChannel, packCin)*packCout + (i/packCin)*packCin*packCout + (o%packCout)*packCin + (i%packCin);//(Co/packCout, Ci/packCin, packCout, packCin)
                int filterIdx = o*mInputChannel + i;
                if(mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16()){
                    ((half_float::half*)kernelBufferPtr)[bufferIdx] = (half_float::half)(filterDataPtr[filterIdx]);
                }else{
                    ((float*)kernelBufferPtr)[bufferIdx] = (float)(filterDataPtr[filterIdx]);
                }
            }
        }
    }else{
        MNN_ERROR("Map error ptrCL == nullptr \n");
        MNN_ASSERT(false);
    }
    mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(*(mKernelBuffer.get()), kernelBufferPtr);
}

void ConvBufExecution::_generateFilterConvertRegion(Tensor* virtualFilter, Tensor* originBuffer) const {
    auto filterDes = TensorUtils::getDescribe(virtualFilter);
    filterDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
    filterDes->regions.clear();
    for (int so=0; so<4; ++so) {
        int oSize = (mOutputChannel - so + 3) / 4;
        if (oSize <= 0) {
            continue;
        }
        Tensor::InsideDescribe::Region slice;
        slice.origin = originBuffer;
        slice.size[0] = oSize;
        slice.size[1] = mInputChannel;
        slice.size[2] = mKernelWidth * mKernelHeight;
        slice.src.stride[0] = mInputChannel * mKernelWidth * mKernelHeight * 4;
        slice.src.stride[1] = mKernelWidth * mKernelHeight;
        slice.src.stride[2] = 1;
        slice.src.offset = so * mInputChannel * mKernelWidth * mKernelHeight;
        slice.dst.stride[0] = mKernelWidth * mKernelHeight * 4;
        slice.dst.stride[1] = mKernelWidth * mKernelHeight * UP_DIV(mOutputChannel, 4) * 4;
        slice.dst.stride[2] = 4;
        slice.dst.offset = so;
        filterDes->regions.emplace_back(std::move(slice));
    }
}


ConvBufExecution::ConvBufExecution(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, const MNN::Op *op, Backend *backend)
    : ConvBufCommonExecution(op->main_as_Convolution2D(), backend) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start ConvExecution init !\n");
#endif
    mOpenCLBackend                 = static_cast<OpenCLBackend *>(backend);
    const auto *conv2dParams       = op->main_as_Convolution2D();
    const auto *conv2dCommonParams = conv2dParams->common();
    mConv2dParams                  = conv2dParams;
    mConv2dCommonParams            = conv2dCommonParams;
    mStrides                       = {conv2dCommonParams->strideY(), conv2dCommonParams->strideX()};
    mDilations                     = {conv2dCommonParams->dilateY(), conv2dCommonParams->dilateX()};

    auto padding = ConvolutionCommon::convolutionPad(inputs[0], outputs[0], mConv2dCommonParams);
    mPaddings[0] = padding.second;//padY
    mPaddings[1] = padding.first;//padX
        
    mKernelWidth   = conv2dCommonParams->kernelX();
    mKernelHeight  = conv2dCommonParams->kernelY();
    mOutputChannel = conv2dCommonParams->outputCount();
    std::string kernelName = "conv_2d_c4h1w4";
    mInputChannel = inputs[0]->channel();

    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
    if (inputs.size() != 1) {
        // Multi - Input
        mConv1x1Opt = false;
        std::shared_ptr<Tensor> virtualFilter(
            Tensor::createDevice<float>({ROUND_UP(mOutputChannel, 4) * ROUND_UP(mInputChannel, 4) * mKernelWidth * mKernelHeight}));
        mVirtualFilter = virtualFilter;
        mRasterExe.reset(new RasterBufExecution({virtualFilter.get()}, op, mOpenCLBackend));
    } else {
        int weightSize   = 0;
        ConvolutionCommon::getConvParameters(&quanCommon, conv2dParams, &mFilterDataPtr, &weightSize);
        //select opt conv method
        mConv1x1Opt = (mKernelHeight == mKernelWidth && mKernelHeight == 1 && mPaddings[0] == 0 &&
        mPaddings[1] == 0 && mStrides[0] == 1 && mStrides[1] == 1 && inputs[0]->width() >= 4);
    }
    if (mConv1x1Opt) {
        //At first, set packCout equal to 4
        setConv1x1WeightBuffer(4, 4, mFilterDataPtr);
        kernelName = "conv_2d_1x1_c4h1w4";
    } else {
        mFilter.reset(
            Tensor::createDevice<float>({ROUND_UP(mOutputChannel, 4) * ROUND_UP(mInputChannel, 4) * mKernelWidth * mKernelHeight}));
        if (mFilterDataPtr != nullptr) {
            auto res = mOpenCLBackend->onAcquireBuffer(mFilter.get(), Backend::STATIC);
            if (!res) {
                mValid = false;
                return;
            }
            std::shared_ptr<Tensor> originBuffer(
                Tensor::createDevice<float>({mOutputChannel * mInputChannel * mKernelWidth * mKernelHeight}));
            std::shared_ptr<Tensor> originBufferHost(
                Tensor::create<float>({mOutputChannel * mInputChannel * mKernelWidth * mKernelHeight}, (void*)mFilterDataPtr));
            res = mOpenCLBackend->onAcquireBuffer(originBuffer.get(), Backend::STATIC);
            if (!res) {
                mValid = false;
                return;
            }
            mOpenCLBackend->onCopyBuffer(originBufferHost.get(), originBuffer.get());
            std::shared_ptr<Tensor> virtualFilter(
                Tensor::createDevice<float>({ROUND_UP(mOutputChannel, 4) * ROUND_UP(mInputChannel, 4) * mKernelWidth * mKernelHeight}));
            _generateFilterConvertRegion(virtualFilter.get(), originBuffer.get());
            std::shared_ptr<Execution> raster(new RasterBufExecution({virtualFilter.get()}, op, mOpenCLBackend));
            raster->onResize({virtualFilter.get()}, {mFilter.get()});
            raster->onExecute({virtualFilter.get()}, {mFilter.get()});
            mOpenCLBackend->onReleaseBuffer(originBuffer.get(), Backend::STATIC);
        }
    }
    // Create Kernel
    if (mConv2dCommonParams->relu()) {
        mBuildOptions.emplace("-DRELU");
    } else if (mConv2dCommonParams->relu6()) {
        mBuildOptions.emplace("-DRELU6");
    }
    mBuildOptions.emplace(std::string("-DIN_C_BLOCK=" + std::to_string(UP_DIV(mInputChannel, 4))));

    mKernel           = mOpenCLBackend->getOpenCLRuntime()->buildKernel("conv_2d_buf", kernelName, mBuildOptions);
    mMaxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(mKernel));

#ifdef LOG_VERBOSE
    MNN_PRINT("end ConvExecution init !\n");
#endif
}

ConvBufExecution::~ConvBufExecution() {
    if(!mConv1x1Opt){
        if (mVirtualFilter == nullptr) {
            mOpenCLBackend->onReleaseBuffer(mFilter.get(), Backend::STATIC);
        }
    }
}

ErrorCode ConvBufExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start ConvExecution onResize !\n");
#endif
    auto input  = inputs[0];
    auto output = outputs[0];
    if (inputs.size() > 1) {
        // Multi Input, need pretreat
        _generateFilterConvertRegion(mVirtualFilter.get(), inputs[1]);
        bool res = backend()->onAcquireBuffer(mFilter.get(), Backend::DYNAMIC);
        if (!res) {
            return OUT_OF_MEMORY;
        }
        mRasterExe->onResize({mVirtualFilter.get()}, {mFilter.get()});
    }

    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);
    const int height             = outputShape.at(1);
    const int width              = outputShape.at(2);
    const int outChannel         = outputShape.at(3);

    const int inputHeight   = inputShape.at(1);
    const int inputWidth    = inputShape.at(2);
    const int inputChannels = inputShape.at(3);

    const int inputChannelBlocks = UP_DIV(inputChannels, 4);
    auto padding = ConvolutionCommon::convolutionPad(input, output, mConv2dCommonParams);
    mPaddings[0] = padding.second;//padY
    mPaddings[1] = padding.first;//padX
    
    if (mConv1x1Opt) {

        // {"conv_2d_1x1_c4h1w4", "conv_2d_1x1_c4h1w2", "conv_2d_1x1_c4h1w1", "conv_2d_1x1_c8h1w4"};
        const int total_kernel = 5;
        std::string kernelName[total_kernel] = {"conv_2d_1x1_c4h1w4", "conv_2d_1x1_c4h1w2", "conv_2d_1x1_c4h1w1", "conv_2d_1x1_c8h1w4", "conv_2d_1x1_c8h1w2"};
        int itemC[total_kernel] = {4, 4, 4, 8, 8};
        int itemW[total_kernel] = {4, 2, 1, 4, 2};
        int c8_index_start = 3;
        
        int actual_kernel = total_kernel;
        if(mOpenCLBackend->getOpenCLRuntime()->getCLTuneLevel() == Normal) {
            actual_kernel = 2;
            kernelName[0] = "conv_2d_1x1_c4h1w1";
            itemC[0]      = 4;
            itemW[0]      = 1;

            kernelName[1] = "conv_2d_1x1_c8h1w2";
            itemC[1]      = 8;
            itemW[1]      = 2;
            c8_index_start = 1;
        } else if(mOpenCLBackend->getOpenCLRuntime()->getCLTuneLevel() == Fast || mOpenCLBackend->getOpenCLRuntime()->getCLTuneLevel() == None) {
            actual_kernel = 1;
            
            kernelName[0] = "conv_2d_1x1_c8h1w2";
            itemC[0]      = 8;
            itemW[0]      = 2;
            c8_index_start = 0;
        }
        
        cl::Kernel kernel[total_kernel];
        std::vector<uint32_t> globalWorkSize[total_kernel];
        std::vector<uint32_t> localWorkSize[total_kernel];
        std::pair<int, int> min_cost(INT_MAX, 0);//(min_time, min_index)
        for(int knl_idx = 0; knl_idx < actual_kernel; knl_idx++) {
            kernel[knl_idx]        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("conv_2d_buf", kernelName[knl_idx], mBuildOptions);
            uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(kernel[knl_idx]));
            
            uint32_t idx            = 0;
            
            globalWorkSize[knl_idx] = {static_cast<uint32_t>(UP_DIV(outputShape.at(3), itemC[knl_idx]) * UP_DIV(outputShape.at(2), itemW[knl_idx])), static_cast<uint32_t>(outputShape.at(0) * outputShape.at(1))};
            
            kernel[knl_idx].setArg(idx++, globalWorkSize[knl_idx][0]);
            kernel[knl_idx].setArg(idx++, globalWorkSize[knl_idx][1]);
            kernel[knl_idx].setArg(idx++, UP_DIV(width, itemW[knl_idx]));
            kernel[knl_idx].setArg(idx++, openCLBuffer(input));
            kernel[knl_idx].setArg(idx++, *mKernelBuffer.get());
            kernel[knl_idx].setArg(idx++, openCLBuffer(mBias.get()));
            kernel[knl_idx].setArg(idx++, openCLBuffer(output));
            kernel[knl_idx].setArg(idx++, static_cast<int>(inputChannelBlocks));
            kernel[knl_idx].setArg(idx++, height);
            kernel[knl_idx].setArg(idx++, width);
            kernel[knl_idx].setArg(idx++, UP_DIV(outChannel, 4));
            
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
        if(min_index >= c8_index_start) {//if best kernel is "conv_2d_1x1_c8h1w4", set weight packCout to 8
            int weightSize   = 0;
            ConvolutionCommon::getConvParameters(&quanCommon, mConv2dParams, &mFilterDataPtr, &weightSize);
            setConv1x1WeightBuffer(8, 4, mFilterDataPtr);
        }
        mGlobalWorkSize = {globalWorkSize[min_index][0], globalWorkSize[min_index][1]};
        
        mKernel        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("conv_2d_buf", kernelName[min_index], mBuildOptions);
        uint32_t idx            = 0;
        mKernel.setArg(idx++, mGlobalWorkSize[0]);
        mKernel.setArg(idx++, mGlobalWorkSize[1]);
        mKernel.setArg(idx++, UP_DIV(width, itemW[min_index]));
        mKernel.setArg(idx++, openCLBuffer(input));
        mKernel.setArg(idx++, *mKernelBuffer.get());
        mKernel.setArg(idx++, openCLBuffer(mBias.get()));
        mKernel.setArg(idx++, openCLBuffer(output));
        mKernel.setArg(idx++, static_cast<int>(inputChannelBlocks));
        mKernel.setArg(idx++, height);
        mKernel.setArg(idx++, width);
        mKernel.setArg(idx++, UP_DIV(outChannel, 4));
        
        //printf("conv1x1 %d, %d %d, %d %d, %d %d\n", min_index, mGlobalWorkSize[0], mGlobalWorkSize[1], mLocalWorkSize[0], mLocalWorkSize[1], outChannel, width);
    } else {
        int inputImageShape[2]  = {inputHeight, inputWidth};
        int outputImageShape[2] = {height, width};
        int kernelShape[2]      = {mKernelHeight, mKernelWidth};
        int strideShape[2]      = {mStrides[0], mStrides[1]};
        int paddingShape[2]     = {mPaddings[0], mPaddings[1]};
        int dilationShape[2]    = {mDilations[0], mDilations[1]};
        
        // {"conv_2d_c4h1w4", "conv_2d_c4h1w2", "conv_2d_c4h1w1", "conv_2d_c8h1w1", };
        const int total_kernel = 4;
        std::string kernelName[total_kernel] = {"conv_2d_c4h1w4", "conv_2d_c4h1w2", "conv_2d_c4h1w1", "conv_2d_c8h1w1"};
        int itemC[total_kernel] = {4, 4, 4, 8};
        int itemW[total_kernel] = {4, 2, 1, 1};
        
        
        int actual_kernel = total_kernel;
        if(mOpenCLBackend->getOpenCLRuntime()->getCLTuneLevel() == Normal) {
            actual_kernel = 2;
            kernelName[0] = "conv_2d_c4h1w4";
            itemC[0]      = 4;
            itemW[0]      = 4;

            kernelName[1] = "conv_2d_c4h1w2";
            itemC[1]      = 4;
            itemW[1]      = 2;
        } else if(mOpenCLBackend->getOpenCLRuntime()->getCLTuneLevel() == Fast || mOpenCLBackend->getOpenCLRuntime()->getCLTuneLevel() == None) {
            actual_kernel = 1;
            
            kernelName[0] = "conv_2d_c4h1w4";
            itemC[0]      = 4;
            itemW[0]      = 4;
        }

        cl::Kernel kernel[total_kernel];
        std::vector<uint32_t> globalWorkSize[total_kernel];
        std::vector<uint32_t> localWorkSize[total_kernel];
        std::pair<int, int> min_cost(INT_MAX, 0);//(min_time, min_index)
        for(int knl_idx = 0; knl_idx < actual_kernel; knl_idx++) {
            kernel[knl_idx]        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("conv_2d_buf", kernelName[knl_idx], mBuildOptions);
            uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(kernel[knl_idx]));
            
            globalWorkSize[knl_idx] = {static_cast<uint32_t>(UP_DIV(outputShape.at(3), itemC[knl_idx]) * UP_DIV(outputShape.at(2), itemW[knl_idx])), static_cast<uint32_t>(outputShape.at(0) * outputShape.at(1))};
            uint32_t idx            = 0;
            kernel[knl_idx].setArg(idx++, globalWorkSize[knl_idx][0]);
            kernel[knl_idx].setArg(idx++, globalWorkSize[knl_idx][1]);
            kernel[knl_idx].setArg(idx++, openCLBuffer(input));
            kernel[knl_idx].setArg(idx++, openCLBuffer(mFilter.get()));
            kernel[knl_idx].setArg(idx++, openCLBuffer(mBias.get()));
            kernel[knl_idx].setArg(idx++, openCLBuffer(output));
            kernel[knl_idx].setArg(idx++, sizeof(inputImageShape), inputImageShape);
            kernel[knl_idx].setArg(idx++, inputChannels);
            kernel[knl_idx].setArg(idx++, inputChannelBlocks);
            kernel[knl_idx].setArg(idx++, sizeof(outputImageShape), outputImageShape);
            kernel[knl_idx].setArg(idx++, sizeof(kernelShape), kernelShape);
            kernel[knl_idx].setArg(idx++, sizeof(strideShape), strideShape);
            kernel[knl_idx].setArg(idx++, sizeof(paddingShape), paddingShape);
            kernel[knl_idx].setArg(idx++, sizeof(dilationShape), dilationShape);
            kernel[knl_idx].setArg(idx++, UP_DIV(width, itemW[knl_idx]));
            kernel[knl_idx].setArg(idx++, UP_DIV(outChannel, 4));
            
            std::pair<std::vector<uint32_t>, int> retTune;
            retTune = gws2dLwsTune(kernel[knl_idx], globalWorkSize[knl_idx], kernelName[knl_idx], maxWorkGroupSize);
            //printf("conv %d, %d\n", knl_idx, retTune.second);
            if(min_cost.first > retTune.second) {
                min_cost.first = retTune.second;
                min_cost.second = knl_idx;
                mLocalWorkSize = {retTune.first[0], retTune.first[1]};
            }
        }
        int min_index  = min_cost.second;
        mGlobalWorkSize = {globalWorkSize[min_index][0], globalWorkSize[min_index][1]};
        
        mKernel        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("conv_2d_buf", kernelName[min_index], mBuildOptions);
        
        uint32_t idx            = 0;
        mKernel.setArg(idx++, mGlobalWorkSize[0]);
        mKernel.setArg(idx++, mGlobalWorkSize[1]);
        mKernel.setArg(idx++, openCLBuffer(input));
        mKernel.setArg(idx++, openCLBuffer(mFilter.get()));
        mKernel.setArg(idx++, openCLBuffer(mBias.get()));
        mKernel.setArg(idx++, openCLBuffer(output));
        mKernel.setArg(idx++, sizeof(inputImageShape), inputImageShape);
        mKernel.setArg(idx++, inputChannels);
        mKernel.setArg(idx++, inputChannelBlocks);
        mKernel.setArg(idx++, sizeof(outputImageShape), outputImageShape);
        mKernel.setArg(idx++, sizeof(kernelShape), kernelShape);
        mKernel.setArg(idx++, sizeof(strideShape), strideShape);
        mKernel.setArg(idx++, sizeof(paddingShape), paddingShape);
        mKernel.setArg(idx++, sizeof(dilationShape), dilationShape);
        mKernel.setArg(idx++, UP_DIV(width, itemW[min_index]));
        mKernel.setArg(idx++, UP_DIV(outChannel, 4));
        
        //printf("conv:%d pad:%d filter: %d, chw:%d %d %d, %d %d %d, gws:%d %d\n", min_index, mPaddings[0],  mKernelHeight, inputs[0]->channel(), inputs[0]->height(), inputs[0]->width(), outputs[0]->channel(), outputs[0]->height(), outputs[0]->width(), mGlobalWorkSize[0], mGlobalWorkSize[1]);
    }
    if (inputs.size() > 1) {
        backend()->onReleaseBuffer(mFilter.get(), Backend::DYNAMIC);
    }
#ifdef LOG_VERBOSE
    MNN_PRINT("end ConvExecution onResize !\n");
#endif
    return NO_ERROR;
}

ErrorCode ConvBufExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start ConvExecution onExecute !\n");
#endif
    if (inputs.size() > 1) {
        mRasterExe->onExecute({mVirtualFilter.get()}, {mFilter.get()});
    }

#ifdef ENABLE_OPENCL_TIME_PROFILER
    cl::Event event;
    runKernel2D(mKernel, mGlobalWorkSize, mLocalWorkSize,
                mOpenCLBackend->getOpenCLRuntime(), &event);
    
    int costTime = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
    MNN_PRINT("kernel cost:%d    us ConvBuf2D\n",costTime);
#else
    runKernel2D(mKernel, mGlobalWorkSize, mLocalWorkSize,
                mOpenCLBackend->getOpenCLRuntime());
#endif
    
#ifdef LOG_VERBOSE
    MNN_PRINT("end ConvExecution onExecute !\n");
#endif
    return NO_ERROR;
}

class ConvolutionBufCreator : public OpenCLBackend::Creator {
public:
    virtual ~ConvolutionBufCreator() = default;
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        if (nullptr != op->main_as_Convolution2D()->quanParameter()) {
            auto quan = op->main_as_Convolution2D()->quanParameter();
            if (1 == quan->type() || 2 == quan->type()) {
                if (quan->has_scaleInt()) {
                    // Don't support IDST-int8 because of error
                    return nullptr;
                }
            }
        }
        
        if (inputs.size() == 3) {
            MNN_PRINT("multi input conv for opencl buffer not supoort!\n");
            return nullptr;
        }
        if (inputs.size() > 1) {
            // Multi inputs
            return new ConvBufExecution(inputs, outputs, op, backend);
        }
        auto conv2D = op->main_as_Convolution2D();
        if (ConvBufWinograd::valid(conv2D->common(), inputs[0])) {
            return new ConvBufWinograd(conv2D, backend);
        }
        return new ConvBufExecution(inputs, outputs, op, backend);
    }
};

OpenCLCreatorRegister<ConvolutionBufCreator> __convBuf_op(OpType_Convolution, BUFFER);

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
