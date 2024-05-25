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
#include "ConvSubgroupBufExecution.hpp"
#include "core/ConvolutionCommon.hpp"
#include "core/Backend.hpp"
#include "RasterBufExecution.hpp"
#include "ConvBufLowMemoryExecution.hpp"

namespace MNN {
namespace OpenCL {
std::pair<std::vector<uint32_t>,  uint32_t> ConvBufCommonExecution::gws2dLwsTune(const std::shared_ptr<KernelWrap> &kernelW, const std::vector<uint32_t> &gws, const std::string &kernelName, const uint32_t maxWorkGroupSize) {
    MNN_ASSERT(gws.size() == 2);

    auto kernel = kernelW->get();
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    auto maxWorkItemSizes = runtime->getMaxWorkItemSizes();
    MNN_ASSERT(maxWorkItemSizes.size() >= 2);
    
    auto& tunedLws = runtime->tunedLwsMap();
    auto& tuneLws = runtime->getTuneLwsMap();
    std::pair<std::string, std::vector<uint32_t>> info = std::make_pair(kernelName, gws);
    if (tunedLws.find(info) != tunedLws.end()) {
        //printf("ConvBuf2dGeneralLocalWS Found! gws:%d %d lws:%d %d\n", gws[0], gws[1], tunedLws[info][0], tunedLws[info][1]);
        return tunedLws[info];
    }
    std::pair<std::vector<uint32_t>, uint32_t> tuneLwsRes;
    if(localWSTune(tuneLws, gws, kernelName, tuneLwsRes)){
        return tuneLwsRes;
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
                lws[0]<<=1;
            }
            lws[1]<<=1;
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
                    lws[0]<<=1;
                }
                while(((2*gws[0])%lws[0] > 1) && (lws[0] & (lws[0] - 1)) != 0 && (lws[0] <= gws[0]) && (lws[0] > 6));//divisible powOfTwo lessThanSix
            }
            do {
                lws[1]<<=1;
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
                    lws[0]<<=1;
                }
                while(((2*gws[0])%lws[0] > 1) && (lws[0] & (lws[0] - 1)) != 0 && (lws[0] <= gws[0]) && (lws[0] > 6));//divisible powOfTwo lessThanSix
            }
            do {
                lws[1]<<=1;
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
                    lws[0]<<=1;
                }
                while(((2*gws[0])%lws[0] > 1) && (lws[0] & (lws[0] - 1)) != 0 && (lws[0] <= gws[0]) && (lws[0] <= 6));//divisible powOfTwo lessThanSix
            }
            do {
                lws[1]<<=1;
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
        cl::Event event;
        std::vector<uint32_t> internalGlobalWS(2, 1);
        for (size_t i = 0; i < gws.size(); ++i) {
            internalGlobalWS[i] = ROUND_UP(gws[i], std::max((uint32_t)1, lws_prefer[i]));
        }
        cl_int res = CL_SUCCESS;
        if(lws_prefer[0] == 0 || lws_prefer[1] == 0){
            res = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueNDRangeKernel(
                    kernel, cl::NullRange, cl::NDRange(internalGlobalWS[0], internalGlobalWS[1]), cl::NullRange, nullptr, &event);
        }else{
            res = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueNDRangeKernel(
                    kernel, cl::NullRange, cl::NDRange(internalGlobalWS[0], internalGlobalWS[1]), cl::NDRange(lws_prefer[0], lws_prefer[1]), nullptr, &event);
        }
        MNN_CHECK_CL_SUCCESS(res, kernelName.c_str());
        min_cost = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
    }
    
    if (tunedLws.find(info) == tunedLws.end() && runtime->getCLTuneLevel() != None) {
        //printf("ConvBuf2dGeneralLocalWS %d Insert! gws:%d %d, lws:%d %d, time:%dus\n", (int)tunedLws.size(), gws[0], gws[1], lws_prefer[0], lws_prefer[1], min_cost);
        tunedLws.insert(std::make_pair(info, std::make_pair(lws_prefer, min_cost)));
    }
    return std::make_pair(lws_prefer, min_cost);
}
ConvBufCommonExecution::ConvBufCommonExecution(Backend *backend) {
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
}
ConvBufCommonExecution::ConvBufCommonExecution(const Convolution2D *conv2dParams, Backend *backend) {
    auto openclBackend       = (OpenCLBackend *)backend;
    int biasSize             = conv2dParams->common()->outputCount();
    int buffer_size = ROUND_UP(biasSize, 16);//pack to 16
    if(openclBackend->getOpenCLRuntime()->isSupportedFP16()) {
        buffer_size *= sizeof(half_float::half);
    } else {
        buffer_size *= sizeof(float);
    }

    mResource.reset(new ConvBufResource);
    mResource->mBias.reset(Tensor::createDevice<float>({1, 1, 1, ROUND_UP(biasSize, 16)}));
    backend->onAcquireBuffer(mResource->mBias.get(), Backend::STATIC);
    cl::Buffer &biasBuffer = openCLBuffer(mResource->mBias.get());
    
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
    // Do nothing
}

void ConvBufExecution::setConv1x1WeightBuffer(int packCout, int packCin, const float* filterDataPtr) {
    cl_int res;
    std::shared_ptr<Tensor> filterBuffer(Tensor::createDevice<float>({ROUND_UP(mResource->mOutputChannel, 8)/*Cout pack set to max 8*/, ROUND_UP(mResource->mInputChannel, packCin), mResource->mKernelWidth, mResource->mKernelHeight}));
    
    int buffer_size = filterBuffer->elementSize();
    if(mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16()) {
        buffer_size *= sizeof(half_float::half);
    } else {
        buffer_size *= sizeof(float);
    }
    mResource->mKernelBuffer.reset(new cl::Buffer(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size));
    auto kernelBufferPtr = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(*(mResource->mKernelBuffer.get()), true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &res);
    if(kernelBufferPtr != nullptr && res == CL_SUCCESS){
        ::memset(kernelBufferPtr, 0, buffer_size);
        for(int o = 0; o < mResource->mOutputChannel; o++){
            for(int i = 0 ; i < mResource->mInputChannel; i++){
                int bufferIdx = (o/packCout) * ROUND_UP(mResource->mInputChannel, packCin)*packCout + (i/packCin)*packCin*packCout + (o%packCout)*packCin + (i%packCin);//(Co/packCout, Ci/packCin, packCout, packCin)
                int filterIdx = o*mResource->mInputChannel + i;
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
    mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(*(mResource->mKernelBuffer.get()), kernelBufferPtr);
}

void ConvBufExecution::_generateFilterConvertRegion(Tensor* virtualFilter, Tensor* originBuffer) const {
    auto filterDes = TensorUtils::getDescribe(virtualFilter);
    filterDes->regions.clear();
    for (int so=0; so<4; ++so) {
        int oSize = (mResource->mOutputChannel - so + 3) / 4;
        if (oSize <= 0) {
            continue;
        }
        Tensor::InsideDescribe::Region slice;
        slice.origin = originBuffer;
        slice.size[0] = oSize;
        slice.size[1] = mResource->mInputChannel;
        slice.size[2] = mResource->mKernelWidth * mResource->mKernelHeight;
        slice.src.stride[0] = mResource->mInputChannel * mResource->mKernelWidth * mResource->mKernelHeight * 4;
        slice.src.stride[1] = mResource->mKernelWidth * mResource->mKernelHeight;
        slice.src.stride[2] = 1;
        slice.src.offset = so * mResource->mInputChannel * mResource->mKernelWidth * mResource->mKernelHeight;
        slice.dst.stride[0] = mResource->mKernelWidth * mResource->mKernelHeight * 4;
        slice.dst.stride[1] = mResource->mKernelWidth * mResource->mKernelHeight * UP_DIV(mResource->mOutputChannel, 4) * 4;
        slice.dst.stride[2] = 4;
        slice.dst.offset = so;
        filterDes->regions.emplace_back(std::move(slice));
    }
}

ConvBufExecution::ConvBufExecution(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, const MNN::Op *op, Backend *backend)
    : ConvBufCommonExecution(op->main_as_Convolution2D(), backend), CommonExecution(backend, op) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start ConvExecution init !\n");
#endif
    mOpenCLBackend                 = static_cast<OpenCLBackend *>(backend);
    const auto *conv2dParams       = op->main_as_Convolution2D();
    const auto *conv2dCommonParams = conv2dParams->common();
    mResource->mConv2dParams                  = conv2dParams;
    mResource->mConv2dCommonParams            = conv2dCommonParams;
    mResource->mStrides                       = {conv2dCommonParams->strideY(), conv2dCommonParams->strideX()};
    mResource->mDilations                     = {conv2dCommonParams->dilateY(), conv2dCommonParams->dilateX()};

    auto padding = ConvolutionCommon::convolutionPad(inputs[0], outputs[0], mResource->mConv2dCommonParams);
    mPaddings[0] = padding.second;//padY
    mPaddings[1] = padding.first;//padX
        
    mResource->mKernelWidth   = conv2dCommonParams->kernelX();
    mResource->mKernelHeight  = conv2dCommonParams->kernelY();
    mResource->mOutputChannel = conv2dCommonParams->outputCount();
    mResource->mInputChannel = inputs[0]->channel();

    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
    if (inputs.size() != 1) {
        // Multi - Input
        mResource->mConv1x1Opt = false;
        mResource->mRasterExe.reset(new RasterBufExecution({mResource->mFilter.get()}, op, mOpenCLBackend));
    } else {
        int weightSize   = 0;
        ConvolutionCommon::getConvParameters(&quanCommon, backend, conv2dParams, &mFilterDataPtr, &weightSize);
        //select opt conv method
        mResource->mConv1x1Opt = (mResource->mKernelHeight == mResource->mKernelWidth && mResource->mKernelHeight == 1 && mPaddings[0] == 0 &&
        mPaddings[1] == 0 && mResource->mStrides[0] == 1 && mResource->mStrides[1] == 1 && inputs[0]->width() >= 4);
    }
    if (mResource->mConv1x1Opt) {
        //At first, set packCout equal to 4
        if(mResource->mOutputChannel >= 16){
            setConv1x1WeightBuffer(8, 4, mFilterDataPtr);
            mResource->mConv1x1C8Opt = true;
        }else{
            setConv1x1WeightBuffer(4, 4, mFilterDataPtr);
        }
    } else {
        mResource->mFilter.reset(
            Tensor::createDevice<float>({ROUND_UP(mResource->mOutputChannel, 4) * ROUND_UP(mResource->mInputChannel, 4) * mResource->mKernelWidth * mResource->mKernelHeight}));
        if (mFilterDataPtr != nullptr) {
            std::vector<int> filterImageShape{ROUND_UP(mResource->mInputChannel, 4), (UP_DIV(mResource->mOutputChannel, 4) * mResource->mKernelWidth * mResource->mKernelHeight)};
            std::shared_ptr<Tensor> filterBuffer(
                Tensor::createDevice<float>({mResource->mOutputChannel, ROUND_UP(mResource->mInputChannel, 4), mResource->mKernelWidth, mResource->mKernelHeight}));
            
            int buffer_size = filterBuffer->elementSize() * sizeof(float);
            cl::Buffer filterBufferCL(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size);
            filterBuffer->buffer().device = (uint64_t)(&filterBufferCL);

            cl_int res;
            auto ptrCL = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(filterBufferCL, true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &res);
            if(ptrCL != nullptr && res == CL_SUCCESS) {
                ::memset(ptrCL, 0, buffer_size);
                const int copy_size = mResource->mKernelWidth * mResource->mKernelHeight * sizeof(float);
                for(int oc=0; oc<mResource->mOutputChannel; oc++) {
                    for(int ic=0; ic<mResource->mInputChannel; ic++) {
                        ::memcpy((float *)ptrCL + (oc * ROUND_UP(mResource->mInputChannel, 4) + ic) * mResource->mKernelWidth * mResource->mKernelHeight, mFilterDataPtr + (oc * mResource->mInputChannel + ic) * mResource->mKernelWidth * mResource->mKernelHeight, copy_size);
                    }
                }
            }else{
                MNN_ERROR("Map error ptrCL == nullptr \n");
            }
            mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(filterBufferCL, ptrCL);

            mResource->mFilter.reset(Tensor::createDevice<float>({1, filterImageShape[1], 1, 4 * filterImageShape[0]}));
            mOpenCLBackend->onAcquireBuffer(mResource->mFilter.get(), Backend::STATIC);
            MNN::OpenCL::BufferConvertor bufferConvertor{mOpenCLBackend->getOpenCLRuntime()};
            
            bool needTrans = true;
            bufferConvertor.convertToNC4HW4Buffer(filterBuffer.get(), MNN::OpenCL::CONV2D_FILTER, mResource->mFilter.get(), needTrans);
        }
    }
        
    if (mResource->mConv2dCommonParams->relu()) {
        mResource->mBuildOptions.emplace("-DRELU");
    } else if (mResource->mConv2dCommonParams->relu6()) {
        mResource->mBuildOptions.emplace("-DRELU6");
    }

#ifdef LOG_VERBOSE
    MNN_PRINT("end ConvExecution init !\n");
#endif
}

ConvBufExecution::~ConvBufExecution() {
    // Do nothing
}

ConvBufExecution::ConvBufExecution(std::shared_ptr<ConvBufResource> resource, const MNN::Op* op, Backend *backend)
    : ConvBufCommonExecution(backend), CommonExecution(backend, op) {
    mResource = resource;
    const auto *conv2dParams       = op->main_as_Convolution2D();
    const auto *conv2dCommonParams = conv2dParams->common();
    mResource->mConv2dParams       = conv2dParams;
    mResource->mConv2dCommonParams  = conv2dCommonParams;
}

bool ConvBufExecution::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    *dst = new ConvBufExecution(mResource, op, bn);
    return true;
}

ErrorCode ConvBufExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start ConvExecution onResize !\n");
#endif
    auto input  = inputs[0];
    auto output = outputs[0];
    if (inputs.size() > 1) {
        // Multi Input, need pretreat
        _generateFilterConvertRegion(mResource->mFilter.get(), inputs[1]);
        bool res = backend()->onAcquireBuffer(mResource->mFilter.get(), Backend::DYNAMIC);
        if (!res) {
            return OUT_OF_MEMORY;
        }
        mResource->mRasterExe->onResize({}, {mResource->mFilter.get()});
    }
    mOpenCLBackend->startRecord(mRecording);
    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);
    const int height             = outputShape.at(1);
    const int width              = outputShape.at(2);
    const int outChannel         = outputShape.at(3);

    const int inputHeight   = inputShape.at(1);
    const int inputWidth    = inputShape.at(2);
    const int inputChannels = inputShape.at(3);

    const int inputChannelBlocks = UP_DIV(inputChannels, 4);
    auto padding = ConvolutionCommon::convolutionPad(input, output, mResource->mConv2dCommonParams);
    mPaddings[0] = padding.second;//padY
    mPaddings[1] = padding.first;//padX
    
    std::string info = std::to_string(inputChannels) + "_" + std::to_string(outChannel) + "_" + std::to_string(mResource->mKernelHeight) + "_" + std::to_string(mResource->mKernelWidth) + "_" + std::to_string(mResource->mStrides[0]) + "_" + std::to_string(mResource->mStrides[1]) + "_" + std::to_string(mResource->mDilations[0]) + "_" + std::to_string(mResource->mDilations[1]);
    if (mResource->mConv1x1Opt) {
    
        // {"conv_2d_1x1_c4h1w4", "conv_2d_1x1_c4h1w2", "conv_2d_1x1_c4h1w1", "conv_2d_1x1_c8h1w4"};
        const int total_kernel = 3;
        std::string kernelName[total_kernel] = {"conv_2d_1x1_c4h1w4", "conv_2d_1x1_c4h1w2", "conv_2d_1x1_c4h1w1"};
        int itemC[total_kernel] = {4, 4, 4};
        int itemW[total_kernel] = {4, 2, 1};
        
        int actual_kernel = total_kernel;
        if(mResource->mConv1x1C8Opt) {
            actual_kernel = 2;
            kernelName[0] = "conv_2d_1x1_c8h1w4";
            itemC[0]      = 8;
            itemW[0]      = 4;

            kernelName[1] = "conv_2d_1x1_c8h1w2";
            itemC[1]      = 8;
            itemW[1]      = 2;
        }
        
        std::shared_ptr<KernelWrap> kernel[total_kernel];
        std::vector<uint32_t> globalWorkSize[total_kernel];
        std::vector<uint32_t> localWorkSize[total_kernel];
        std::pair<int, int> min_cost(INT_MAX, 0);//(min_time, min_index)
        for(int knl_idx = 0; knl_idx < actual_kernel; knl_idx++) {
            std::set<std::string> buildOption = mResource->mBuildOptions;
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
            
            ret |= kernel[knl_idx]->get().setArg(idx++, globalWorkSize[knl_idx][0]);
            ret |= kernel[knl_idx]->get().setArg(idx++, globalWorkSize[knl_idx][1]);
            ret |= kernel[knl_idx]->get().setArg(idx++, UP_DIV(width, itemW[knl_idx]));
            ret |= kernel[knl_idx]->get().setArg(idx++, openCLBuffer(input));
            ret |= kernel[knl_idx]->get().setArg(idx++, *mResource->mKernelBuffer.get());
            ret |= kernel[knl_idx]->get().setArg(idx++, openCLBuffer(mResource->mBias.get()));
            ret |= kernel[knl_idx]->get().setArg(idx++, openCLBuffer(output));
            ret |= kernel[knl_idx]->get().setArg(idx++, static_cast<int>(inputChannelBlocks));
            ret |= kernel[knl_idx]->get().setArg(idx++, height);
            ret |= kernel[knl_idx]->get().setArg(idx++, width);
            ret |= kernel[knl_idx]->get().setArg(idx++, UP_DIV(outChannel, 4));
            MNN_CHECK_CL_SUCCESS(ret, "setArg Conv1x1Buf Kernel Select");

            std::pair<std::vector<uint32_t>, int> retTune;
            retTune = gws2dLwsTune(kernel[knl_idx], globalWorkSize[knl_idx], kernelName[knl_idx] + info, maxWorkGroupSize);
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
        
        std::set<std::string> buildOption = mResource->mBuildOptions;
        if(outputShape.at(3) % itemC[min_index] != 0){
            buildOption.emplace("-DCHANNEL_LEAVE");
        }
        if((outputShape.at(2) % itemW[min_index]) != 0){
            buildOption.emplace("-DBLOCK_LEAVE");
        }
        mKernel        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("conv_2d_buf", kernelName[min_index], buildOption);
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;

        ret |= mKernel->get().setArg(idx++, mGlobalWorkSize[0]);
        ret |= mKernel->get().setArg(idx++, mGlobalWorkSize[1]);
        ret |= mKernel->get().setArg(idx++, UP_DIV(width, itemW[min_index]));
        ret |= mKernel->get().setArg(idx++, openCLBuffer(input));
        ret |= mKernel->get().setArg(idx++, *mResource->mKernelBuffer.get());
        ret |= mKernel->get().setArg(idx++, openCLBuffer(mResource->mBias.get()));
        ret |= mKernel->get().setArg(idx++, openCLBuffer(output));
        ret |= mKernel->get().setArg(idx++, static_cast<int>(inputChannelBlocks));
        ret |= mKernel->get().setArg(idx++, height);
        ret |= mKernel->get().setArg(idx++, width);
        ret |= mKernel->get().setArg(idx++, UP_DIV(outChannel, 4));
        MNN_CHECK_CL_SUCCESS(ret, "setArg Conv1x1Buf");

        //printf("conv1x1 %d, %d %d, %d %d, %d %d\n", min_index, mGlobalWorkSize[0], mGlobalWorkSize[1], mLocalWorkSize[0], mLocalWorkSize[1], outChannel, width);
    } else {
        int inputImageShape[2]  = {inputHeight, inputWidth};
        int outputImageShape[2] = {height, width};
        int kernelShape[2]      = {mResource->mKernelHeight, mResource->mKernelWidth};
        int strideShape[2]      = {mResource->mStrides[0],mResource->mStrides[1]};
        int paddingShape[2]     = {mPaddings[0], mPaddings[1]};
        int dilationShape[2]    = {mResource->mDilations[0], mResource->mDilations[1]};
        
        // {"conv_2d_c4h1w2", "conv_2d_c4h1w1", "conv_2d_c8h1w1", "conv_2d_c4h1w4", "conv_2d_c8h2w1", "conv_2d_c4h4w1"};
        const int total_kernel = 7;
        std::string kernelName[total_kernel] = {"conv_2d_c4h1w1", "conv_2d_c4h1w2", "conv_2d_c4h4w1", "conv_2d_c8h2w1", "conv_2d_c8h4w1", "conv_2d_c4h1w4", "conv_2d_c8h1w4"};
        int itemC[total_kernel] = {4, 4, 4, 8, 8, 4, 8};
        int itemH[total_kernel] = {1, 1, 4, 2, 4, 1, 1};
        int itemW[total_kernel] = {1, 2, 1, 1, 1, 4, 4};
        
        
        int actual_kernel = total_kernel;
        
        
        std::shared_ptr<KernelWrap> kernel[total_kernel];
        std::vector<uint32_t> globalWorkSize[total_kernel];
        std::vector<uint32_t> localWorkSize[total_kernel];
        std::pair<int, int> min_cost(INT_MAX, 0);//(min_time, min_index)
        for(int knl_idx = 0; knl_idx < actual_kernel; knl_idx++) {
            std::set<std::string> buildOption = mResource->mBuildOptions;
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
            ret |= kernel[knl_idx]->get().setArg(idx++, globalWorkSize[knl_idx][0]);
            ret |= kernel[knl_idx]->get().setArg(idx++, globalWorkSize[knl_idx][1]);
            ret |= kernel[knl_idx]->get().setArg(idx++, openCLBuffer(input));
            ret |= kernel[knl_idx]->get().setArg(idx++, openCLBuffer(mResource->mFilter.get()));
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
            MNN_CHECK_CL_SUCCESS(ret, "setArg ConvBuf Kernel Select");

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
        mKernel        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("conv_2d_buf", kernelName[min_index], buildOption);
        
        uint32_t idx            = 0;
        cl_int ret = CL_SUCCESS;

        ret |= mKernel->get().setArg(idx++, mGlobalWorkSize[0]);
        ret |= mKernel->get().setArg(idx++, mGlobalWorkSize[1]);
        ret |= mKernel->get().setArg(idx++, openCLBuffer(input));
        ret |= mKernel->get().setArg(idx++, openCLBuffer(mResource->mFilter.get()));
        ret |= mKernel->get().setArg(idx++, openCLBuffer(mResource->mBias.get()));
        ret |= mKernel->get().setArg(idx++, openCLBuffer(output));
        ret |= mKernel->get().setArg(idx++, sizeof(inputImageShape), inputImageShape);
        ret |= mKernel->get().setArg(idx++, inputChannels);
        ret |= mKernel->get().setArg(idx++, inputChannelBlocks);
        ret |= mKernel->get().setArg(idx++, sizeof(outputImageShape), outputImageShape);
        ret |= mKernel->get().setArg(idx++, sizeof(kernelShape), kernelShape);
        ret |= mKernel->get().setArg(idx++, sizeof(strideShape), strideShape);
        ret |= mKernel->get().setArg(idx++, sizeof(paddingShape), paddingShape);
        ret |= mKernel->get().setArg(idx++, sizeof(dilationShape), dilationShape);
        ret |= mKernel->get().setArg(idx++, UP_DIV(width, itemW[min_index]));
        ret |= mKernel->get().setArg(idx++, UP_DIV(outChannel, 4));
        ret |= mKernel->get().setArg(idx++, UP_DIV(height, itemH[min_index]));
        MNN_CHECK_CL_SUCCESS(ret, "setArg ConvBuf");
    }
    if (inputs.size() > 1) {
        backend()->onReleaseBuffer(mResource->mFilter.get(), Backend::DYNAMIC);
    }
    mOpenCLBackend->recordKernel2d(mKernel, mGlobalWorkSize, mLocalWorkSize);
    mGlobalWorkSize[0] = ROUND_UP(mGlobalWorkSize[0], std::max((uint32_t)1, mLocalWorkSize[0]));
    mGlobalWorkSize[1] = ROUND_UP(mGlobalWorkSize[1], std::max((uint32_t)1, mLocalWorkSize[1]));
    mOpenCLBackend->endRecord(mRecording);
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
        mResource->mRasterExe->onExecute({}, {mResource->mFilter.get()});
        if (inputs.size() > 2) {
            auto buffer_size = inputs[2]->elementSize();
            if(mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16()) {
                buffer_size *= sizeof(half_float::half);
            } else {
                buffer_size *= sizeof(float);
            }
            mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueCopyBuffer(openCLBuffer(inputs[2]), openCLBuffer(mResource->mBias.get()), 0, 0, buffer_size);
        }
    }
#ifdef ENABLE_OPENCL_TIME_PROFILER
    cl::Event event;
    runKernel2D(mKernel, mGlobalWorkSize, mLocalWorkSize, mOpenCLBackend->getOpenCLRuntime(), &event);
    mOpenCLBackend->getOpenCLRuntime()->pushEvent({"ConvBuf2D", event});
#else
    if(mOpenCLBackend->isUseRecordQueue()){
        mOpenCLBackend->addRecord(mRecording, mOpRecordUpdateInfo);
#ifdef LOG_VERBOSE
        MNN_PRINT("End ConvExecution onExecute... \n");
#endif
        return NO_ERROR;
    }
    runKernel2D(mKernel, mGlobalWorkSize, mLocalWorkSize, mOpenCLBackend->getOpenCLRuntime());
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
        auto conv2D  = op->main_as_Convolution2D();
        auto input   = inputs[0];
        auto output  = outputs[0];
        auto padding = ConvolutionCommon::convolutionPad(inputs[0], outputs[0], conv2D->common());
        std::vector<int> inputShape  = tensorShapeFormat(input);
        std::vector<int> outputShape = tensorShapeFormat(output);
        const int outputChannel         = outputShape.at(3);
        const int inputChannels = inputShape.at(3);
#ifdef MNN_LOW_MEMORY
        {
            auto conv2dParams = op->main_as_Convolution2D();
            if (conv2dParams->quanParameter() != nullptr) {
                if (((conv2dParams->quanParameter()->type() == 4) ||
                     (conv2dParams->quanParameter()->type() == 1) ||
                     (conv2dParams->quanParameter()->type() == 2))) {
                    // Todo: support int4 inputchannel % 4 not equal 4
                    return new ConvBufLowMemoryExecution(inputs, outputs, op, backend);
                } else {
                    MNN_ERROR("OpenCL Conv buf low memory init error. For Opencl Backend, only support low memory mode of int8 or int4 dequantization currently.\n");
                    MNN_ASSERT(false);
                }
            }
        }  
#endif
        if (nullptr != op->main_as_Convolution2D()->quanParameter()) {
            auto quan = op->main_as_Convolution2D()->quanParameter();
            if (1 == quan->type() || 2 == quan->type()) {
                if (quan->has_scaleInt()) {
                    // Don't support IDST-int8 because of error
                    return nullptr;
                }
            }
        }
        
        if(op->main_as_Convolution2D()->common()->group() > 1){
            // Don't support group > 1 now
            return nullptr;
        }
        
        if (inputs.size() > 1) {
            // Multi inputs
            for (int i = 0; i < inputs.size(); ++i) {
                TensorUtils::setTensorSupportPack(inputs[i], false);
            }
            for (int i = 0; i < outputs.size(); ++i) {
                TensorUtils::setTensorSupportPack(outputs[i], false);
            }
            return new ConvBufExecution(inputs, outputs, op, backend);
        }

        if (ConvBufWinograd::valid(conv2D->common(), inputs[0], outputs[0], static_cast<OpenCLBackend *>(backend)->getOpenCLRuntime()->getGpuType() == INTEL)) {
            if(static_cast<OpenCLBackend *>(backend)->getOpenCLRuntime()->isSupportedIntelSubgroup()){
                std::vector<int> inputShape = tensorShapeFormat(input);
                std::vector<int> outputShape = tensorShapeFormat(output);
                const int src_width = inputShape.at(2);
                const int dst_width = outputShape.at(2);
                int pad_right                = (UP_DIV(dst_width, 2) - 1) * 2 + 3 - padding.first - src_width + 1;
                TensorUtils::setTensorPad(input, padding.first, pad_right, 0, 0);
                TensorUtils::setTensorChannelPack(input, 16);
            }
            return new ConvBufWinograd(op, backend);
        }
#ifdef MNN_SUPPORT_INTEL_SUBGROUP
        if (static_cast<OpenCLBackend *>(backend)->getOpenCLRuntime()->isSupportedIntelSubgroup() && outputChannel >= 16) {
            if (inputChannels >= 16) {
                auto pads = ConvolutionCommon::convolutionPadFull(inputs[0], outputs[0], conv2D->common());
                TensorUtils::setTensorPad(inputs[0], std::get<0>(pads), std::get<2>(pads), 0, 0);
                TensorUtils::setTensorChannelPack(inputs[0], 16);
            }
            return new ConvSubgroupBuf(inputs, outputs, op, backend);
        }
#endif /* MNN_SUPPORT_INTEL_SUBGROUP */
        for (int i = 0; i < inputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(inputs[i], false);
        }
        for (int i = 0; i < outputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(outputs[i], false);
        }
        return new ConvBufExecution(inputs, outputs, op, backend);
    }
};

REGISTER_OPENCL_OP_CREATOR(ConvolutionBufCreator, OpType_Convolution, BUFFER);

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
