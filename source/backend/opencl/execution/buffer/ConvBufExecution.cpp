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

static float EstimateOccupancy(int blockWidth, int x, int y, int f, int b, int slm_div_factor, int maxThreadsPerDevice) {

    auto threads =  UP_DIV(x, blockWidth) * y * UP_DIV(f, 16) * slm_div_factor * b;

    return static_cast<float>(threads) / static_cast<float>(maxThreadsPerDevice);
}


static std::pair<int, int> GetTuningParams(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, const uint32_t maxWorkGroupSize, const bool isSupportedFP16, const int maxThreadsPerDevice) {

    auto input  = inputs[0];
    auto output = outputs[0];
    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);
    const int height             = outputShape.at(1);
    const int width              = outputShape.at(2);
    const int outChannel         = outputShape.at(3);
    const int batch              = outputShape.at(0);

    const int inputHeight   = inputShape.at(1);
    const int inputWidth    = inputShape.at(2);
    const int inputChannels = inputShape.at(3);

    size_t ic_blocks = UP_DIV(inputChannels, 16);

    size_t max_slm_div_factor = maxWorkGroupSize / 16;
    int blockWidth = 2;
    int slm_div_factor = 1;
    int xf = width * outChannel;
    if (xf <= 256) {
        if (width <= 8 || xf <= 128)
            blockWidth = 2;
        else
            blockWidth = 4;
    } else if (xf <= 1536) {
        blockWidth = 4;
    } else {
        if (width >= 8 && width < 12 && xf < 2600)
            blockWidth = 4;
        else if (width < 12 && xf < 8192)
            blockWidth = 8;
        else
            blockWidth =  8;
    }

    bool slm_exception = width == 3 && height == 3 && !isSupportedFP16 && outChannel <= 512;

    if (!slm_exception)
        while (ic_blocks % (slm_div_factor * 2) == 0 && (slm_div_factor * 2 <= max_slm_div_factor) &&
               EstimateOccupancy(blockWidth, width, height, outChannel, batch, slm_div_factor, maxThreadsPerDevice) <
                   4.0)
            slm_div_factor *= 2;

    return {blockWidth, slm_div_factor};
}

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
    
    if (tunedLws.find(info) == tunedLws.end()) {
        //printf("ConvBuf2dGeneralLocalWS %d Insert! gws:%d %d, lws:%d %d, time:%dus\n", (int)tunedLws.size(), gws[0], gws[1], lws_prefer[0], lws_prefer[1], min_cost);
        tunedLws.insert(std::make_pair(info, std::make_pair(lws_prefer, min_cost)));
    }
    return std::make_pair(lws_prefer, min_cost);
}

ConvBufCommonExecution::ConvBufCommonExecution(const Convolution2D *conv2dParams, Backend *backend) : Execution(backend) {
    auto openclBackend       = (OpenCLBackend *)backend;
    int biasSize             = conv2dParams->common()->outputCount();
    int buffer_size = ROUND_UP(biasSize, 16);//pack to 16
    if(openclBackend->getOpenCLRuntime()->isSupportedFP16()) {
        buffer_size *= sizeof(half_float::half);
    } else {
        buffer_size *= sizeof(float);
    }

    mBias.reset(Tensor::createDevice<float>({1, 1, 1, ROUND_UP(biasSize, 16)}));
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

void ConvBufExecution::transformWeight(const Tensor *weightDest, const Tensor *source) {
    int co      = source->length(0);
    int ci     = source->length(1);
    int KernelY = source->length(2);
    int KernelX = source->length(3);

    ::memset(weightDest->host<float>(), 0, weightDest->size());

    auto weightPtr      = source->host<float>();
    for (int oz = 0; oz < co; ++oz) {
        auto srcOz = weightPtr + oz * ci * KernelY * KernelX;

        int ozC4 = oz / 16;
        int mx   = oz % 16;

        auto dstOz = weightDest->host<float>() + weightDest->stride(0) * ozC4 + mx;
        for (int sz = 0; sz < ci; ++sz) {
            int szC4         = sz / 16;
            int my           = sz % 16;
            auto srcSz       = srcOz + KernelY * KernelX * sz;
            auto dstSz = dstOz + szC4 * weightDest->stride(1) + my * 16;

            for (int i = 0; i < KernelY * KernelX; ++i) {
                *(dstSz + i * weightDest->stride(2)) = srcSz[i];
            }
        }
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
    mUseSubgroup = mOpenCLBackend->getOpenCLRuntime()->getGpuType() == INTEL && mOpenCLBackend->getOpenCLRuntime()->isSupportedIntelSubgroup() && inputs.size() == 1 && mOutputChannel >= 16;

    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
    if (inputs.size() != 1) {
        // Multi - Input
        mConv1x1Opt = false;
        mRasterExe.reset(new RasterBufExecution({mFilter.get()}, op, mOpenCLBackend));
    } else {
        int weightSize   = 0;
        ConvolutionCommon::getConvParameters(&quanCommon, conv2dParams, &mFilterDataPtr, &weightSize);
        //select opt conv method
        mConv1x1Opt = (mKernelHeight == mKernelWidth && mKernelHeight == 1 && mPaddings[0] == 0 &&
        mPaddings[1] == 0 && mStrides[0] == 1 && mStrides[1] == 1 && inputs[0]->width() >= 4);
    }
    if (mUseSubgroup) {
        // create tensor for intel filter
        mFilter.reset(Tensor::createDevice<float>(std::vector<int>{
            UP_DIV(mOutputChannel, 16), UP_DIV(mInputChannel, 16), mKernelWidth * mKernelHeight, 16, 16}));
        auto res = mOpenCLBackend->onAcquireBuffer(mFilter.get(), Backend::STATIC);
        cl_int ret_code;
        if (!res) {
            mValid = false;
            return;
        }
        if (mFilterDataPtr != nullptr) {
            std::shared_ptr<Tensor> sourceWeight(
                Tensor::create<float>(std::vector<int>{mOutputChannel, mInputChannel, mKernelWidth, mKernelHeight},
                                      (void *)mFilterDataPtr, Tensor::CAFFE));
            std::shared_ptr<Tensor> destWeight(Tensor::create<float>(std::vector<int>{
                UP_DIV(mOutputChannel, 16), UP_DIV(mInputChannel, 16), mKernelWidth * mKernelHeight, 16, 16}));

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
    }else if (mConv1x1Opt) {
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
            _generateFilterConvertRegion(mFilter.get(), originBuffer.get());
            std::shared_ptr<Execution> raster(new RasterBufExecution({}, op, mOpenCLBackend));
            raster->onResize({}, {mFilter.get()});
            raster->onExecute({}, {mFilter.get()});
            // STATIC mode's buffer will be released by tensor free
        }
    }
    // Create Kernel
    if (mConv2dCommonParams->relu()) {
        mBuildOptions.emplace("-DRELU");
    } else if (mConv2dCommonParams->relu6()) {
        mBuildOptions.emplace("-DRELU6");
    }

    mKernel           = mOpenCLBackend->getOpenCLRuntime()->buildKernel("conv_2d_buf", kernelName, mBuildOptions);
    mMaxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(mKernel));

#ifdef LOG_VERBOSE
    MNN_PRINT("end ConvExecution init !\n");
#endif
}

ConvBufExecution::~ConvBufExecution() {
    // Do nothing
}

ErrorCode ConvBufExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start ConvExecution onResize !\n");
#endif
    auto input  = inputs[0];
    auto output = outputs[0];
    if (inputs.size() > 1) {
        // Multi Input, need pretreat
        _generateFilterConvertRegion(mFilter.get(), inputs[1]);
        bool res = backend()->onAcquireBuffer(mFilter.get(), Backend::DYNAMIC);
        if (!res) {
            return OUT_OF_MEMORY;
        }
        mRasterExe->onResize({}, {mFilter.get()});
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
    
    std::string info = std::to_string(inputChannels) + "_" + std::to_string(mKernelHeight) + "_" + std::to_string(mKernelWidth) + "_" + std::to_string(mStrides[0]) + "_" + std::to_string(mStrides[1]) + "_" + std::to_string(mDilations[0]) + "_" + std::to_string(mDilations[1]);
    if (mUseSubgroup) {
        // create temp buffer for subgroup
        int input_width_pad = mStrides[1] * (8 - 1) + (mKernelWidth - 1) * mDilations[1] + 1 + width * mStrides[1] + mPaddings[1];
        int input_height_pad = inputHeight + 2 * mPaddings[0];
        if (input->channel() >=16){
            mSource.reset(Tensor::createDevice<float>(std::vector<int>{inputShape.at(0), UP_DIV(input->channel(), 16),(input_height_pad) * (input_width_pad), 16}, Tensor::CAFFE_C4));
        } else {
            input_width_pad = inputWidth;
            input_height_pad = inputHeight;
            mSource.reset(Tensor::createDevice<float>(std::vector<int>{inputShape.at(0), input->channel(), inputHeight, inputWidth}, Tensor::CAFFE_C4));
        }
        std::string kernelName[3];
        uint32_t MaxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->MaxWorkGroupSize());
        uint32_t MaxThreadsPerDevice = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->MaxThreadsPerDevice());
        bool isSupportedFP16 = mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16();

        mOpenCLBackend->onAcquireBuffer(mSource.get(), Backend::DYNAMIC);
        mOpenCLBackend->onReleaseBuffer(mSource.get(), Backend::DYNAMIC);

        int inputImageShape[2]             = {inputHeight, inputWidth};
        int outputImageShape[2]            = {height, width};
        int kernelShape[2]                 = {mKernelHeight, mKernelWidth};
        int strideShape[2]                 = {mStrides[0], mStrides[1]};
        int paddingShape[2]                = {mPaddings[0], mPaddings[1]};
        int dilationShape[2]               = {mDilations[0], mDilations[1]};
        std::set<std::string> buildOptions = mBuildOptions;
        auto tune_param = GetTuningParams(inputs, outputs, MaxWorkGroupSize, isSupportedFP16, MaxThreadsPerDevice);
        uint32_t blockWidth = tune_param.first;
        uint32_t sub_group_size = 16;
        uint32_t slm_div_factor = tune_param.second;
        uint32_t work_group_size = sub_group_size * slm_div_factor;
        uint32_t feature_block_size = 16;        
        uint32_t input_line_size = strideShape[1] * (blockWidth - 1) + (kernelShape[1] - 1) * dilationShape[1] + 1;
        uint32_t input_block_size = UP_DIV(input_line_size * kernelShape[0], sub_group_size);
        buildOptions.emplace("-DOUTPUT_X_BLOCK_SIZE=" + std::to_string(blockWidth));
        buildOptions.emplace("-DINPUT_LINE_SIZE=" + std::to_string(input_line_size));
        buildOptions.emplace("-DINPUT_BLOCK_SIZE=" + std::to_string(input_block_size));
        buildOptions.emplace("-DSUB_GROUP_SIZE=" + std::to_string(sub_group_size));
        buildOptions.emplace("-DX_BLOCKS=" + std::to_string(UP_DIV(outputImageShape[1], blockWidth)));
        buildOptions.emplace("-DSLM_DIV_FACTOR=" + std::to_string(slm_div_factor));
        buildOptions.emplace("-DWORK_GROUP_SIZE=" + std::to_string(work_group_size));
        buildOptions.emplace("-DIC_BLOCKS=" + std::to_string(UP_DIV(inputChannels, feature_block_size)));
        buildOptions.emplace("-DINPUT_CHANNEL=" + std::to_string(inputChannels));
        buildOptions.emplace("-DOUTPUT_CHANNEL=" + std::to_string(outChannel));
        buildOptions.emplace("-DFILTER_HEIGHT=" + std::to_string(kernelShape[0]));
        buildOptions.emplace("-DFILTER_WIDTH=" + std::to_string(kernelShape[1]));
        buildOptions.emplace("-DDILATION_HEIGHT=" + std::to_string(dilationShape[0]));
        buildOptions.emplace("-DDILATION_WIDTH=" + std::to_string(dilationShape[1]));
        buildOptions.emplace("-DSTRIDE_HEIGHT=" + std::to_string(strideShape[0]));
        buildOptions.emplace("-DSTRIDE_WIDTH=" + std::to_string(strideShape[1]));
        buildOptions.emplace("-DINPUT_HEIGHT=" + std::to_string(inputImageShape[0]));
        buildOptions.emplace("-DINPUT_WIDTH=" + std::to_string(inputImageShape[1]));
        buildOptions.emplace("-DOUTPUT_HEIGHT=" + std::to_string(outputImageShape[0]));
        buildOptions.emplace("-DOUTPUT_WIDTH=" + std::to_string(outputImageShape[1]));
        buildOptions.emplace("-DPADDING_HEIGHT=" + std::to_string(paddingShape[0]));
        buildOptions.emplace("-DPADDING_WIDTH=" + std::to_string(paddingShape[1]));
        buildOptions.emplace("-DINPUT_HEIGHT_PAD=" + std::to_string(input_height_pad));
        buildOptions.emplace("-DINPUT_WIDTH_PAD=" + std::to_string(input_width_pad));
        if (outChannel % feature_block_size != 0) {
             buildOptions.emplace("-DOUTPUT_LEFTOVERS=" + std::to_string(1));
        }

        {
            uint32_t channel_block = 4;
            if (inputChannels < 16) {
                kernelName[0] = "transpose_c1";
                mKernelSub[0] = mOpenCLBackend->getOpenCLRuntime()->buildKernel("conv_2d_buf_subgroup", kernelName[0], buildOptions);
            } else {
                kernelName[0] = "transpose_c16";
                mKernelSub[0] = mOpenCLBackend->getOpenCLRuntime()->buildKernel("conv_2d_buf_subgroup", kernelName[0], buildOptions);
                channel_block = 16;
            }
            uint32_t mMaxWGS_S = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(mKernelSub[0]));

            mGlobalWorkSizeSub[0] = {static_cast<uint32_t>(input_width_pad), static_cast<uint32_t>(input_height_pad),
                                     static_cast<uint32_t>(inputShape.at(0) * UP_DIV(inputShape.at(3), channel_block))};
            uint32_t idx      = 0;
            mKernelSub[0].setArg(idx++, mGlobalWorkSizeSub[0][0]);
            mKernelSub[0].setArg(idx++, mGlobalWorkSizeSub[0][1]);
            mKernelSub[0].setArg(idx++, mGlobalWorkSizeSub[0][2]);
            mKernelSub[0].setArg(idx++, openCLBuffer(input));
            mKernelSub[0].setArg(idx++, openCLBuffer(mSource.get()));
            mKernelSub[0].setArg(idx++, UP_DIV(inputShape.at(3), channel_block));

            mLocalWorkSizeSub[0]  = localWS3DDefault(mGlobalWorkSizeSub[0], mMaxWGS_S, mOpenCLBackend->getOpenCLRuntime(), kernelName[0], mKernelSub[0]).first;
        }

        {
            if (inputChannels < 16){
                kernelName[1] = "conv_2d_buf_subgroup_c1";
                mKernelSub[1] = mOpenCLBackend->getOpenCLRuntime()->buildKernel("conv_2d_buf_subgroup", kernelName[1], buildOptions);
            } else {
                kernelName[1] = "conv_2d_buf_subgroup_c16";
                mKernelSub[1] = mOpenCLBackend->getOpenCLRuntime()->buildKernel("conv_2d_buf_subgroup", kernelName[1], buildOptions);
            }

            mGlobalWorkSizeSub[1] = {static_cast<uint32_t>(UP_DIV(outputShape.at(2), blockWidth) * outputShape.at(1)),
                                     static_cast<uint32_t>(ROUND_UP(outputShape.at(3), sub_group_size) * slm_div_factor),
                                 static_cast<uint32_t>(outputShape.at(0))};
            mLocalWorkSizeSub[1]  = {1, static_cast<uint32_t>(sub_group_size * slm_div_factor), 1};
            uint32_t idx      = 0;
            mKernelSub[1].setArg(idx++, openCLBuffer(mSource.get()));
            mKernelSub[1].setArg(idx++, openCLBuffer(output));
            mKernelSub[1].setArg(idx++, openCLBuffer(mFilter.get()));
            mKernelSub[1].setArg(idx++, openCLBuffer(mBias.get()));
        }
    }else if (mConv1x1Opt) {
    
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
        
        // {"conv_2d_c4h1w2", "conv_2d_c4h1w1", "conv_2d_c8h1w1", "conv_2d_c4h1w4", "conv_2d_c8h2w1", "conv_2d_c4h4w1"};
        const int total_kernel = 6;
        std::string kernelName[total_kernel] = {"conv_2d_c4h1w1", "conv_2d_c4h1w2", "conv_2d_c4h4w1", "conv_2d_c8h2w1", "conv_2d_c8h4w1", "conv_2d_c4h1w4"};
        int itemC[total_kernel] = {4, 4, 4, 8, 8, 4};
        int itemH[total_kernel] = {1, 1, 4, 2, 4, 1};
        int itemW[total_kernel] = {1, 2, 1, 1, 1, 4};
        
        
        int actual_kernel = total_kernel;
        if(mOpenCLBackend->getOpenCLRuntime()->getCLTuneLevel() == Normal) {
            actual_kernel = 2;
        } else if(mOpenCLBackend->getOpenCLRuntime()->getCLTuneLevel() == Fast) {
            actual_kernel = 1;
        }else if(mOpenCLBackend->getOpenCLRuntime()->getCLTuneLevel() == Wide){
            actual_kernel = 4;
            auto gpuType = mOpenCLBackend->getOpenCLRuntime()->getGpuType();
            auto maliArType = mOpenCLBackend->getOpenCLRuntime()->getMaliAr();
            if(gpuType == MNN::MALI && maliArType == MNN::VALHALL){
                if(outputShape.at(3) <= 8){
                    kernelName[3] = "conv_2d_c4h1w4";
                    itemC[3]      = 4;
                    itemH[3]      = 1;
                    itemW[3]      = 4;
                }else{
                    kernelName[2] = "conv_2d_c8h2w1";
                    itemC[2]      = 8;
                    itemH[2]      = 2;
                    itemW[2]      = 1;
                                
                    kernelName[3] = "conv_2d_c8h4w1";
                    itemC[3]      = 8;
                    itemH[3]      = 4;
                    itemW[3]      = 1;
                }
            }
        }
        
        cl::Kernel kernel[total_kernel];
        std::vector<uint32_t> globalWorkSize[total_kernel];
        std::vector<uint32_t> localWorkSize[total_kernel];
        std::pair<int, int> min_cost(INT_MAX, 0);//(min_time, min_index)
        for(int knl_idx = 0; knl_idx < actual_kernel; knl_idx++) {
            kernel[knl_idx]        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("conv_2d_buf", kernelName[knl_idx], mBuildOptions);
            uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(kernel[knl_idx]));
            
            globalWorkSize[knl_idx] = {static_cast<uint32_t>(UP_DIV(outputShape.at(3), itemC[knl_idx]) * UP_DIV(outputShape.at(2), itemW[knl_idx])), static_cast<uint32_t>(outputShape.at(0) * UP_DIV(outputShape.at(1), itemH[knl_idx]))};
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
            kernel[knl_idx].setArg(idx++, UP_DIV(height, itemH[knl_idx]));
            
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
        mKernel.setArg(idx++, UP_DIV(height, itemH[min_index]));
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
        mRasterExe->onExecute({}, {mFilter.get()});
        if (inputs.size() > 2) {
            auto buffer_size = inputs[2]->elementSize();
            if(mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16()) {
                buffer_size *= sizeof(half_float::half);
            } else {
                buffer_size *= sizeof(float);
            }
            mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueCopyBuffer(openCLBuffer(inputs[2]), openCLBuffer(mBias.get()), 0, 0, buffer_size);
        }
    }
#ifdef ENABLE_OPENCL_TIME_PROFILER
    if (mUseSubgroup) {
        int costTime = 0;
        for (int i = 0; i < 2; i++) {
            cl::Event event;
            run3DKernelDefault(mKernelSub[i], mGlobalWorkSizeSub[i], mLocalWorkSizeSub[i],
                               mOpenCLBackend->getOpenCLRuntime(), &event);
            int costTime0 = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
            costTime += costTime0;
            MNN_PRINT("kernel cost:%d    us ConvBuf2DSub step %d\n", costTime0, i);
        }
        MNN_PRINT("kernel cost:%d    us total ConvBuf2DSub\n", costTime);
    } else {
        cl::Event event;
        runKernel2D(mKernel, mGlobalWorkSize, mLocalWorkSize, mOpenCLBackend->getOpenCLRuntime(), &event);
        int costTime = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
        MNN_PRINT("kernel cost:%d    us ConvBuf2D\n",costTime);
    }  
#else
    if (mUseSubgroup) {
        for (int i = 0; i < 2; i++) {
            run3DKernelDefault(mKernelSub[i], mGlobalWorkSizeSub[i], mLocalWorkSizeSub[i],
                               mOpenCLBackend->getOpenCLRuntime());
        }
    } else {
        runKernel2D(mKernel, mGlobalWorkSize, mLocalWorkSize, mOpenCLBackend->getOpenCLRuntime());
    }
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
        
        if (inputs.size() > 1) {
            // Multi inputs
            return new ConvBufExecution(inputs, outputs, op, backend);
        }
        auto conv2D = op->main_as_Convolution2D();
        if (ConvBufWinograd::valid(conv2D->common(), inputs[0], outputs[0], static_cast<OpenCLBackend *>(backend)->getOpenCLRuntime()->getGpuType() == INTEL)) {
            return new ConvBufWinograd(conv2D, backend);
        }
        return new ConvBufExecution(inputs, outputs, op, backend);
    }
};

OpenCLCreatorRegister<ConvolutionBufCreator> __convBuf_op(OpType_Convolution, BUFFER);

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
