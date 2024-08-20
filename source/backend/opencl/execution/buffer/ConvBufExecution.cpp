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

ConvBufCommonExecution::ConvBufCommonExecution(Backend *backend) {
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
}
ConvBufCommonExecution::ConvBufCommonExecution(const Convolution2D *conv2dParams, Backend *backend) {
    auto openclBackend       = (OpenCLBackend *)backend;
    int biasSize             = conv2dParams->common()->outputCount();
    int buffer_size = ROUND_UP(biasSize, 32);//pack to packN
    if(openclBackend->getOpenCLRuntime()->isSupportedFP16()) {
        buffer_size *= sizeof(half_float::half);
    } else {
        buffer_size *= sizeof(float);
    }

    mResource.reset(new ConvBufResource);
    mResource->mBias.reset(Tensor::createDevice<float>({1, 1, 1, ROUND_UP(biasSize, 32)}));
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
        bool isConv1x1 = (mResource->mKernelHeight == mResource->mKernelWidth && mResource->mKernelHeight == 1 && mPaddings[0] == 0 &&
                          mPaddings[1] == 0 && mResource->mStrides[0] == 1 && mResource->mStrides[1] == 1);

        mResource->mConv1x1Opt = isConv1x1;
        mResource->mConv1x1C8Opt = mResource->mConv1x1Opt && mResource->mOutputChannel >= 16;
        bool useConvGemm = isConv1x1 && mResource->mInputChannel > 32 && mResource->mOutputChannel > 64;
        if (useConvGemm) {
            mResource->mConvGemmOptLevel = 2;
        }
    }
    if (mResource->mConv1x1Opt) {
        // Tile Match with mConvGemmOptLevel == 2
        int tileK = 4;
        int tileN = 32;
        
        int buffer_size = ROUND_UP(mResource->mOutputChannel, tileN) * ROUND_UP(mResource->mInputChannel, tileK);
        mResource->mFilter.reset(
            Tensor::createDevice<float>({buffer_size}));
        mOpenCLBackend->onAcquireBuffer(mResource->mFilter.get(), Backend::STATIC);

        if (mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16()) {
            buffer_size *= sizeof(half_float::half);
        } else {
            buffer_size *= sizeof(float);
        }

        cl::Buffer &filterBuffer = openCLBuffer(mResource->mFilter.get());
        cl_int error;
        auto ptrCL = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(
                filterBuffer, true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &error);
        if(nullptr != ptrCL && error == CL_SUCCESS){
            memset((void *)ptrCL, 0, buffer_size);
            if (mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16()) {
                // [Ci, Co] ( [K, N] )
                for (int o = 0; o < mResource->mOutputChannel; o++) {
                    for (int i = 0; i < mResource->mInputChannel; i++) {
                        ((half_float::half *)ptrCL)[i * ROUND_UP(mResource->mOutputChannel, tileN) + o] = (half_float::half)(mFilterDataPtr[o * mResource->mInputChannel + i]);
                    }
                }
            } else {
                for (int o = 0; o < mResource->mOutputChannel; o++) {
                    for (int i = 0; i < mResource->mInputChannel; i++) {
                        ((float *)ptrCL)[i * ROUND_UP(mResource->mOutputChannel, tileN) + o] = (mFilterDataPtr[o * mResource->mInputChannel + i]);
                    }
                }
            }
        }else{
            MNN_ERROR("Map error filterPtrCL == nullptr \n");
        }
        mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(filterBuffer, ptrCL);

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
    
    // printf("nchw %d %d %d %d, cohw %d %d %d, khw %d %d  gemm:%d \n", inputs[0]->batch(), inputs[0]->channel(), inputs[0]->height(), inputs[0]->width(), outputs[0]->channel(), outputs[0]->height(), outputs[0]->width(), mResource->mKernelWidth, mResource->mKernelHeight, mResource->mConvGemmOptLevel);
    
    std::string info = std::to_string(inputChannels) + "_" + std::to_string(outChannel) + "_" + std::to_string(mResource->mKernelHeight) + "_" + std::to_string(mResource->mKernelWidth) + "_" + std::to_string(mResource->mStrides[0]) + "_" + std::to_string(mResource->mStrides[1]) + "_" + std::to_string(mResource->mDilations[0]) + "_" + std::to_string(mResource->mDilations[1]);
    
    if (mResource->mConvGemmOptLevel > 0) {
        int area = height * width;
        int M = outputShape.at(0) * area;
        int N = outputShape.at(3);
        int K = inputShape.at(3);
        
        bool isAlign = (K % 8 == 0 && area == 1 && N % 64 == 0 && M % 64 == 0);
        bool isLimitSize = (M * 1.0 / 512 * N / 512 * K / 512 <= 1.0) && (1.0 * M * K / N / N >= 16.0);
        if(isAlign && isLimitSize) {
            mResource->mConvGemmOptLevel = 1;
        } else if(M < 128 || 1.0 * M / 512 * N / 512 * K / 256 < 1.0) {
            mResource->mConvGemmOptLevel = 0;
        }
    }
    
    if (mResource->mConvGemmOptLevel == 2) {
        // set large tile
        int tileM = 16;
        int tileN = 32;
        int tileK = 4;

        int area = height * width;
        int M = outputShape.at(0) * area;
        int N = outputShape.at(3);
        int K = inputShape.at(3);
        
        int alignM = ROUND_UP(M, tileM);
        int alignN = ROUND_UP(N, tileN);
        int alignK = ROUND_UP(K, tileK);
        
        // ReArrange input
        mConvGemmInpTensor.reset(Tensor::createDevice<float>({alignK * alignM}));
        mOpenCLBackend->onAcquireBuffer(mConvGemmInpTensor.get(), Backend::DYNAMIC);
        if(N != alignN || M != alignM || area != 1) {
            mNeedOutTempTensor = true;
            mConvGemmOutTensor.reset(Tensor::createDevice<float>({alignN * alignM}));
            mOpenCLBackend->onAcquireBuffer(mConvGemmOutTensor.get(), Backend::DYNAMIC);
        }
        mOpenCLBackend->onReleaseBuffer(mConvGemmInpTensor.get(), Backend::DYNAMIC);
        if(mNeedOutTempTensor) {
            mOpenCLBackend->onReleaseBuffer(mConvGemmOutTensor.get(), Backend::DYNAMIC);
        }
        
        {
            std::set<std::string> buildOptions;
            
            int m_pack = 1;
            if(area == 1) {
                m_pack = 4;
                buildOptions.emplace("-DAREA_EQUAL_1");
            } else if(outputShape.at(0) == 1) {
                m_pack = 4;
                buildOptions.emplace("-DBATCH_EQUAL_1");
            }
            mPreKernel = mOpenCLBackend->getOpenCLRuntime()->buildKernel("gemm_buf", "transpose_pad", buildOptions);
            uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(mPreKernel));
            mPreGlobalWorkSize = {static_cast<uint32_t>(alignM/m_pack), static_cast<uint32_t>(alignK/4)};

            int offset = 0;
            int idx            = 0;
            cl_int ret = CL_SUCCESS;
            ret |= mPreKernel->get().setArg(idx++, static_cast<int>(mPreGlobalWorkSize[0]));
            ret |= mPreKernel->get().setArg(idx++, static_cast<int>(mPreGlobalWorkSize[1]));
            ret |= mPreKernel->get().setArg(idx++, static_cast<int>(alignM));
            ret |= mPreKernel->get().setArg(idx++, static_cast<int>(alignK));
            ret |= mPreKernel->get().setArg(idx++, static_cast<int>(M));
            ret |= mPreKernel->get().setArg(idx++, static_cast<int>(K));
            ret |= mPreKernel->get().setArg(idx++, static_cast<int>(area));
            ret |= mPreKernel->get().setArg(idx++, openCLBuffer(input));
            ret |= mPreKernel->get().setArg(idx++, openCLBuffer(mConvGemmInpTensor.get()));
            MNN_CHECK_CL_SUCCESS(ret, "setArg mConvgemmOptLevel==2 PreKernel");
            mPreLocalWorkSize = localWS2DDefault(mPreGlobalWorkSize, maxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), "transpose_pad", mPreKernel).first;

            mOpenCLBackend->recordKernel2d(mPreKernel, mPreGlobalWorkSize, mPreLocalWorkSize);
            mPreGlobalWorkSize[0] = ROUND_UP(mPreGlobalWorkSize[0], std::max((uint32_t)1, mPreLocalWorkSize[0]));
            mPreGlobalWorkSize[1] = ROUND_UP(mPreGlobalWorkSize[1], std::max((uint32_t)1, mPreLocalWorkSize[1]));
        }
        std::set<std::string> buildOptions;
        
        uint32_t hasBias = 0;
        if(!mNeedOutTempTensor) {
            hasBias = 1;
            buildOptions = mResource->mBuildOptions;
            buildOptions.emplace("-DBIAS");
        }
        uint32_t layout = 4;
        uint32_t batch = 1;
        
        cl::Buffer outBuffer = mNeedOutTempTensor ? openCLBuffer(mConvGemmOutTensor.get()) : openCLBuffer(output);
        std::vector<uint32_t> param;
        if(mNeedOutTempTensor) {
            param = getGemmParams({(uint32_t)alignM, (uint32_t)alignN, (uint32_t)alignK, layout, batch, hasBias}, {openCLBuffer(mConvGemmInpTensor.get()), openCLBuffer(mResource->mFilter.get()), openCLBuffer(mConvGemmOutTensor.get())}, mOpenCLBackend->getOpenCLRuntime());
        } else {
            param = getGemmParams({(uint32_t)alignM, (uint32_t)alignN, (uint32_t)alignK, layout, batch, hasBias}, {openCLBuffer(mConvGemmInpTensor.get()), openCLBuffer(mResource->mFilter.get()), openCLBuffer(output), openCLBuffer(mResource->mBias.get())}, mOpenCLBackend->getOpenCLRuntime());
        }

        int KWG=param[0], KWI=param[1], MDIMA=param[2], MDIMC=param[3], MWG=param[4], NDIMB=param[5], NDIMC=param[6], NWG=param[7], SA=param[8], SB=param[9], STRM=param[10], STRN=param[11], VWM=param[12], VWN=param[13];
        buildOptions.emplace("-DKWG=" + std::to_string(KWG));
        buildOptions.emplace("-DKWI=" + std::to_string(KWI));
        buildOptions.emplace("-DMDIMA=" + std::to_string(MDIMA));
        buildOptions.emplace("-DMDIMC=" + std::to_string(MDIMC));
        buildOptions.emplace("-DMWG=" + std::to_string(MWG));
        buildOptions.emplace("-DNDIMB=" + std::to_string(NDIMB));
        buildOptions.emplace("-DNDIMC=" + std::to_string(NDIMC));
        buildOptions.emplace("-DNWG=" + std::to_string(NWG));
        buildOptions.emplace("-DSA=" + std::to_string(SA));
        buildOptions.emplace("-DSB=" + std::to_string(SB));
        buildOptions.emplace("-DSTRM=" + std::to_string(STRM));
        buildOptions.emplace("-DSTRN=" + std::to_string(STRN));
        buildOptions.emplace("-DVWM=" + std::to_string(VWM));
        buildOptions.emplace("-DVWN=" + std::to_string(VWN));
        if(layout >= 4) {
            buildOptions.emplace("-DOUTPUTMN");
        }

        tileM = MWG;
        tileN = NWG;
        int localM = MDIMC;
        int localN = NDIMC;
                 
        if(mOpenCLBackend->getOpenCLRuntime()->getGpuType() == GpuType::ADRENO) {
            buildOptions.emplace("-DUSE_CL_MAD=1");
            buildOptions.emplace("-DRELAX_WORKGROUP_SIZE=1");
        }

        mKernel = mOpenCLBackend->getOpenCLRuntime()->buildKernel("matmul_params_buf", "Xgemm", buildOptions);
        
        int out_per_thread_m = tileM / localM;
        int out_per_thread_n = tileN / localN;
        
        mGlobalWorkSize = {static_cast<uint32_t>(alignM/out_per_thread_m), static_cast<uint32_t>(alignN/out_per_thread_n)};
        mLocalWorkSize = {static_cast<uint32_t>(localM), static_cast<uint32_t>(localN)};
        
        float alpha = 1.0;
        float beta = 0.0f;
        int offset = 0;
        int idx            = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mKernel->get().setArg(idx++, static_cast<int>(alignM));
        ret |= mKernel->get().setArg(idx++, static_cast<int>(alignN));
        ret |= mKernel->get().setArg(idx++, static_cast<int>(alignK));
        ret |= mKernel->get().setArg(idx++, alpha);
        ret |= mKernel->get().setArg(idx++, beta);
        ret |= mKernel->get().setArg(idx++, openCLBuffer(mConvGemmInpTensor.get()));
        ret |= mKernel->get().setArg(idx++, openCLBuffer(mResource->mFilter.get()));
        if(mNeedOutTempTensor) {
            ret |= mKernel->get().setArg(idx++, openCLBuffer(mConvGemmOutTensor.get()));
        } else {
            ret |= mKernel->get().setArg(idx++, openCLBuffer(mResource->mBias.get()));
            ret |= mKernel->get().setArg(idx++, openCLBuffer(output));
        }
        ret |= mKernel->get().setArg(idx++, offset);
        ret |= mKernel->get().setArg(idx++, offset);
        ret |= mKernel->get().setArg(idx++, offset);
        
        MNN_CHECK_CL_SUCCESS(ret, "setArg Conv1x1Buf mConvgemmOptLevel==2 Kernel Select");
        mOpenCLBackend->recordKernel2d(mKernel, mGlobalWorkSize, mLocalWorkSize);
        mGlobalWorkSize[0] = ROUND_UP(mGlobalWorkSize[0], std::max((uint32_t)1, mLocalWorkSize[0]));
        mGlobalWorkSize[1] = ROUND_UP(mGlobalWorkSize[1], std::max((uint32_t)1, mLocalWorkSize[1]));
        
        if(mNeedOutTempTensor) {
            std::set<std::string> buildOptions = mResource->mBuildOptions;
            if(area == 1) {
                buildOptions.emplace("-DAREA_EQUAL_1");
            }
            mPostKernel = mOpenCLBackend->getOpenCLRuntime()->buildKernel("gemm_buf", "transpose_bias", buildOptions);
            uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(mPostKernel));

            mPostGlobalWorkSize = {static_cast<uint32_t>(M), static_cast<uint32_t>(UP_DIV(N, 16))};

            int offset = 0;
            int idx            = 0;
            cl_int ret = CL_SUCCESS;
            ret |= mPostKernel->get().setArg(idx++, static_cast<int>(mPostGlobalWorkSize[0]));
            ret |= mPostKernel->get().setArg(idx++, static_cast<int>(mPostGlobalWorkSize[1]));
            ret |= mPostKernel->get().setArg(idx++, static_cast<int>(alignM));
            ret |= mPostKernel->get().setArg(idx++, static_cast<int>(alignN));
            ret |= mPostKernel->get().setArg(idx++, static_cast<int>(M));
            ret |= mPostKernel->get().setArg(idx++, static_cast<int>(N));
            ret |= mPostKernel->get().setArg(idx++, static_cast<int>(area));
            ret |= mPostKernel->get().setArg(idx++, openCLBuffer(mConvGemmOutTensor.get()));
            ret |= mPostKernel->get().setArg(idx++, openCLBuffer(mResource->mBias.get()));
            ret |= mPostKernel->get().setArg(idx++, openCLBuffer(output));

            MNN_CHECK_CL_SUCCESS(ret, "setArg mConvgemmOptLevel==2 PostKernel");
            mPostLocalWorkSize = localWS2DDefault(mPostGlobalWorkSize, maxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), "transpose_bias", mPostKernel).first;
            mOpenCLBackend->recordKernel2d(mPostKernel, mPostGlobalWorkSize, mPostLocalWorkSize);
            mPostGlobalWorkSize[0] = ROUND_UP(mPostGlobalWorkSize[0], std::max((uint32_t)1, mPostLocalWorkSize[0]));
            mPostGlobalWorkSize[1] = ROUND_UP(mPostGlobalWorkSize[1], std::max((uint32_t)1, mPostLocalWorkSize[1]));
            
            mOpenCLBackend->endRecord(mRecording);
        }
        return NO_ERROR;
    } else if (mResource->mConvGemmOptLevel == 1) {
        // set small tile
        int tileM = 64;
        int tileN = 64;
        int tileK = 8;
        int localM = 16;
        int localN = 16;
        int M = outputShape.at(0);
        int N = outputShape.at(3);
        int K = inputShape.at(3);

        std::set<std::string> buildOptions = mResource->mBuildOptions;;
        buildOptions.emplace(" -DBIAS");

        if(N % 128 == 0) {
            tileN = 128;
            buildOptions.emplace(" -DOPWM=64 -DOPWN=128 -DCPWK=8 -DOPTM=4 -DOPTN=8");
        } else {
            buildOptions.emplace(" -DOPWM=64 -DOPWN=64 -DCPWK=8 -DOPTM=4 -DOPTN=4");
        }

        
        mKernel = mOpenCLBackend->getOpenCLRuntime()->buildKernel("matmul_local_buf", "matmul_local_buf", buildOptions);
        int out_per_thread_m = tileM / localM;
        int out_per_thread_n = tileN / localN;
        
        mGlobalWorkSize = {static_cast<uint32_t>(M/out_per_thread_m), static_cast<uint32_t>(N/out_per_thread_n)};
        mLocalWorkSize = {static_cast<uint32_t>(localM), static_cast<uint32_t>(localN)};

        int idx            = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mKernel->get().setArg(idx++, static_cast<int>(M));
        ret |= mKernel->get().setArg(idx++, static_cast<int>(N));
        ret |= mKernel->get().setArg(idx++, static_cast<int>(K));
        ret |= mKernel->get().setArg(idx++, openCLBuffer(input));
        ret |= mKernel->get().setArg(idx++, openCLBuffer(mResource->mFilter.get()));
        ret |= mKernel->get().setArg(idx++, openCLBuffer(mResource->mBias.get()));
        ret |= mKernel->get().setArg(idx++, openCLBuffer(output));

        MNN_CHECK_CL_SUCCESS(ret, "setArg Conv1x1Buf mConvgemmOptLevel==1 Kernel Select");
    } else if (mResource->mConv1x1Opt) {
    
        int tileN = 32;
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
            ret |= kernel[knl_idx]->get().setArg(idx++, openCLBuffer(mResource->mFilter.get()));
            ret |= kernel[knl_idx]->get().setArg(idx++, openCLBuffer(mResource->mBias.get()));
            ret |= kernel[knl_idx]->get().setArg(idx++, openCLBuffer(output));
            ret |= kernel[knl_idx]->get().setArg(idx++, static_cast<int>(inputChannelBlocks));
            ret |= kernel[knl_idx]->get().setArg(idx++, height);
            ret |= kernel[knl_idx]->get().setArg(idx++, width);
            ret |= kernel[knl_idx]->get().setArg(idx++, UP_DIV(outChannel, 4));
            ret |= kernel[knl_idx]->get().setArg(idx++, ROUND_UP(outChannel, tileN));

            MNN_CHECK_CL_SUCCESS(ret, "setArg Conv1x1Buf Kernel Select");

            std::pair<std::vector<uint32_t>, int> retTune;
            retTune = localWS2DDefault(globalWorkSize[knl_idx], maxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), kernelName[knl_idx] + info, kernel[knl_idx]);
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
        ret |= mKernel->get().setArg(idx++, openCLBuffer(mResource->mFilter.get()));
        ret |= mKernel->get().setArg(idx++, openCLBuffer(mResource->mBias.get()));
        ret |= mKernel->get().setArg(idx++, openCLBuffer(output));
        ret |= mKernel->get().setArg(idx++, static_cast<int>(inputChannelBlocks));
        ret |= mKernel->get().setArg(idx++, height);
        ret |= mKernel->get().setArg(idx++, width);
        ret |= mKernel->get().setArg(idx++, UP_DIV(outChannel, 4));
        ret |= mKernel->get().setArg(idx++, ROUND_UP(outChannel, tileN));
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
    if (mPreKernel) {
        cl::Event event0;
        runKernel2D(mPreKernel, mPreGlobalWorkSize, mPreLocalWorkSize, mOpenCLBackend->getOpenCLRuntime(), &event0);
        mOpenCLBackend->getOpenCLRuntime()->pushEvent({"ConvBuf2D-gemm2-0", event0});
    }
    cl::Event event;
    runKernel2D(mKernel, mGlobalWorkSize, mLocalWorkSize, mOpenCLBackend->getOpenCLRuntime(), &event);
    std::string name = "ConvBuf2D";
    std::string b = std::to_string(inputs[0]->batch());
    std::string ci = std::to_string(inputs[0]->channel());
    std::string hi = std::to_string(inputs[0]->height());
    std::string wi = std::to_string(inputs[0]->width());
    std::string co = std::to_string(outputs[0]->channel());
    std::string ho = std::to_string(outputs[0]->height());
    std::string wo = std::to_string(outputs[0]->width());
    std::string kh = std::to_string(mResource->mKernelHeight);
    std::string kw = std::to_string(mResource->mKernelWidth);
    std::string total = std::to_string(1.0 / 1000000 * inputs[0]->batch() * inputs[0]->channel() * outputs[0]->channel() * outputs[0]->height() * outputs[0]->width() * mResource->mKernelHeight * mResource->mKernelWidth);
    if (mResource->mConvGemmOptLevel > 0) {
        std::string m = std::to_string(outputs[0]->width() * outputs[0]->height() * inputs[0]->batch());
        name += "-gemm";
        name += std::to_string(mResource->mConvGemmOptLevel) + "-m" + m + "n" + co + "k" + ci;
    } else if (mResource->mConv1x1Opt) {
        name += "-conv1x1";
        name += "-b" + b + "ci" + ci + "hi" + hi + "wi" + wi + "co" + co;
    } else {
        name += "-ori-b" + b + "ci" + ci + "hi" + hi + "wi" + wi + "co" + co+ "ho" + ho + "wo" + wo + "kh" + kh + "kw" + kw;
    }
    name += "-total:" + total + "*10^6";
    mOpenCLBackend->getOpenCLRuntime()->pushEvent({name.c_str(), event});
    if (mPostKernel) {
        cl::Event event2;
        runKernel2D(mPostKernel, mPostGlobalWorkSize, mPostLocalWorkSize, mOpenCLBackend->getOpenCLRuntime(), &event2);
        mOpenCLBackend->getOpenCLRuntime()->pushEvent({"ConvBuf2D-gemm2-2", event2});
    }
#else
    if(mOpenCLBackend->isUseRecordQueue()){
        mOpenCLBackend->addRecord(mRecording, mOpRecordUpdateInfo);
#ifdef LOG_VERBOSE
        MNN_PRINT("End ConvExecution onExecute... \n");
#endif
        return NO_ERROR;
    }
    if (mPreKernel) {
        runKernel2D(mPreKernel, mPreGlobalWorkSize, mPreLocalWorkSize, mOpenCLBackend->getOpenCLRuntime());
    }
    runKernel2D(mKernel, mGlobalWorkSize, mLocalWorkSize, mOpenCLBackend->getOpenCLRuntime());
    if (mPostKernel) {
        runKernel2D(mPostKernel, mPostGlobalWorkSize, mPostLocalWorkSize, mOpenCLBackend->getOpenCLRuntime());
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
                    if ((1 == conv2dParams->quanParameter()->type() || 2 == conv2dParams->quanParameter()->type()) && conv2dParams->quanParameter()->has_scaleInt()) {
                        // Don't support IDST-int8 because of error
                        return nullptr;
                    }
                    return new ConvBufLowMemoryExecution(inputs, outputs, op, backend);
                } else {
                    //MNN_ERROR("OpenCL Conv buf low memory init error. For Opencl Backend, only support low memory mode of int8 or int4 dequantization currently.\n");
                    return nullptr;
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
