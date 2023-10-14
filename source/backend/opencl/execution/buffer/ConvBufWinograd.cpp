//
//  ConvBufWinograd.cpp
//  MNN
//
//  Created by MNN on 2019/01/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED

#include "backend/opencl/execution/buffer/ConvBufWinograd.hpp"
#include "core/Backend.hpp"
#include "core/ConvolutionCommon.hpp"
#include "math/WingoradGenerater.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"

#define UNIT 2
#define INTERP 1
namespace MNN {
namespace OpenCL {
bool ConvBufWinograd::valid(const Convolution2DCommon* common, const Tensor* input, const Tensor* output, bool isIntel, int limit) {
    if (common->strideX() != 1 || common->strideY() != 1) {
        return false;
    }
    if (common->dilateX() != 1 || common->dilateY() != 1) {
        return false;
    }
    if(common->kernelX() != 3 || common->kernelY() != 3){
        return false;
    }
    if (isIntel) {
        return input->width() * input->height() <= 4096;
    }
    if (output->channel() > 512) {
        return false;
    }
    const int input_channel_limit = output->channel() <= 64 ? 1024 : 512;
    if(input->channel() < 32 || input->channel() > input_channel_limit){
        return false;
    }
    return (input->width() <= 32 && input->height() <= 32);
}

ConvBufWinograd::ConvBufWinograd(const MNN::Convolution2D* op, Backend* backend) : Execution(backend) {
    mOpenCLBackend = static_cast<OpenCLBackend*>(backend);
    mCommon        = op->common();
    MNN_ASSERT((3 == mCommon->kernelY() && 3 == mCommon->kernelX()));
    MNN_ASSERT(1 == mCommon->strideX() && 1 == mCommon->strideY());
    MNN_ASSERT(1 == mCommon->dilateX() && 1 == mCommon->dilateY());
    auto runTime = mOpenCLBackend->getOpenCLRuntime();
    int ky       = mCommon->kernelY();
    int kx       = mCommon->kernelX();

    int weightSize             = 0;
    const float* filterDataPtr = nullptr;
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
    ConvolutionCommon::getConvParameters(&quanCommon, backend, op, &filterDataPtr, &weightSize);

    int oc     = mCommon->outputCount();
    int ic     = weightSize / oc / mCommon->kernelX() / mCommon->kernelY();
    auto ocC4  = UP_DIV(oc, 4);
    auto icC4  = UP_DIV(ic, 4);
    auto queue = runTime->commandQueue();

    auto imageChannelType = CL_HALF_FLOAT;
    if (mOpenCLBackend->getPrecision() == BackendConfig::Precision_High) {
        imageChannelType = CL_FLOAT;
    }
    // Create Buffer Object
#ifdef MNN_SUPPORT_INTEL_SUBGROUP
    mUseSubgroup = runTime->isSupportedIntelSubgroup();
    if (mUseSubgroup) {
        // create buffer for intel subgroup
        cl_int ret_code;
        size_t bias_element = ALIGN_UP4(oc);
        size_t buffer_size;
        if(mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16()) {
            buffer_size = bias_element * sizeof(half_float::half);
        } else {
            buffer_size = bias_element * sizeof(float);
        }
        
        mBias.reset(Tensor::createDevice<float>({1, 1, 1, (int)ALIGN_UP4(oc)}));
        mOpenCLBackend->onAcquireBuffer(mBias.get(), Backend::STATIC);
        cl::Buffer &bias_buffer = *(cl::Buffer *)mBias->buffer().device;

        auto bias_ptr = queue.enqueueMapBuffer(bias_buffer, CL_TRUE, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &ret_code);
        if(bias_ptr == nullptr || ret_code) {
            MNN_ERROR("clBuffer map error!\n");
        }
        ::memset(bias_ptr, 0, buffer_size);
        if(mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16()) {
            for(int i=0; i<oc; i++) {
                ((half_float::half *)bias_ptr)[i] = (half_float::half)op->bias()->data()[i];
            }
        } else {
            ::memcpy(bias_ptr, op->bias()->data(), oc*sizeof(float));
        }
        queue.enqueueUnmapMemObject(bias_buffer, bias_ptr);


        auto ocC16 = UP_DIV(oc, 16);
        auto icC16 = UP_DIV(ic, 16);
        std::shared_ptr<Tensor> sourceWeight(
            Tensor::create<float>(std::vector<int>{oc, ic, ky, kx}, (void*)(filterDataPtr), Tensor::CAFFE));

        int unit       = UNIT;
        int kernelSize = kx;
        Math::WinogradGenerater generator(unit, kernelSize, INTERP);
        int alpha = unit + kernelSize - 1;
        auto weightDest = generator.allocTransformWeight(sourceWeight.get(), 16, 16);
        generator.transformWeight(weightDest.get(), sourceWeight.get());
        auto weightDestSize = weightDest->size();

        buffer_size = weightDest->elementSize();
        if (mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16()) {
            buffer_size *= sizeof(half_float::half);
        } else {
            buffer_size *= sizeof(float);
        }

        mWeight.reset(Tensor::createDevice<float>({alpha * alpha, ocC16, icC16, 16 * 16}, Tensor::CAFFE_C4)); // NHWC
        mOpenCLBackend->onAcquireBuffer(mWeight.get(), Backend::STATIC);

        cl::Buffer& weightBuffer = *(cl::Buffer*)mWeight->buffer().device;

        auto weight_ptr =
            queue.enqueueMapBuffer(weightBuffer, CL_TRUE, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &ret_code);
        if (weight_ptr != nullptr && ret_code == CL_SUCCESS) {
            if (mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16()) {
                for (int i = 0; i < weightDest->elementSize(); i++) {
                    ((half_float::half*)weight_ptr)[i] = (half_float::half)(weightDest->host<float>()[i]);
                }
            } else {
                ::memcpy(weight_ptr, weightDest->host<float>(), buffer_size);
            }
        } else {
            MNN_ERROR("Map error weightPtr == nullptr \n");
        }

        queue.enqueueUnmapMemObject(weightBuffer, weight_ptr);
    }else
#endif /* MNN_SUPPORT_INTEL_SUBGROUP */    
    {
        cl_int ret_code;
        size_t bias_element = ALIGN_UP4(oc);
        size_t buffer_size;
        if(mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16()) {
            buffer_size = bias_element * sizeof(half_float::half);
        } else {
            buffer_size = bias_element * sizeof(float);
        }
        
        mBias.reset(Tensor::createDevice<float>({1, 1, 1, (int)ALIGN_UP4(oc)}));
        mOpenCLBackend->onAcquireBuffer(mBias.get(), Backend::STATIC);
        cl::Buffer &bias_buffer = *(cl::Buffer *)mBias->buffer().device;

        auto bias_ptr = queue.enqueueMapBuffer(bias_buffer, CL_TRUE, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &ret_code);
        if(bias_ptr == nullptr || ret_code) {
            MNN_ERROR("clBuffer map error!\n");
        }
        ::memset(bias_ptr, 0, buffer_size);
        if(mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16()) {
            for(int i=0; i<oc; i++) {
                ((half_float::half *)bias_ptr)[i] = (half_float::half)op->bias()->data()[i];
            }
        } else {
            ::memcpy(bias_ptr, op->bias()->data(), oc*sizeof(float));
        }
        queue.enqueueUnmapMemObject(bias_buffer, bias_ptr);
        

        std::shared_ptr<Tensor> sourceWeight(
            Tensor::create<float>(std::vector<int>{oc, ic, ky, kx}, (void*)(filterDataPtr), Tensor::CAFFE));

        int unit       = UNIT;
        int kernelSize = kx;
        Math::WinogradGenerater generator(unit, kernelSize, INTERP);
        int alpha       = unit + kernelSize - 1;
        auto weightDest = generator.allocTransformWeight(sourceWeight.get());
        generator.transformWeight(weightDest.get(), sourceWeight.get());
        auto weightDestSize = weightDest->size();
        
        buffer_size = weightDest->elementSize();
        if(mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16()) {
            buffer_size *= sizeof(half_float::half);
        } else {
            buffer_size *= sizeof(float);
        }
        
        mWeight.reset(Tensor::createDevice<float>({1, ocC4 * alpha * alpha, icC4 * 4, 4}));//NHWC
        mOpenCLBackend->onAcquireBuffer(mWeight.get(), Backend::STATIC);
        
        cl::Buffer &weightBuffer = *(cl::Buffer *)mWeight->buffer().device;

        auto weight_ptr = queue.enqueueMapBuffer(weightBuffer, CL_TRUE, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &ret_code);
        if(weight_ptr != nullptr && ret_code == CL_SUCCESS){
            if(mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16()){
                for(int i=0; i<weightDest->elementSize(); i++) {
                    ((half_float::half*)weight_ptr)[i] = (half_float::half)(weightDest->host<float>()[i]);
                }
            }else{
                ::memcpy(weight_ptr, weightDest->host<float>(), buffer_size);
            }
        } else{
            MNN_ERROR("Map error weightPtr == nullptr \n");
        }

        queue.enqueueUnmapMemObject(weightBuffer, weight_ptr);
        
    }
}

ConvBufWinograd::~ConvBufWinograd() {
    mOpenCLBackend->onReleaseBuffer(mWeight.get(), Backend::STATIC);
    mOpenCLBackend->onReleaseBuffer(mBias.get(), Backend::STATIC);
}

ErrorCode ConvBufWinograd::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];
    mKernelX    = mCommon->kernelX();
    mKernelY    = mCommon->kernelY();
    mStrideX    = mCommon->strideX();
    mStrideY    = mCommon->strideY();
    
    int alpha  = mKernelX + UNIT - 1;
    auto wUnit = UP_DIV(output->width(), UNIT);
    auto hUnit = UP_DIV(output->height(), UNIT);
    
    auto pad = ConvolutionCommon::convolutionPad(input, output, mCommon);
    int padY = pad.second;
    int padX = pad.first;

    auto runTime = mOpenCLBackend->getOpenCLRuntime();

#ifdef MNN_SUPPORT_INTEL_SUBGROUP
    if (mUseSubgroup) {
        
        mSource.reset(Tensor::createDevice<float>(std::vector<int>{alpha * alpha, UP_DIV(input->channel(), 16), ROUND_UP(wUnit * hUnit, 8), 16}, Tensor::CAFFE_C4));
        mDest.reset(Tensor::createDevice<float>(std::vector<int>{alpha * alpha, UP_DIV(output->channel(), 16), ROUND_UP(wUnit * hUnit, 8), 16}, Tensor::CAFFE_C4));

        mOpenCLBackend->onAcquireBuffer(mSource.get(), Backend::DYNAMIC);
        mOpenCLBackend->onAcquireBuffer(mDest.get(), Backend::DYNAMIC);
        mOpenCLBackend->onReleaseBuffer(mSource.get(), Backend::DYNAMIC);
        mOpenCLBackend->onReleaseBuffer(mDest.get(), Backend::DYNAMIC);

        auto icC4  = UP_DIV(input->channel(), 4);
        auto icC16 = UP_DIV(input->channel(), 16);
        auto ocC4  = UP_DIV(output->channel(), 4);
        auto ocC16     = UP_DIV(output->channel(), 16);
        auto inputpad  = TensorUtils::getDescribe(input)->mPads;
        auto outputpad = TensorUtils::getDescribe(output)->mPads;
        int in_c_pack  = TensorUtils::getTensorChannelPack(input);
        int out_c_pack = TensorUtils::getTensorChannelPack(output);

        uint32_t total_num = input->batch();
        mSourceTransform.resize(total_num);
        mMatMul.resize(total_num);
        mDestTransform.resize(total_num);
        mMaxWGS_S.resize(total_num);
        mMaxWGS_D.resize(total_num);
        mMaxWGS_M.resize(total_num);

        std::set<std::string> basic;
        std::string srcTranseKernelname = "_c16_c16";
        std::string dstTranseKernelname = "_c16_c16";
        if (in_c_pack == 4) {
            srcTranseKernelname = "_c4_c16";
        }
        if (out_c_pack == 4) {
            dstTranseKernelname = "_c16_c4";
        }
        /*Create Kernel*/
        for (int i = 0; i < total_num; i++) {
            char format[20];
            ::memset(format, 0, sizeof(format));
            sprintf(format, "%d_%d_%d", UNIT, mKernelX, INTERP);
            auto formatStr = std::string(format);
            mSourceTransform[i] = runTime->buildKernel("winogradTransform_subgroup_buf", "winoTransSrcBuf" + formatStr + srcTranseKernelname, basic);
            mMaxWGS_S[i] = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(mSourceTransform[i]));
            {
                std::set<std::string> buildOptions = basic;
                if (mCommon->relu()) {
                    buildOptions.emplace("-DRELU");
                }
                if (mCommon->relu6()) {
                    buildOptions.emplace("-DRELU6");
                }
                if (output->width() % 2 != 0) {
                    buildOptions.emplace("-DOUTPUT_LEFTOVERS");
                }
                mDestTransform[i] = runTime->buildKernel("winogradTransform_subgroup_buf", "winoTransDstBuf" + formatStr + dstTranseKernelname, buildOptions);
                mMaxWGS_D[i] = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(mDestTransform[i]));
            }
        }

        mGWS_S.resize(total_num);
        mGWS_D.resize(total_num);
        mGWS_M.resize(total_num);
        mLWS_S.resize(total_num);
        mLWS_D.resize(total_num);
        mLWS_M.resize(total_num);

        for (int b = 0; b < input->batch(); ++b) {
            int hCount = hUnit;
            int wCount = wUnit;
            int width_pack = ROUND_UP(hCount * wCount, 8);

            // Source Transform
            {
                mGWS_S[b] = {static_cast<uint32_t>(wCount * hCount), static_cast<uint32_t>(input->channel())};
                int index = 0;
                cl_int ret = CL_SUCCESS;
                ret |= mSourceTransform[b].setArg(index++, mGWS_S[b][0]);
                ret |= mSourceTransform[b].setArg(index++, mGWS_S[b][1]);
                ret |= mSourceTransform[b].setArg(index++, openCLBuffer(input));
                ret |= mSourceTransform[b].setArg(index++, openCLBuffer(mSource.get()));
                ret |= mSourceTransform[b].setArg(index++, wCount);
                ret |= mSourceTransform[b].setArg(index++, hCount);
                ret |= mSourceTransform[b].setArg(index++, padX);
                ret |= mSourceTransform[b].setArg(index++, padY);
                ret |= mSourceTransform[b].setArg(index++, input->width());
                ret |= mSourceTransform[b].setArg(index++, input->height());
                ret |= mSourceTransform[b].setArg(index++, icC4);
                ret |= mSourceTransform[b].setArg(index++, icC16);
                ret |= mSourceTransform[b].setArg(index++, width_pack);
                ret |= mSourceTransform[b].setArg(index++, b);
                ret |= mSourceTransform[b].setArg(index++, static_cast<uint32_t>(inputpad.left));
                ret |= mSourceTransform[b].setArg(index++, static_cast<uint32_t>(inputpad.right));
                MNN_CHECK_CL_SUCCESS(ret, "setArg ConvWinogradBuf Source Trans");

                if (in_c_pack == 4) {
                    mGWS_S[b] = {static_cast<uint32_t>(wCount * hCount), static_cast<uint32_t>(ROUND_UP(input->channel(), 16) / 4)};
                    std::string kernelName = srcTranseKernelname + "_" + std::to_string(mGWS_S[b][0]) + "_" + std::to_string(mGWS_S[b][1]);
                    mLWS_S[b] = localWS2DDefault(mGWS_S[b], mMaxWGS_S[b], mOpenCLBackend->getOpenCLRuntime(), kernelName, mSourceTransform[b]).first;
                } else {
                    mLWS_S[b] = {1, 16};
                }
            }

            // MatMul
            {
                auto gemmHeight = ocC4;
                auto gemmWidth  = wCount * hCount;

                mGWS_M[b] = {static_cast<uint32_t>(UP_DIV(gemmWidth, 8)), static_cast<uint32_t>(ROUND_UP(output->channel(), 16)), static_cast<uint32_t>(alpha * alpha)};
                mLWS_M[b] = {1, 16, 1};
                std::set<std::string> buildOptions = basic;
                mMatMul[b] = mOpenCLBackend->getOpenCLRuntime()->buildKernel("winogradTransform_subgroup_buf", "gemm_buf_intel", buildOptions);

                int index = 0;
                cl_int ret = CL_SUCCESS;
                ret |= mMatMul[b].setArg(index++, openCLBuffer(mSource.get()));
                ret |= mMatMul[b].setArg(index++, openCLBuffer(mWeight.get()));
                ret |= mMatMul[b].setArg(index++, openCLBuffer(mDest.get()));
                ret |= mMatMul[b].setArg(index++, width_pack);
                ret |= mMatMul[b].setArg(index++, ocC16);
                ret |= mMatMul[b].setArg(index++, icC16);
                ret |= mMatMul[b].setArg(index++, alpha * alpha);
                MNN_CHECK_CL_SUCCESS(ret, "setArg ConvWinogradBuf MatMul");
            }

            // Dest Transform
            {
                mGWS_D[b] = {static_cast<uint32_t>(wCount * hCount), static_cast<uint32_t>(output->channel())};

                int index = 0;
                cl_int ret = CL_SUCCESS;
                ret |= mDestTransform[b].setArg(index++, mGWS_D[b][0]);
                ret |= mDestTransform[b].setArg(index++, mGWS_D[b][1]);
                ret |= mDestTransform[b].setArg(index++, openCLBuffer(mDest.get()));
                ret |= mDestTransform[b].setArg(index++, openCLBuffer(mBias.get()));
                ret |= mDestTransform[b].setArg(index++, openCLBuffer(output));
                ret |= mDestTransform[b].setArg(index++, wCount);
                ret |= mDestTransform[b].setArg(index++, hCount);
                ret |= mDestTransform[b].setArg(index++, output->width());
                ret |= mDestTransform[b].setArg(index++, output->height());
                ret |= mDestTransform[b].setArg(index++, ocC4);
                ret |= mDestTransform[b].setArg(index++, ocC16);
                ret |= mDestTransform[b].setArg(index++, width_pack);
                ret |= mDestTransform[b].setArg(index++, b);
                ret |= mDestTransform[b].setArg(index++, static_cast<uint32_t>(outputpad.left));
                ret |= mDestTransform[b].setArg(index++, static_cast<uint32_t>(outputpad.right));
                MNN_CHECK_CL_SUCCESS(ret, "setArg ConvWinogradBuf Dest Trans");

                if (out_c_pack == 4) {
                    mGWS_D[b] = {static_cast<uint32_t>(wCount * hCount), static_cast<uint32_t>(ocC4)};
                    std::string kernelName = dstTranseKernelname + "_" + std::to_string(mGWS_D[b][0]) + "_" + std::to_string(mGWS_D[b][1]);
                    mLWS_D[b] = localWS2DDefault(mGWS_D[b], mMaxWGS_D[b], mOpenCLBackend->getOpenCLRuntime(), kernelName, mDestTransform[b]).first;
                } else {
                    mLWS_D[b] = {1, 16};
                }
            }
        }
    } else 
#endif /* MNN_SUPPORT_INTEL_SUBGROUP */    
    {
        mSource.reset(Tensor::createDevice<float>(
            std::vector<int>{alpha * alpha, input->channel(), ROUND_UP(UP_DIV(wUnit * hUnit, 4), 2), 4},
            Tensor::CAFFE_C4));
        mDest.reset(Tensor::createDevice<float>(
            std::vector<int>{4, wUnit * hUnit, UP_DIV(output->channel(), 4), alpha * alpha}, Tensor::CAFFE_C4));

        mOpenCLBackend->onAcquireBuffer(mSource.get(), Backend::DYNAMIC);
        mOpenCLBackend->onAcquireBuffer(mDest.get(), Backend::DYNAMIC);
        mOpenCLBackend->onReleaseBuffer(mSource.get(), Backend::DYNAMIC);
        mOpenCLBackend->onReleaseBuffer(mDest.get(), Backend::DYNAMIC);

        auto icC4 = UP_DIV(input->channel(), 4);
        auto ocC4 = UP_DIV(output->channel(), 4);

        uint32_t total_num = input->batch();
        mSourceTransform.resize(total_num);
        mMatMul.resize(total_num);
        mDestTransform.resize(total_num);
        mMaxWGS_S.resize(total_num);
        mMaxWGS_D.resize(total_num);
        mMaxWGS_M.resize(total_num);

        std::set<std::string> basic;
        /*Create Kernel*/
        for (int i = 0; i < total_num; i++) {
            char format[20];
            ::memset(format, 0, sizeof(format));
            sprintf(format, "%d_%d_%d", UNIT, mKernelX, INTERP);
            auto formatStr      = std::string(format);
            mSourceTransform[i] = runTime->buildKernel("winogradTransform_buf", "winoTransSrcBuf" + formatStr, basic);
            mMaxWGS_S[i] = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(mSourceTransform[i]));
            {
                std::set<std::string> buildOptions = basic;
                if (mCommon->relu()) {
                    buildOptions.emplace("-DRELU");
                }
                if (mCommon->relu6()) {
                    buildOptions.emplace("-DRELU6");
                }
                mDestTransform[i] = runTime->buildKernel("winogradTransform_buf", "winoTransDstBuf" + formatStr, buildOptions);
                mMaxWGS_D[i] = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(mDestTransform[i]));
            }
        }

        mGWS_S.resize(total_num);
        mGWS_D.resize(total_num);
        mGWS_M.resize(total_num);
        mLWS_S.resize(total_num);
        mLWS_D.resize(total_num);
        mLWS_M.resize(total_num);

        for (int b = 0; b < input->batch(); ++b) {
            int hCount = hUnit;
            int wCount = wUnit;

            // Source Transform
            {
                mGWS_S[b] = {static_cast<uint32_t>(wCount * hCount), static_cast<uint32_t>(icC4)};
                int index = 0;
                cl_int ret = CL_SUCCESS;
                ret |= mSourceTransform[b].setArg(index++, mGWS_S[b][0]);
                ret |= mSourceTransform[b].setArg(index++, mGWS_S[b][1]);
                ret |= mSourceTransform[b].setArg(index++, openCLBuffer(input));
                ret |= mSourceTransform[b].setArg(index++, openCLBuffer(mSource.get()));
                ret |= mSourceTransform[b].setArg(index++, wCount);
                ret |= mSourceTransform[b].setArg(index++, hCount);
                ret |= mSourceTransform[b].setArg(index++, padX);
                ret |= mSourceTransform[b].setArg(index++, padY);
                ret |= mSourceTransform[b].setArg(index++, input->width());
                ret |= mSourceTransform[b].setArg(index++, input->height());
                ret |= mSourceTransform[b].setArg(index++, icC4);
                ret |= mSourceTransform[b].setArg(index++, b);
                MNN_CHECK_CL_SUCCESS(ret, "setArg ConvWinogradBuf SubGroup Source Trans");

                std::string kernelName = "winoTransSrcBuf";
                mLWS_S[b] = localWS2DDefault(mGWS_S[b], mMaxWGS_S[b], mOpenCLBackend->getOpenCLRuntime(), kernelName, mSourceTransform[b]).first;
            }

            // MatMul
            {
                auto gemmHeight = ocC4;
                auto gemmWidth  = UP_DIV(wCount * hCount, 4);

                const int total_kernel                     = 2;
                const std::string kernelName[total_kernel] = {"gemm_buf", "gemm_buf2"};
                int itemW[total_kernel]                    = {1, 2};

                int actual_kernel = total_kernel;
                if (mOpenCLBackend->getOpenCLRuntime()->getCLTuneLevel() == Normal || mOpenCLBackend->getOpenCLRuntime()->getCLTuneLevel() == Fast || mOpenCLBackend->getOpenCLRuntime()->getCLTuneLevel() == None) {
                    actual_kernel = 1;
                }

                cl::Kernel kernel[total_kernel];
                std::vector<uint32_t> globalWorkSize[total_kernel];
                std::vector<uint32_t> localWorkSize[total_kernel];
                std::pair<uint32_t, int> min_cost(UINT_MAX, 0); //(min_time, min_index)
                for (int knl_idx = 0; knl_idx < actual_kernel; knl_idx++) {
                    kernel[knl_idx] = mOpenCLBackend->getOpenCLRuntime()->buildKernel("gemm_buf", kernelName[knl_idx], basic);
                    uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(kernel[knl_idx]));

                    globalWorkSize[knl_idx] = {static_cast<uint32_t>(UP_DIV(gemmWidth, itemW[knl_idx]) * gemmHeight), static_cast<uint32_t>(alpha * alpha)};
                    uint32_t index = 0;
                    cl_int ret = CL_SUCCESS;
                    ret |= kernel[knl_idx].setArg(index++, globalWorkSize[knl_idx][0]);
                    ret |= kernel[knl_idx].setArg(index++, globalWorkSize[knl_idx][1]);
                    ret |= kernel[knl_idx].setArg(index++, openCLBuffer(mSource.get()));
                    ret |= kernel[knl_idx].setArg(index++, openCLBuffer(mWeight.get()));
                    ret |= kernel[knl_idx].setArg(index++, openCLBuffer(mDest.get()));
                    ret |= kernel[knl_idx].setArg(index++, gemmWidth);
                    ret |= kernel[knl_idx].setArg(index++, gemmHeight);
                    ret |= kernel[knl_idx].setArg(index++, icC4);
                    ret |= kernel[knl_idx].setArg(index++, alpha * alpha);
                    MNN_CHECK_CL_SUCCESS(ret, "setArg ConvWinogradBuf SubGroup MatMul Kernel Select");

                    std::pair<std::vector<uint32_t>, uint32_t> retTune;
                    retTune = localWS2DDefault(globalWorkSize[knl_idx], maxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), kernelName[knl_idx], kernel[knl_idx]);
                    // printf("gemm %d, %d\n", knl_idx, retTune.second);
                    if (min_cost.first > retTune.second) {
                        min_cost.first  = retTune.second;
                        min_cost.second = knl_idx;
                        mLWS_M[b]       = {retTune.first[0], retTune.first[1]};
                    }
                }
                int min_index = min_cost.second;
                // mKernel = kernel[min_index];
                mGWS_M[b]  = {globalWorkSize[min_index][0], globalWorkSize[min_index][1]};
                mMatMul[b] = mOpenCLBackend->getOpenCLRuntime()->buildKernel("gemm_buf", kernelName[min_index], basic);

                int index = 0;
                cl_int ret = CL_SUCCESS;
                ret |= mMatMul[b].setArg(index++, mGWS_M[b][0]);
                ret |= mMatMul[b].setArg(index++, mGWS_M[b][1]);
                ret |= mMatMul[b].setArg(index++, openCLBuffer(mSource.get()));
                ret |= mMatMul[b].setArg(index++, openCLBuffer(mWeight.get()));
                ret |= mMatMul[b].setArg(index++, openCLBuffer(mDest.get()));
                ret |= mMatMul[b].setArg(index++, gemmWidth);
                ret |= mMatMul[b].setArg(index++, gemmHeight);
                ret |= mMatMul[b].setArg(index++, icC4);
                ret |= mMatMul[b].setArg(index++, alpha * alpha);
                MNN_CHECK_CL_SUCCESS(ret, "setArg ConvWinogradBuf SubGroup MatMul");
            }

            // Dest Transform
            {
                mGWS_D[b] = {static_cast<uint32_t>(wCount * hCount), static_cast<uint32_t>(ocC4)};

                int index = 0;
                cl_int ret = CL_SUCCESS;
                ret |= mDestTransform[b].setArg(index++, mGWS_D[b][0]);
                ret |= mDestTransform[b].setArg(index++, mGWS_D[b][1]);
                ret |= mDestTransform[b].setArg(index++, openCLBuffer(mDest.get()));
                ret |= mDestTransform[b].setArg(index++, openCLBuffer(mBias.get()));
                ret |= mDestTransform[b].setArg(index++, openCLBuffer(output));
                ret |= mDestTransform[b].setArg(index++, wCount);
                ret |= mDestTransform[b].setArg(index++, hCount);
                ret |= mDestTransform[b].setArg(index++, output->width());
                ret |= mDestTransform[b].setArg(index++, output->height());
                ret |= mDestTransform[b].setArg(index++, ocC4);
                ret |= mDestTransform[b].setArg(index++, b);
                MNN_CHECK_CL_SUCCESS(ret, "setArg ConvWinogradBuf SubGroup Dest Trans");
                
                std::string kernelName = "winoTransDstBuf";
                mLWS_D[b] = localWS2DDefault(mGWS_D[b], mMaxWGS_D[b], mOpenCLBackend->getOpenCLRuntime(), kernelName, mDestTransform[b]).first;
            }
        }
    }
    
    return NO_ERROR;
}

ErrorCode ConvBufWinograd::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];

    for (int b = 0; b < input->batch(); ++b) {
        int index = b;
        /*Source Transform*/
        {
        #ifdef ENABLE_OPENCL_TIME_PROFILER
            cl::Event event;
            runKernel2D(mSourceTransform[index], mGWS_S[index], mLWS_S[index],
                        mOpenCLBackend->getOpenCLRuntime(), &event);
            mOpenCLBackend->getOpenCLRuntime()->pushEvent({"ConvWino0", event});
        #else
            runKernel2D(mSourceTransform[index], mGWS_S[index], mLWS_S[index],
                        mOpenCLBackend->getOpenCLRuntime());
        #endif
        }

        /*MatMul*/
        {
        #ifdef ENABLE_OPENCL_TIME_PROFILER
            cl::Event event;
            if (mUseSubgroup) {
                run3DKernelDefault(mMatMul[index], mGWS_M[index], mLWS_M[index], mOpenCLBackend->getOpenCLRuntime(), &event);
            } else {
                runKernel2D(mMatMul[index], mGWS_M[index], mLWS_M[index], mOpenCLBackend->getOpenCLRuntime(), &event);
            }
            mOpenCLBackend->getOpenCLRuntime()->pushEvent({"ConvWino1", event});
        #else
            if (mUseSubgroup) {
                run3DKernelDefault(mMatMul[index], mGWS_M[index], mLWS_M[index], mOpenCLBackend->getOpenCLRuntime());
            } else {
                runKernel2D(mMatMul[index], mGWS_M[index], mLWS_M[index], mOpenCLBackend->getOpenCLRuntime());
            }
        #endif
        }

        // Dest Transform
        {
        #ifdef ENABLE_OPENCL_TIME_PROFILER
            cl::Event event;
            runKernel2D(mDestTransform[index], mGWS_D[index], mLWS_D[index],
                        mOpenCLBackend->getOpenCLRuntime(), &event);
            mOpenCLBackend->getOpenCLRuntime()->pushEvent({"ConvWino2", event});
        #else
            runKernel2D(mDestTransform[index], mGWS_D[index], mLWS_D[index],
                        mOpenCLBackend->getOpenCLRuntime());
        #endif
        }
    }

    return NO_ERROR;
}

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
