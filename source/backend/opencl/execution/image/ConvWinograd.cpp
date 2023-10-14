//
//  ConvWinograd.cpp
//  MNN
//
//  Created by MNN on 2019/01/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/image/ConvWinograd.hpp"
#include <string.h>
#include "core/Backend.hpp"
#include "core/ConvolutionCommon.hpp"
#include "math/WingoradGenerater.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"
#define UNIT 2
#define INTERP 1
namespace MNN {
namespace OpenCL {
bool ConvWinograd::valid(const Convolution2DCommon* common, const Tensor* input, const Tensor* output, int maxWidth, int maxHeight, int limit) {
    if (common->strideX() != 1 || common->strideY() != 1) {
        return false;
    }
    if (common->dilateX() != 1 || common->dilateY() != 1) {
        return false;
    }
    if(common->kernelX() != common->kernelY()) {
        return false;
    }
    if(common->kernelX() != 3 && common->kernelX() != 5){
        return false;
    }
    
    int ic = input->channel();
    int oc = common->outputCount();
    int ow = output->width();
    int oh =output->height();
    int kh = common->kernelX();
    int wUnit = UP_DIV(ow, UNIT);
    int hUnit = UP_DIV(oh, UNIT);
    int alpha  = kh + UNIT - 1;
    int sourceWidth  = UP_DIV(ic, 4) * 4 * wUnit;
    int sourceHeight = alpha * alpha * hUnit;
    int destWidth  = alpha * alpha * wUnit * 4;
    int destHeight = UP_DIV(ic, 4) * hUnit;

    if(sourceWidth > maxWidth || sourceHeight > maxHeight || destWidth > maxWidth || destHeight > maxHeight){
        return false;
    }
    if(ic >= 32 && oc >= 32){
        return true;
    }
    return ((oc * oh * ow) / (ic * kh) <= 5);
}


ConvWinograd::ConvWinograd(const MNN::Convolution2D* op, Backend* backend) : Execution(backend) {
    mOpenCLBackend = static_cast<OpenCLBackend*>(backend);
    mCommon        = op->common();
    MNN_ASSERT((3 == mCommon->kernelY() && 3 == mCommon->kernelX()) ||
               (5 == mCommon->kernelX() && 5 == mCommon->kernelY()));
    MNN_ASSERT(1 == mCommon->strideX() && 1 == mCommon->strideY());
    MNN_ASSERT(1 == mCommon->dilateX() && 1 == mCommon->dilateY());
    auto runTime = mOpenCLBackend->getOpenCLRuntime();
    int ky       = mCommon->kernelY();
    int kx       = mCommon->kernelX();

    int weightSize             = 0;
    const float* filterDataPtr = nullptr;

    std::shared_ptr<MNN::ConvolutionCommon::Int8Common> quanCommon;
    if (nullptr != op->quanParameter()) {
        quanCommon = ConvolutionCommon::load(op, backend, true);
        if (nullptr == quanCommon) {
            MNN_ERROR("Memory not Enough, can't extract IDST Convolution \n");
        }
        if (quanCommon->weightFloat.get() == nullptr) {
            MNN_PRINT("quanCommon->weightFloat.get() == nullptr \n");
        }
        // Back to float
        filterDataPtr = quanCommon->weightFloat.get();
        weightSize    = quanCommon->weightFloat.size();
    }

    if (nullptr == filterDataPtr) {
        weightSize    = op->weight()->size();
        filterDataPtr = op->weight()->data();
    }

    int co     = mCommon->outputCount();
    int ci     = weightSize / co / mCommon->kernelX() / mCommon->kernelY();
    auto coC4  = UP_DIV(co, 4);
    auto ciC4  = UP_DIV(ci, 4);
    auto queue = runTime->commandQueue();

    auto imageChannelType = CL_HALF_FLOAT;
    if (mOpenCLBackend->getPrecision() == BackendConfig::Precision_High) {
        imageChannelType = CL_FLOAT;
    }
    // Create Image
    {
        mBias.reset(new cl::Image2D(runTime->context(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, imageChannelType),
                                    UP_DIV(co, 4), 1, 0, nullptr, nullptr));
        
        int buffer_size = ALIGN_UP4(co);
        if(mOpenCLBackend->getOpenCLRuntime()->isWeightCpuTransHalf()) {
            buffer_size *= sizeof(half_float::half);
        } else {
            buffer_size *= sizeof(float);
        }
        std::shared_ptr<cl::Buffer> biasBuffer(
            new cl::Buffer(runTime->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size));

        cl_int error;
        auto biasC = queue.enqueueMapBuffer(*biasBuffer, CL_TRUE, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &error);
        if(biasC != nullptr && error == CL_SUCCESS){
            if(mOpenCLBackend->getOpenCLRuntime()->isWeightCpuTransHalf()){
                for(int i=0; i<co; i++) {
                    ((half_float::half*)biasC)[i] = (half_float::half)(op->bias()->data()[i]);
                }
                for(int i=co; i<ALIGN_UP4(co); i++) {
                    ((half_float::half*)biasC)[i] = (half_float::half)(0.0f);
                }
            }else{
                ::memset(biasC, 0, buffer_size);
                ::memcpy(biasC, op->bias()->data(), co * sizeof(float));
            }
        }else{
            MNN_ERROR("Map error biasC == nullptr \n");
        }
        queue.enqueueUnmapMemObject(*biasBuffer, biasC);
        copyBufferToImage(runTime, *biasBuffer, *mBias, coC4, 1);

        std::shared_ptr<Tensor> sourceWeight(
            Tensor::create<float>(std::vector<int>{co, ci, ky, kx}, (void*)(filterDataPtr), Tensor::CAFFE));

        int unit       = UNIT;
        int kernelSize = kx;
        Math::WinogradGenerater generator(unit, kernelSize, INTERP);
        int alpha       = unit + kernelSize - 1;
        auto weightDest = generator.allocTransformWeight(sourceWeight.get());
        generator.transformWeight(weightDest.get(), sourceWeight.get());
        auto weightDestSize = weightDest->size();
        
        buffer_size = weightDest->elementSize();
        if(mOpenCLBackend->getOpenCLRuntime()->isWeightCpuTransHalf()) {
            buffer_size *= sizeof(half_float::half);
        } else {
            buffer_size *= sizeof(float);
        }
        cl::Buffer weightBuffer(runTime->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size);
        {
            cl_int error;
            auto weightPtr = queue.enqueueMapBuffer(weightBuffer, CL_TRUE, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &error);
            if(weightPtr != nullptr && error == CL_SUCCESS){
                if(mOpenCLBackend->getOpenCLRuntime()->isWeightCpuTransHalf()){
                    for(int i=0; i<weightDest->elementSize(); i++) {
                        ((half_float::half*)weightPtr)[i] = (half_float::half)(weightDest->host<float>()[i]);
                    }
                }else{
                    ::memcpy(weightPtr, weightDest->host<float>(), buffer_size);
                }
            } else{
                MNN_ERROR("Map error weightPtr == nullptr \n");
            }

            queue.enqueueUnmapMemObject(weightBuffer, weightPtr);
        }
        mWeight.reset(new cl::Image2D(runTime->context(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, imageChannelType),
                                      ciC4 * 4, coC4 * alpha * alpha, 0, nullptr, nullptr));
        copyBufferToImage(runTime, weightBuffer, *mWeight, ciC4 * 4, coC4 * alpha * alpha);
    }
}

ErrorCode ConvWinograd::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];
    mKernelX    = mCommon->kernelX();
    mKernelY    = mCommon->kernelY();
    mStrideX    = mCommon->strideX();
    mStrideY    = mCommon->strideY();
    mPadMode    = mCommon->padMode();
    
    int alpha  = mCommon->kernelX() + UNIT - 1;
    auto wUnit = UP_DIV(output->width(), UNIT);
    auto hUnit = UP_DIV(output->height(), UNIT);
    
    auto pad = ConvolutionCommon::convolutionPad(inputs[0], outputs[0], mCommon);
    const int padY = pad.second;
    const int padX  = pad.first;
    
    auto runTime = mOpenCLBackend->getOpenCLRuntime();
    startRecord(runTime, mRecording);

    auto bn = backend();
    mSource.reset(Tensor::createDevice<float>(
        std::vector<int>{alpha * alpha, input->channel(), hUnit, wUnit}, Tensor::CAFFE_C4));
    mDest.reset(Tensor::createDevice<float>(
        std::vector<int>{UP_DIV(output->channel(), 4), wUnit * 4, hUnit, alpha * alpha}, Tensor::CAFFE_C4));

    bn->onAcquireBuffer(mSource.get(), Backend::DYNAMIC);
    bn->onAcquireBuffer(mDest.get(), Backend::DYNAMIC);
    bn->onReleaseBuffer(mSource.get(), Backend::DYNAMIC);
    bn->onReleaseBuffer(mDest.get(), Backend::DYNAMIC);

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
    for(int i = 0; i < input->batch(); i++) {
        char format[20];
        ::memset(format, 0, sizeof(format));
        sprintf(format, "%d_%d_%d", UNIT, mKernelX, INTERP);
        auto formatStr = std::string(format);
        mSourceTransform[i] =
            runTime->buildKernel("winogradTransformSource" + formatStr,
                                 "winogradTransformSource", basic);
        mMaxWGS_S[i] = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(mSourceTransform[i]));
        {
            std::set<std::string> buildOptions = basic;
            if (mCommon->relu()) {
                buildOptions.emplace("-DRELU");
            }
            if (mCommon->relu6()) {
                buildOptions.emplace("-DRELU6");
            }
            mDestTransform[i] =
                runTime->buildKernel("winogradTransformDest" + formatStr,
                                     "winogradTransformDest", buildOptions);
            mMaxWGS_D[i] = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(mDestTransform[i]));
        }
        mMaxWGS_M[i] = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(mMatMul[i]));
    }
    
    mGWS_S.resize(total_num);
    mGWS_D.resize(total_num);
    mGWS_M.resize(total_num);
    mLWS_S.resize(total_num);
    mLWS_D.resize(total_num);
    mLWS_M.resize(total_num);

    for (int b = 0; b < input->batch(); ++b) {
        cl_int ret = CL_SUCCESS;
        ret |= mSourceTransform[b].setArg(0, openCLImage(input));
        ret |= mSourceTransform[b].setArg(1, openCLImage(mSource.get()));
        ret |= mSourceTransform[b].setArg(2, wUnit);
        ret |= mSourceTransform[b].setArg(3, hUnit);
        ret |= mSourceTransform[b].setArg(4, padX);
        ret |= mSourceTransform[b].setArg(5, padY);
        ret |= mSourceTransform[b].setArg(6, input->width());
        ret |= mSourceTransform[b].setArg(7, input->height());
        ret |= mSourceTransform[b].setArg(8, icC4);
        ret |= mSourceTransform[b].setArg(9, b);


        ret |= mDestTransform[b].setArg(0, openCLImage(mDest.get()));
        ret |= mDestTransform[b].setArg(1, *mBias);
        ret |= mDestTransform[b].setArg(2, openCLImage(output));
        ret |= mDestTransform[b].setArg(3, wUnit);
        ret |= mDestTransform[b].setArg(4, hUnit);
        ret |= mDestTransform[b].setArg(5, output->width());
        ret |= mDestTransform[b].setArg(6, output->height());
        ret |= mDestTransform[b].setArg(7, ocC4);
        ret |= mDestTransform[b].setArg(8, b);
        MNN_CHECK_CL_SUCCESS(ret, "setArg ConvWinogradExecution");

        /*Source Transform*/
        {
            mGWS_S[b] = {static_cast<uint32_t>(wUnit * hUnit), static_cast<uint32_t>(icC4)};
            std::string kernelName = "winogradTransformSource";
            mLWS_S[b] = localWS2DDefault(mGWS_S[b], mMaxWGS_S[b], mOpenCLBackend->getOpenCLRuntime(), kernelName, mSourceTransform[b]).first;
            recordKernel2d(mSourceTransform[b], mGWS_S[b], mLWS_S[b], mOpenCLBackend->getOpenCLRuntime());
        }

        /*MatMul*/
        {
            const int total_kernel                     = 2;
            const std::string kernelName[total_kernel] = {"gemmWinograd", "gemmWinogradW2"};
            int itemW[total_kernel]                    = {4, 8};
            auto gemmHeight = ocC4;
            int actual_kernel = total_kernel;
            
            cl::Kernel kernel[total_kernel];
            std::vector<uint32_t> globalWorkSize[total_kernel];
            std::vector<uint32_t> localWorkSize[total_kernel];
            std::pair<uint32_t, int> min_cost(UINT_MAX, 0); //(min_time, min_index)
            for (int knl_idx = 0; knl_idx < actual_kernel; knl_idx++) {
                cl_int ret = CL_SUCCESS;
                kernel[knl_idx] = mOpenCLBackend->getOpenCLRuntime()->buildKernel("gemm", kernelName[knl_idx], basic);
                uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(kernel[knl_idx]));

                globalWorkSize[knl_idx] = {static_cast<uint32_t>(UP_DIV(wUnit, itemW[knl_idx]) * hUnit), static_cast<uint32_t>(alpha * alpha * ocC4)};
                ret |= kernel[knl_idx].setArg(0, openCLImage(mSource.get()));
                ret |= kernel[knl_idx].setArg(1, *mWeight);
                ret |= kernel[knl_idx].setArg(2, openCLImage(mDest.get()));
                ret |= kernel[knl_idx].setArg(3, wUnit);
                ret |= kernel[knl_idx].setArg(4, hUnit);
                ret |= kernel[knl_idx].setArg(5, ocC4);
                ret |= kernel[knl_idx].setArg(6, icC4);
                ret |= kernel[knl_idx].setArg(7, alpha*alpha);
                MNN_CHECK_CL_SUCCESS(ret, "setArg ConvWinogradExecution gemm");

                std::pair<std::vector<uint32_t>, uint32_t> retTune;
                retTune = localWS2DDefault(globalWorkSize[knl_idx], maxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), kernelName[knl_idx], kernel[knl_idx]);
                // printf("gemm %d, %d\n", knl_idx, retTune.second);
                if (min_cost.first > retTune.second) {
                    min_cost.first  = retTune.second;
                    min_cost.second = knl_idx;
                    mLWS_M[b]       = {retTune.first[0], retTune.first[1]};
                }
            }
            cl_int ret = CL_SUCCESS;
            int min_index = min_cost.second;
            //printf("gemm min_index = %d  %d\n", min_index, min_cost.first);
            mMatMul[b] = runTime->buildKernel("gemm", kernelName[min_index], basic);
            
            ret |= mMatMul[b].setArg(0, openCLImage(mSource.get()));
            ret |= mMatMul[b].setArg(1, *mWeight);
            ret |= mMatMul[b].setArg(2, openCLImage(mDest.get()));
            ret |= mMatMul[b].setArg(3, wUnit);
            ret |= mMatMul[b].setArg(4, hUnit);
            ret |= mMatMul[b].setArg(5, ocC4);
            ret |= mMatMul[b].setArg(6, icC4);
            ret |= mMatMul[b].setArg(7, alpha*alpha);
            MNN_CHECK_CL_SUCCESS(ret, "setArg ConvWinogradExecution gemm");
            mGWS_M[b] = {globalWorkSize[min_index][0], globalWorkSize[min_index][1]};
            recordKernel2d(mMatMul[b], mGWS_M[b], mLWS_M[b], mOpenCLBackend->getOpenCLRuntime());
        }

        // Dest Transform
        {
            mGWS_D[b] = {static_cast<uint32_t>(wUnit*hUnit), static_cast<uint32_t>(ocC4)};
            std::string kernelName = "winogradTransformDest";
            mLWS_D[b] = localWS2DDefault(mGWS_D[b], mMaxWGS_D[b], mOpenCLBackend->getOpenCLRuntime(), kernelName, mDestTransform[b]).first;
            recordKernel2d(mDestTransform[b], mGWS_D[b], mLWS_D[b], mOpenCLBackend->getOpenCLRuntime());
        }
    }
    endRecord(runTime, mRecording);
    
    return NO_ERROR;
}

ErrorCode ConvWinograd::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];

    #ifndef ENABLE_OPENCL_TIME_PROFILER
    if(mOpenCLBackend->getOpenCLRuntime()->isUseRecordQueue()){
        if(mOpenCLBackend->getOpenCLRuntime()->isDevideOpRecord())
            mOpenCLBackend->getOpenCLRuntime()->getRecordings()->emplace_back(mRecording);
        return NO_ERROR;
    }
    #endif
    for (int b = 0; b < input->batch(); ++b) {
        /*Source Transform*/
        {
        #ifdef ENABLE_OPENCL_TIME_PROFILER
            cl::Event event;
            runKernel2D(mSourceTransform[b], mGWS_S[b], mLWS_S[b],
                        mOpenCLBackend->getOpenCLRuntime(), &event);
            
            mOpenCLBackend->getOpenCLRuntime()->pushEvent({"ConvWino0", event});
            #else
                runKernel2D(mSourceTransform[b], mGWS_S[b], mLWS_S[b],
                        mOpenCLBackend->getOpenCLRuntime());
            #endif
        }

        /*MatMul*/
        {
        #ifdef ENABLE_OPENCL_TIME_PROFILER
            cl::Event event;
            runKernel2D(mMatMul[b], mGWS_M[b], mLWS_M[b],
                        mOpenCLBackend->getOpenCLRuntime(), &event);
            
            mOpenCLBackend->getOpenCLRuntime()->pushEvent({"ConvWino1", event});
#else
            runKernel2D(mMatMul[b], mGWS_M[b], mLWS_M[b],
                        mOpenCLBackend->getOpenCLRuntime());
#endif
        }

        // Dest Transform
        {
        #ifdef ENABLE_OPENCL_TIME_PROFILER
            cl::Event event;
            runKernel2D(mDestTransform[b], mGWS_D[b], mLWS_D[b],
                        mOpenCLBackend->getOpenCLRuntime(), &event);
            
            mOpenCLBackend->getOpenCLRuntime()->pushEvent({"ConvWino2", event});
        #else
            runKernel2D(mDestTransform[b], mGWS_D[b], mLWS_D[b],
                        mOpenCLBackend->getOpenCLRuntime());
        #endif
        }
    }

    return NO_ERROR;
}

} // namespace OpenCL
} // namespace MNN
