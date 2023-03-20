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
        quanCommon = ConvolutionCommon::load(op->quanParameter(), true);
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
        mMatMul[i] = runTime->buildKernel("gemm", "gemmWinograd", basic);
        mMaxWGS_M[i] = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(mMatMul[i]));
    }
    
    mGWS_S.resize(total_num);
    mGWS_D.resize(total_num);
    mGWS_M.resize(total_num);
    mLWS_S.resize(total_num);
    mLWS_D.resize(total_num);
    mLWS_M.resize(total_num);

    for (int b = 0; b < input->batch(); ++b) {
        mSourceTransform[b].setArg(0, openCLImage(input));
        mSourceTransform[b].setArg(1, openCLImage(mSource.get()));
        mSourceTransform[b].setArg(2, wUnit);
        mSourceTransform[b].setArg(3, hUnit);
        mSourceTransform[b].setArg(4, padX);
        mSourceTransform[b].setArg(5, padY);
        mSourceTransform[b].setArg(6, input->width());
        mSourceTransform[b].setArg(7, input->height());
        mSourceTransform[b].setArg(8, icC4);
        mSourceTransform[b].setArg(9, b);

        mMatMul[b].setArg(0, openCLImage(mSource.get()));
        mMatMul[b].setArg(1, *mWeight);
        mMatMul[b].setArg(2, openCLImage(mDest.get()));
        mMatMul[b].setArg(3, wUnit);
        mMatMul[b].setArg(4, hUnit);
        mMatMul[b].setArg(5, ocC4);
        mMatMul[b].setArg(6, icC4);
        mMatMul[b].setArg(7, alpha*alpha);

        mDestTransform[b].setArg(0, openCLImage(mDest.get()));
        mDestTransform[b].setArg(1, *mBias);
        mDestTransform[b].setArg(2, openCLImage(output));
        mDestTransform[b].setArg(3, wUnit);
        mDestTransform[b].setArg(4, hUnit);
        mDestTransform[b].setArg(5, output->width());
        mDestTransform[b].setArg(6, output->height());
        mDestTransform[b].setArg(7, ocC4);
        mDestTransform[b].setArg(8, b);

        /*Source Transform*/
        {
            mGWS_S[b] = {static_cast<uint32_t>(wUnit * hUnit), static_cast<uint32_t>(icC4)};
            std::string kernelName = "winogradTransformSource";
            mLWS_S[b] = localWS2DDefault(mGWS_S[b], mMaxWGS_S[b], mOpenCLBackend->getOpenCLRuntime(), kernelName, mSourceTransform[b]).first;
        }

        /*MatMul*/
        {
            auto gemmHeight = ocC4;
            mGWS_M[b] = {static_cast<uint32_t>(UP_DIV(wUnit, 4) * hUnit), static_cast<uint32_t>(alpha * alpha * ocC4)};
            std::string kernelName = "gemmWinograd";
            mLWS_M[b] = localWS2DDefault(mGWS_M[b], mMaxWGS_M[b], mOpenCLBackend->getOpenCLRuntime(), kernelName, mMatMul[b]).first;
        }

        // Dest Transform
        {
            mGWS_D[b] = {static_cast<uint32_t>(wUnit*hUnit), static_cast<uint32_t>(ocC4)};
            std::string kernelName = "winogradTransformDest";
            mLWS_D[b] = localWS2DDefault(mGWS_D[b], mMaxWGS_D[b], mOpenCLBackend->getOpenCLRuntime(), kernelName, mDestTransform[b]).first;
        }
    }
    
    return NO_ERROR;
}

ErrorCode ConvWinograd::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];

    #ifdef ENABLE_OPENCL_TIME_PROFILER
    int costTime = 0;
    #endif
    for (int b = 0; b < input->batch(); ++b) {
        /*Source Transform*/
        {
        #ifdef ENABLE_OPENCL_TIME_PROFILER
            cl::Event event;
            runKernel2D(mSourceTransform[b], mGWS_S[b], mLWS_S[b],
                        mOpenCLBackend->getOpenCLRuntime(), &event);
                    
            int costTime0 = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
            costTime += costTime0;
            MNN_PRINT("kernel cost:%d    us ConvWino0\n",costTime0);
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
                            
            int costTime1 = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
            costTime += costTime1;
            MNN_PRINT("kernel cost:%d    us ConvWino1\n",costTime1);
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
                    
            int costTime2 = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
            costTime += costTime2;
            MNN_PRINT("kernel cost:%d    us ConvWino2\n",costTime2);
        #else
            runKernel2D(mDestTransform[b], mGWS_D[b], mLWS_D[b],
                        mOpenCLBackend->getOpenCLRuntime());
        #endif
        }
    }
    #ifdef ENABLE_OPENCL_TIME_PROFILER
    MNN_PRINT("kernel cost:%d    us ConvWino total\n",costTime);
    #endif

    return NO_ERROR;
}

} // namespace OpenCL
} // namespace MNN
