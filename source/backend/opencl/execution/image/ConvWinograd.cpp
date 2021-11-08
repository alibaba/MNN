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
bool ConvWinograd::valid(const Convolution2DCommon* common, const Tensor* input, int limit) {
    if (common->strideX() != 1 || common->strideY() != 1) {
        return false;
    }
    if (common->dilateX() != 1 || common->dilateY() != 1) {
        return false;
    }
    if (input->channel() < 8 || common->outputCount() < 8) {
        return false;
    }

    return (common->kernelX() == 3 && common->kernelY() == 3) || (common->kernelX() == 5 && common->kernelY() == 5);
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

    int maxWidth  = runTime->getMaxImage2DSize()[0];
    int maxHeight = runTime->getMaxImage2DSize()[1];

    int sourceWidth  = UP_DIV(input->channel(), 4) * 4;
    int sourceHeight = alpha * alpha * UP_DIV(wUnit * hUnit, 4);

    int sliceNumber    = 1;
    const int maxSlice = 100;

    if (maxWidth < sourceWidth || maxHeight < sourceHeight) {
        for (int i = 2; i < maxSlice; ++i) {
            int realWidth  = (size_t)UP_DIV(input->channel(), 4) * 4;
            int readHeight = (size_t)alpha * alpha * UP_DIV(UP_DIV(wUnit, i) * UP_DIV(hUnit, i), 4);

            if (realWidth < maxWidth && readHeight < maxHeight) {
                sliceNumber = i;
                break;
            }
        }
    }

    mSliceNumber = sliceNumber;

    int wPiece = UP_DIV(wUnit, sliceNumber);
    int hPiece = UP_DIV(hUnit, sliceNumber);

    auto bn = backend();
    mSource.reset(Tensor::createDevice<float>(
        std::vector<int>{alpha * alpha, input->channel(), UP_DIV(wPiece * hPiece, 4), 4}, Tensor::CAFFE_C4));
    mDest.reset(Tensor::createDevice<float>(
        std::vector<int>{4, wPiece * hPiece, UP_DIV(output->channel(), 4), alpha * alpha}, Tensor::CAFFE_C4));

    bn->onAcquireBuffer(mSource.get(), Backend::DYNAMIC);
    bn->onAcquireBuffer(mDest.get(), Backend::DYNAMIC);
    bn->onReleaseBuffer(mSource.get(), Backend::DYNAMIC);
    bn->onReleaseBuffer(mDest.get(), Backend::DYNAMIC);

    auto icC4 = UP_DIV(input->channel(), 4);
    auto ocC4 = UP_DIV(output->channel(), 4);
    
    uint32_t total_num = input->batch()*mSliceNumber*mSliceNumber;
    mSourceTransform.resize(total_num);
    mMatMul.resize(total_num);
    mDestTransform.resize(total_num);
    mMaxWGS_S.resize(total_num);
    mMaxWGS_D.resize(total_num);
    mMaxWGS_M.resize(total_num);
    
    std::set<std::string> basic;
    /*Create Kernel*/
    for(int i = 0; i < input->batch()*mSliceNumber*mSliceNumber; i++) {
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
        mMatMul[i] = runTime->buildKernel("gemm", "gemm", basic);
        mMaxWGS_M[i] = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(mMatMul[i]));
    }
    
    mGWS_S.resize(total_num);
    mGWS_D.resize(total_num);
    mGWS_M.resize(total_num);
    mLWS_S.resize(total_num);
    mLWS_D.resize(total_num);
    mLWS_M.resize(total_num);

    for (int b = 0; b < input->batch(); ++b) {
        std::vector<int> offsetData;
        offsetData.push_back(0);
        offsetData.push_back(0);

        for (int y = 0; y < mSliceNumber; ++y) {
            int hCount = hPiece;
            if (y == mSliceNumber - 1) {
                hCount = hUnit - (mSliceNumber - 1) * hPiece;
            }
            offsetData[1] = y * hPiece;

            for (int x = 0; x < mSliceNumber; ++x) {
                int wCount = wPiece;
                if (x == mSliceNumber - 1) {
                    wCount = wUnit - (mSliceNumber - 1) * wPiece;
                }
                offsetData[0] = x * wPiece;

                auto dest = mDest.get();
                int index = b*mSliceNumber*mSliceNumber + y*mSliceNumber + x;

                mSourceTransform[index].setArg(0, openCLImage(input));
                mSourceTransform[index].setArg(1, openCLImage(mSource.get()));
                mSourceTransform[index].setArg(4, padX);
                mSourceTransform[index].setArg(5, padY);
                mSourceTransform[index].setArg(6, input->width());
                mSourceTransform[index].setArg(7, input->height());
                mSourceTransform[index].setArg(8, icC4);

                mMatMul[index].setArg(0, openCLImage(mSource.get()));
                mMatMul[index].setArg(1, *mWeight);
                mMatMul[index].setArg(4, ocC4);
                mMatMul[index].setArg(5, icC4);
                mMatMul[index].setArg(6, alpha*alpha);

                mDestTransform[index].setArg(1, *mBias);
                mDestTransform[index].setArg(2, openCLImage(output));
                mDestTransform[index].setArg(5, output->width());
                mDestTransform[index].setArg(6, output->height());
                mDestTransform[index].setArg(7, ocC4);
                
                
                mSourceTransform[index].setArg(2, wCount);
                mSourceTransform[index].setArg(3, hCount);
                mSourceTransform[index].setArg(9, offsetData[0]);
                mSourceTransform[index].setArg(10, offsetData[1]);
                mSourceTransform[index].setArg(11, b);

                auto gemmWidth = UP_DIV(wCount * hCount, 4);
                mMatMul[index].setArg(2, openCLImage(dest));
                mMatMul[index].setArg(3, gemmWidth);

                mDestTransform[index].setArg(0, openCLImage(dest));
                mDestTransform[index].setArg(3, wCount);
                mDestTransform[index].setArg(4, hCount);
                mDestTransform[index].setArg(8, offsetData[0]);
                mDestTransform[index].setArg(9, offsetData[1]);
                mDestTransform[index].setArg(10, b);

                /*Source Transform*/
                {
                    mGWS_S[index] = {static_cast<uint32_t>(wCount * hCount), static_cast<uint32_t>(icC4)};
                    std::string kernelName = "winogradTransformSource";
                    mLWS_S[index] = localWS2DDefault(mGWS_S[index], mMaxWGS_S[index], mOpenCLBackend->getOpenCLRuntime(), kernelName, mSourceTransform[index]).first;
                }

                /*MatMul*/
                {
                    auto gemmHeight = ocC4;
                    mGWS_M[index] = {static_cast<uint32_t>(gemmWidth*gemmHeight), static_cast<uint32_t>(alpha * alpha)};
                    std::string kernelName = "gemm";
                    mLWS_M[index] = localWS2DDefault(mGWS_M[index], mMaxWGS_M[index], mOpenCLBackend->getOpenCLRuntime(), kernelName, mMatMul[index]).first;
                }

                // Dest Transform
                {
                    mGWS_D[index] = {static_cast<uint32_t>(wCount*hCount), static_cast<uint32_t>(ocC4)};
                    std::string kernelName = "winogradTransformDest";
                    mLWS_D[index] = localWS2DDefault(mGWS_D[index], mMaxWGS_D[index], mOpenCLBackend->getOpenCLRuntime(), kernelName, mDestTransform[index]).first;
                }

            }
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
        for (int y = 0; y < mSliceNumber; ++y) {
            for (int x = 0; x < mSliceNumber; ++x) {
                int index = b*mSliceNumber*mSliceNumber + y*mSliceNumber + x;

                /*Source Transform*/
                {
                #ifdef ENABLE_OPENCL_TIME_PROFILER
                    cl::Event event;
                    runKernel2D(mSourceTransform[index], mGWS_S[index], mLWS_S[index],
                                mOpenCLBackend->getOpenCLRuntime(), &event);
                    
                    int costTime0 = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
                    costTime += costTime0;
                    MNN_PRINT("kernel cost:%d    us ConvWino0\n",costTime0);
                #else
                    runKernel2D(mSourceTransform[index], mGWS_S[index], mLWS_S[index],
                                mOpenCLBackend->getOpenCLRuntime());
                #endif
                }

                /*MatMul*/
                {
                #ifdef ENABLE_OPENCL_TIME_PROFILER
                    cl::Event event;
                    runKernel2D(mMatMul[index], mGWS_M[index], mLWS_M[index],
                                mOpenCLBackend->getOpenCLRuntime(), &event);
                    
                    int costTime1 = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
                    costTime += costTime1;
                    MNN_PRINT("kernel cost:%d    us ConvWino1\n",costTime1);
                #else
                    runKernel2D(mMatMul[index], mGWS_M[index], mLWS_M[index],
                                mOpenCLBackend->getOpenCLRuntime());
                #endif
                }

                // Dest Transform
                {
                #ifdef ENABLE_OPENCL_TIME_PROFILER
                    cl::Event event;
                    runKernel2D(mDestTransform[index], mGWS_D[index], mLWS_D[index],
                                mOpenCLBackend->getOpenCLRuntime(), &event);
                    
                    int costTime2 = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
                    costTime += costTime2;
                    MNN_PRINT("kernel cost:%d    us ConvWino2\n",costTime2);
                #else
                    runKernel2D(mDestTransform[index], mGWS_D[index], mLWS_D[index],
                                mOpenCLBackend->getOpenCLRuntime());
                #endif
                }
            }
        }
    }
    #ifdef ENABLE_OPENCL_TIME_PROFILER
    MNN_PRINT("kernel cost:%d    us ConvWino total\n",costTime);
    #endif

    return NO_ERROR;
}

} // namespace OpenCL
} // namespace MNN
