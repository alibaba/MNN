//
//  ConvWinograd.cpp
//  MNN
//
//  Created by MNN on 2019/01/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ConvWinograd.hpp"
#include <string.h>
#include "Backend.hpp"
#include "ConvolutionIntFactory.hpp"
#include "WingoradGenerater.hpp"
#include "core/OpenCLRunningUtils.hpp"
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
    std::set<std::string> basic;
    /*Create Kernel*/
    {
        char format[20];
        ::memset(format, 0, sizeof(format));
        sprintf(format, "%d_%d_%d", UNIT, kx, INTERP);
        auto formatStr = std::string(format);
        mSourceTransform =
            runTime->buildKernel("winogradTransformSource" + formatStr, "winogradTransformSource", basic);
        {
            std::set<std::string> buildOptions = basic;
            if (mCommon->relu()) {
                buildOptions.emplace("-DRELU");
            }
            if (mCommon->relu6()) {
                buildOptions.emplace("-DRELU6");
            }
            mDestTransform =
                runTime->buildKernel("winogradTransformDest" + formatStr, "winogradTransformDest", buildOptions);
        }
        mMatMul = runTime->buildKernel("gemm", "gemm", basic);
    }

    int weightSize             = 0;
    const float* filterDataPtr = nullptr;

    std::shared_ptr<MNN::ConvolutionIntFactory::Int8Common> quanCommon;
    if (nullptr != op->quanParameter()) {
        quanCommon = ConvolutionIntFactory::load(op->quanParameter(), true);
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
        auto biasSize = UP_DIV(co, 4) * 4 * sizeof(float);
        std::shared_ptr<cl::Buffer> biasBuffer(
            new cl::Buffer(runTime->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, biasSize));
        
        cl_int error;
        auto biasC = queue.enqueueMapBuffer(*biasBuffer, CL_TRUE, CL_MAP_WRITE, 0, biasSize, nullptr, nullptr, &error);
        if(biasC != nullptr && error == CL_SUCCESS){
            ::memset(biasC, 0, biasSize);
            ::memcpy(biasC, op->bias()->data(), co * sizeof(float));
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
        cl::Buffer weightBuffer(runTime->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, weightDest->size());
        {
            cl_int error;
            auto weightPtr = queue.enqueueMapBuffer(weightBuffer, CL_TRUE, CL_MAP_WRITE, 0, weightDestSize, nullptr, nullptr, &error);
            if(weightPtr != nullptr && error == CL_SUCCESS){
                ::memcpy(weightPtr, weightDest->host<float>(), weightDestSize);
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
    mPadX       = mCommon->padX();
    mPadY       = mCommon->padY();
    mStrideX    = mCommon->strideX();
    mStrideY    = mCommon->strideY();
    mPadMode    = mCommon->padMode();

    int alpha  = mCommon->kernelX() + UNIT - 1;
    auto wUnit = UP_DIV(output->width(), UNIT);
    auto hUnit = UP_DIV(output->height(), UNIT);
    int padX   = mPadX;
    int padY   = mPadY;
    if (mPadMode == PadMode_SAME) {
        int kernelWidthSize  = (mKernelX - 1) * mCommon->dilateX() + 1;
        int kernelHeightSize = (mKernelY - 1) * mCommon->dilateY() + 1;
        int padNeededWidth   = (output->width() - 1) * mStrideX + kernelWidthSize - input->width();
        int padNeededHeight  = (output->height() - 1) * mStrideY + kernelHeightSize - input->height();
        padX                 = padNeededWidth / 2;
        padY                 = padNeededHeight / 2;
    }

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

    mSourceTransform.setArg(0, openCLImage(input));
    mSourceTransform.setArg(1, openCLImage(mSource.get()));
    mSourceTransform.setArg(4, padX);
    mSourceTransform.setArg(5, padY);
    mSourceTransform.setArg(6, input->width());
    mSourceTransform.setArg(7, input->height());
    mSourceTransform.setArg(8, icC4);

    mMatMul.setArg(0, openCLImage(mSource.get()));
    mMatMul.setArg(1, *mWeight);
    mMatMul.setArg(4, ocC4);
    mMatMul.setArg(5, icC4);

    mDestTransform.setArg(1, *mBias);
    mDestTransform.setArg(2, openCLImage(output));
    mDestTransform.setArg(5, output->width());
    mDestTransform.setArg(6, output->height());
    mDestTransform.setArg(7, ocC4);

    return NO_ERROR;
}
ErrorCode ConvWinograd::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];
    int alpha   = mKernelX + UNIT - 1;
    auto wUnit  = UP_DIV(output->width(), UNIT);
    auto hUnit  = UP_DIV(output->height(), UNIT);

    auto icC4    = UP_DIV(input->channel(), 4);
    auto ocC4    = UP_DIV(output->channel(), 4);
    auto runTime = mOpenCLBackend->getOpenCLRuntime();

    int wPiece = UP_DIV(wUnit, mSliceNumber);
    int hPiece = UP_DIV(hUnit, mSliceNumber);

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

                mSourceTransform.setArg(2, wCount);
                mSourceTransform.setArg(3, hCount);
                mSourceTransform.setArg(9, offsetData[0]);
                mSourceTransform.setArg(10, offsetData[1]);
                mSourceTransform.setArg(11, b);

                auto gemmWidth = UP_DIV(wCount * hCount, 4);
                mMatMul.setArg(2, openCLImage(dest));
                mMatMul.setArg(3, gemmWidth);

                mDestTransform.setArg(0, openCLImage(dest));
                mDestTransform.setArg(3, wCount);
                mDestTransform.setArg(4, hCount);
                mDestTransform.setArg(8, offsetData[0]);
                mDestTransform.setArg(9, offsetData[1]);
                mDestTransform.setArg(10, b);

                /*Source Transform*/
                {
                    int align  = 8;
                    auto error = runTime->commandQueue().enqueueNDRangeKernel(
                        mSourceTransform, cl::NullRange,
                        cl::NDRange(UP_DIV(wCount, align) * align, UP_DIV(hCount, align) * align, icC4),
                        cl::NDRange(align, align, 1));
                    MNN_ASSERT(CL_SUCCESS == error);
                }

                /*MatMul*/
                {
                    int align       = 8;
                    auto gemmWidth  = UP_DIV(wCount * hCount, 4);
                    auto gemmHeight = ocC4;
                    auto error      = runTime->commandQueue().enqueueNDRangeKernel(
                        mMatMul, cl::NullRange,
                        cl::NDRange(UP_DIV(gemmWidth, align) * align, UP_DIV(gemmHeight, align) * align, alpha * alpha),
                        cl::NDRange(align, align, 1));
                    MNN_ASSERT(CL_SUCCESS == error);
                }

                // Dest Transform
                {
                    int align  = 8;
                    auto error = runTime->commandQueue().enqueueNDRangeKernel(
                        mDestTransform, cl::NullRange,
                        cl::NDRange(UP_DIV(wCount, align) * align, UP_DIV(hCount, align) * align, ocC4),
                        cl::NDRange(align, align, 1));
                    MNN_ASSERT(CL_SUCCESS == error);
                }
            }
        }
    }

    return NO_ERROR;
}

} // namespace OpenCL
} // namespace MNN
