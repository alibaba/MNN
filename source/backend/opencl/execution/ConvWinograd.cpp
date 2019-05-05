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
    if (input->batch() != 1) {
        return false;
    }
    if (common->strideX() != 1 || common->strideY() != 1) {
        return false;
    }
    if (common->dilateX() != 1 || common->dilateY() != 1) {
        return false;
    }
    int alpha  = common->kernelX() + UNIT - 1;
    auto wUnit = UP_DIV(input->width(), UNIT);
    auto hUnit = UP_DIV(input->height(), UNIT);
    auto plane = UP_DIV(wUnit * hUnit, 4);
    if (plane * alpha * alpha > limit) {
        return false;
    }

    return common->kernelX() == 3 && common->kernelY() == 3;
}

ConvWinograd::ConvWinograd(const MNN::Convolution2D* op, Backend* backend) : Execution(backend) {
    mOpenCLBackend = static_cast<OpenCLBackend*>(backend);
    mCommon        = op->common();
    MNN_ASSERT(3 == mCommon->kernelY() && 3 == mCommon->kernelX());
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
        auto biasC = queue.enqueueMapBuffer(*biasBuffer, CL_TRUE, CL_MAP_WRITE, 0, biasSize);
        ::memset(biasC, 0, biasSize);
        ::memcpy(biasC, op->bias()->data(), co * sizeof(float));
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
            auto weightPtr = queue.enqueueMapBuffer(weightBuffer, CL_TRUE, CL_MAP_WRITE, 0, weightDestSize);
            ::memcpy(weightPtr, weightDest->host<float>(), weightDestSize);
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
        int kernelWidthSize = (mKernelX - 1) * mCommon->dilateX() + 1;
        int kernelHeightSize = (mKernelY - 1) * mCommon->dilateY() + 1;
        int padNeededWidth  = (output->width() - 1) * mStrideX + kernelWidthSize - input->width();
        int padNeededHeight = (output->height() - 1) * mStrideY + kernelHeightSize - input->height();
        padX = padNeededWidth / 2;
        padY = padNeededHeight / 2;
    }
    mSource.reset(Tensor::createDevice<float>(
        std::vector<int>{alpha * alpha, input->channel(), UP_DIV(wUnit * hUnit, 4), 4}, Tensor::CAFFE_C4));
    mDest.reset(Tensor::createDevice<float>(
        std::vector<int>{4, wUnit * hUnit, UP_DIV(output->channel(), 4), alpha * alpha}, Tensor::CAFFE_C4));
    auto icC4 = UP_DIV(input->channel(), 4);
    auto ocC4 = UP_DIV(output->channel(), 4);

    auto bn = backend();
    bn->onAcquireBuffer(mSource.get(), Backend::DYNAMIC);
    bn->onAcquireBuffer(mDest.get(), Backend::DYNAMIC);
    bn->onReleaseBuffer(mSource.get(), Backend::DYNAMIC);
    bn->onReleaseBuffer(mDest.get(), Backend::DYNAMIC);

    mSourceTransform.setArg(0, openCLImage(input));
    mSourceTransform.setArg(1, openCLImage(mSource.get()));
    mSourceTransform.setArg(2, wUnit);
    mSourceTransform.setArg(3, hUnit);
    mSourceTransform.setArg(4, padX);
    mSourceTransform.setArg(5, padY);
    mSourceTransform.setArg(6, input->width());
    mSourceTransform.setArg(7, input->height());
    mSourceTransform.setArg(8, icC4);
    mSourceTransform.setArg(9, 0);
    mSourceTransform.setArg(10, 0);

    auto gemmWidth = UP_DIV(wUnit * hUnit, 4);
    mMatMul.setArg(0, openCLImage(mSource.get()));
    mMatMul.setArg(1, *mWeight);
    mMatMul.setArg(2, openCLImage(mDest.get()));
    mMatMul.setArg(3, gemmWidth);
    mMatMul.setArg(4, ocC4);
    mMatMul.setArg(5, icC4);

    mDestTransform.setArg(0, openCLImage(mDest.get()));
    mDestTransform.setArg(1, *mBias);
    mDestTransform.setArg(2, openCLImage(output));
    mDestTransform.setArg(3, wUnit);
    mDestTransform.setArg(4, hUnit);
    mDestTransform.setArg(5, output->width());
    mDestTransform.setArg(6, output->height());
    mDestTransform.setArg(7, ocC4);
    mDestTransform.setArg(8, 0);
    mDestTransform.setArg(9, 0);

    return NO_ERROR;
}
ErrorCode ConvWinograd::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    // TODO Support batch
    auto input  = inputs[0];
    auto output = outputs[0];
    int alpha   = mKernelX + UNIT - 1;
    auto wUnit  = UP_DIV(output->width(), UNIT);
    auto hUnit  = UP_DIV(output->height(), UNIT);

    auto icC4    = UP_DIV(input->channel(), 4);
    auto ocC4    = UP_DIV(output->channel(), 4);
    auto runTime = mOpenCLBackend->getOpenCLRuntime();
    /*Source Transform*/
    {
        int align  = 8;
        auto error = runTime->commandQueue().enqueueNDRangeKernel(
            mSourceTransform, cl::NullRange,
            cl::NDRange(UP_DIV(wUnit, align) * align, UP_DIV(hUnit, align) * align, icC4),
            cl::NDRange(align, align, 1));
        MNN_ASSERT(CL_SUCCESS == error);
    }

    /*MatMul*/
    {
        int align       = 8;
        auto gemmWidth  = UP_DIV(wUnit * hUnit, 4);
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
            cl::NDRange(UP_DIV(wUnit, align) * align, UP_DIV(hUnit, align) * align, ocC4),
            cl::NDRange(align, align, 1));
        MNN_ASSERT(CL_SUCCESS == error);
    }

    return NO_ERROR;
}

} // namespace OpenCL
} // namespace MNN
