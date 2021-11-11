//
//  DenseConvolutionTiledExecutor.cpp
//  MNN
//
//  Created by MNN on 2018/07/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "DenseConvolutionTiledExecutor.hpp"
#include <MNN/AutoTime.hpp>
#include "backend/cpu/CPUBackend.hpp"
#include "CommonOptFunction.h"
#include "core/Concurrency.h"
#include "ConvOpt.h"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "math/Vec.hpp"
#include "core/BufferAllocator.hpp"
#include "common/MemoryFormater.h"

using Vec4 = MNN::Math::Vec<float, 4>;
namespace MNN {

void DenseConvolutionTiledExecutor::initWeight(float *dest, const float *source, float* cache, int depth, int outputCount, int kernelSize, const CoreFunctions* function) {
    ConvolutionTiledExecutor::initWeight(source, cache, depth, outputCount, kernelSize, function);
    function->MNNPackForMatMul_B(dest, cache, outputCount, kernelSize * depth, true);
    /*MNN_PRINT("dense weight matrix tile:");
    formatMatrix(dest, {UP_DIV(outputCount, 4), kernelSize * depth, 4});*/
}

DenseConvolutionTiledExecutor::DenseConvolutionTiledExecutor(const Convolution2DCommon* common, Backend* b,
                                                   const float* originWeight, size_t originWeightSize,
                                                   const float* bias, size_t biasSize)
    : ConvolutionTiledExecutor(b, bias, biasSize) {

    auto outputCount = (int)biasSize;
    int eP, lP, hP;
    auto core = static_cast<CPUBackend*>(b)->functions();
    int bytes = core->bytes;
    core->MNNGetMatMulPackMode(&eP, &lP, &hP);
    // Don't use common->inputCount for old model common->inputCount is zero
    auto srcCount    = (int)originWeightSize / outputCount / common->kernelX() / common->kernelY();
    auto lSize = srcCount * common->kernelX() * common->kernelY();
    mResource->mWeight.reset(Tensor::createDevice<uint8_t>(
        {UP_DIV(outputCount, hP) * UP_DIV(lSize, lP) * hP * lP * bytes}));
    std::shared_ptr<Tensor> cache(Tensor::createDevice<uint8_t>({outputCount * srcCount * common->kernelX() * common->kernelY() * (int)sizeof(float)})); // cache must be float

    mValid = mValid && backend()->onAcquireBuffer(mResource->mWeight.get(), Backend::STATIC);
    mValid = mValid && backend()->onAcquireBuffer(cache.get(), Backend::STATIC);
    if (!mValid) {
        return;
    }
    initWeight(mResource->mWeight->host<float>(), originWeight, cache->host<float>(), srcCount, outputCount, common->kernelX() * common->kernelY(), core);
    backend()->onReleaseBuffer(cache.get(), Backend::STATIC);
    mProxy.reset(new DenseConvolutionTiledImpl(common, b));
}

DenseConvolutionTiledExecutor::DenseConvolutionTiledExecutor(std::shared_ptr<CPUConvolution::Resource> res, const Convolution2DCommon* common, Backend* b) : ConvolutionTiledExecutor(res, b) {
    mProxy.reset(new DenseConvolutionTiledImpl(common, b));
}

DenseConvolutionTiledExecutor::~DenseConvolutionTiledExecutor() {
    // Do nothing
}
bool DenseConvolutionTiledExecutor::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    *dst = new DenseConvolutionTiledExecutor(mResource, op->main_as_Convolution2D()->common(), bn);
    return true;
}

ErrorCode ConvolutionTiledExecutorMultiInput::onExecute(const std::vector<Tensor*>& inputs,
                                                        const std::vector<Tensor*>& outputs) {
    int depth       = inputs[1]->channel();
    int outputCount = inputs[1]->batch();
    auto function = static_cast<CPUBackend*>(backend())->functions();
    if (nullptr != mTempBias) {
        ::memset(mTempBias->host<float>(), 0, mTempBias->elementSize() * function->bytes);
        if (inputs.size() > 2) {
            ::memcpy(mTempBias->host<float>(), inputs[2]->host<float>(), inputs[2]->elementSize() * function->bytes);
        }
    }
    auto cache = mTempWeightCache->host<float>();
    auto source = inputs[1]->host<float>();
    auto kernelSize = inputs[1]->stride(1);
    // Swap k, ic
    int dims[4] = {
        depth,
        kernelSize,
        kernelSize,
        depth
    };
    if (function->bytes < 4) {
        // TODO: Opt it
        // Lowp
        source = mTempWeightCache->host<float>() + mTempWeightCache->stride(0);
        function->MNNLowpToFp32(inputs[1]->host<int16_t>(), source, inputs[1]->elementSize());
        for (int o=0; o<outputCount; ++o) {
            auto dO = cache + o * depth * kernelSize;
            auto sO = source + o * depth * kernelSize;
            MNNTranspose32Bit((int32_t*)dO, (const int32_t*)sO, &dims[0]);
        }
        function->MNNFp32ToLowp(cache, (int16_t*)cache, inputs[1]->elementSize());
    } else {
        for (int o=0; o<outputCount; ++o) {
            auto dO = cache + o * depth * kernelSize;
            auto sO = source + o * depth * kernelSize;
            MNNTranspose32Bit((int32_t*)dO, (const int32_t*)sO, &dims[0]);
        }
    }
    function->MNNPackForMatMul_B(mTempWeight->host<float>(), mTempWeightCache->host<float>(), outputCount, kernelSize * depth, true);
    return mProxy->onExecute(mInputs, outputs);
}
ErrorCode ConvolutionTiledExecutorMultiInput::onResize(const std::vector<Tensor*>& inputs,
                                                       const std::vector<Tensor*>& outputs) {
    int depth       = inputs[1]->channel();
    int outputCount = outputs[0]->channel();
    auto function = static_cast<CPUBackend*>(backend())->functions();
    int eP, lP, hP;
    function->MNNGetMatMulPackMode(&eP, &lP, &hP);
    auto kernelSize = depth * inputs[1]->stride(1);
    mTempWeight.reset(Tensor::createDevice<float>(
        {UP_DIV(outputCount, hP), UP_DIV(kernelSize, lP), lP * hP}));
    if (function->bytes < 4) {
        mTempWeightCache.reset(Tensor::createDevice<int32_t>({2, outputCount * kernelSize}));
    } else {
        mTempWeightCache.reset(Tensor::createDevice<float>({outputCount * kernelSize}));
    }
    auto res = backend()->onAcquireBuffer(mTempWeight.get(), Backend::DYNAMIC);
    res = res && backend()->onAcquireBuffer(mTempWeightCache.get(), Backend::DYNAMIC);
    mTempBias.reset();
    if (!res) {
        return OUT_OF_MEMORY;
    }
    if (inputs.size() > 2 && inputs[2]->elementSize() % function->pack == 0) {
        mInputs = {inputs[0], mTempWeight.get(), inputs[2]};
    } else {
        mTempBias.reset(Tensor::createDevice<float>({UP_DIV(outputCount, function->pack) * function->pack}));
        backend()->onAcquireBuffer(mTempBias.get(), Backend::DYNAMIC);
        mInputs = {inputs[0], mTempWeight.get(), mTempBias.get()};
    }
    backend()->onReleaseBuffer(mTempWeightCache.get(), Backend::DYNAMIC);
    auto errorCode = mProxy->onResize(mInputs, outputs);
    backend()->onReleaseBuffer(mTempWeight.get(), Backend::DYNAMIC);
    if (nullptr != mTempBias) {
        backend()->onReleaseBuffer(mTempBias.get(), Backend::DYNAMIC);
    }
    return errorCode;
}


void DenseConvolutionTiledImpl::getPackParameter(int* eP, int* lP, int* hP, const CoreFunctions* core) {
    core->MNNGetMatMulPackMode(eP, lP, hP);
    return;
}

ErrorCode DenseConvolutionTiledImpl::onResize(const std::vector<Tensor*>& inputs,
                                                  const std::vector<Tensor*>& outputs) {
    CPUConvolution::onResize(inputs, outputs);
    auto input   = inputs[0];
    auto weight  = inputs[1];
    Tensor *bias = nullptr;
    auto core    = static_cast<CPUBackend *>(backend())->functions();
    int bytes    = core->bytes;
    int unit     = core->pack;
    auto packA   = core->MNNPackC4ForMatMul_A;
    int eP, lP, hP;
    getPackParameter(&eP, &lP, &hP, core);
    auto matmulUnit   = core->MNNPackedMatMul;
    auto matmulRemain = core->MNNPackedMatMulRemain;
    auto strideX           = mCommon->strideX();
    auto strideY           = mCommon->strideY();
    auto dilateX           = mCommon->dilateX();
    auto dilateY           = mCommon->dilateY();
    auto padY              = mPadY;
    auto padX              = mPadX;
    auto kernel_width      = mCommon->kernelX();
    auto kernel_height     = mCommon->kernelY();
    auto output      = outputs[0];
    auto batch       = output->batch();
    auto width       = output->width();
    auto height      = output->height();
    int threadNumber = ((CPUBackend *)backend())->threadNumber();
    auto weightPtr = weight->host<float>();
    auto src_width                = input->width();
    auto src_height               = input->height();
    auto icC4                     = UP_DIV(input->channel(), unit);
    auto ic                       = input->channel();
    auto L                        = ic * mCommon->kernelY() * mCommon->kernelX();
    if (src_width == 1 && width == 1 && height > 1 && kernel_width == 1 && mPadX == 0) {
        /* Convolution only work for Height. Swap x, y*/
        width         = height;
        height        = 1;
        padX          = mPadY;
        padY          = mPadX;
        strideX       = strideY;
        strideY       = 1; /* Don't need stride */
        src_width     = src_height;
        src_height    = 1;
        dilateX       = dilateY;
        dilateY       = 1;
        kernel_width  = kernel_height;
        kernel_height = 1;
    }
    const float *biasPtr = nullptr;
    if (inputs.size() > 2) {
        bias    = inputs[2];
        biasPtr = bias->host<float>();
    }
    auto kernelSize               = mCommon->kernelX() * mCommon->kernelY();
    mTempBufferTranspose.buffer().type          = halide_type_of<uint8_t>();
    mTempBufferTranspose.buffer().dimensions    = 2;
    mTempBufferTranspose.buffer().dim[0].extent = threadNumber;
    mTempBufferTranspose.buffer().dim[1].extent = UP_DIV(L, lP) * lP * eP * bytes;
    TensorUtils::setLinearLayout(&mTempBufferTranspose);
    auto plane    = width * height * batch;
    int tileCount = UP_DIV(plane, eP);

    bool success = backend()->onAcquireBuffer(&mTempBufferTranspose, Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }
    auto outputChannel = output->channel();
    auto oC4           = UP_DIV(outputChannel, unit);
    auto bufferAlloc   = static_cast<CPUBackend *>(backend())->getBufferAllocator();
    auto maxLine       = UP_DIV(eP, width) + 1;
    auto tempPtr = bufferAlloc->alloc(kernelSize * maxLine * threadNumber * (4 * sizeof(int32_t) + sizeof(float *)));
    if (nullptr == tempPtr.first) {
        return OUT_OF_MEMORY;
    }
    backend()->onReleaseBuffer(&mTempBufferTranspose, Backend::DYNAMIC);
    bufferAlloc->free(tempPtr);
    auto threadNumberFirst = std::min(threadNumber, tileCount);
    auto postParameters    = getPostParameters();
    mFunction.first        = threadNumberFirst;

    mFunction.second       = [=](int tId) {
        auto gemmBuffer = mTempBufferTranspose.host<uint8_t>() + mTempBufferTranspose.stride(0) * tId;
        auto srcPtr     = (float const **)((uint8_t *)tempPtr.first + tempPtr.second +
                                       tId * kernelSize * maxLine * (4 * sizeof(int32_t) + sizeof(float *)));
        auto el         = (int32_t *)(srcPtr + kernelSize * maxLine);

        int32_t info[4];
        info[1] = src_width * src_height * batch;
        info[2] = eP;
        info[3] = strideX;
        size_t parameters[6];
        parameters[0]          = eP * bytes;
        parameters[1]          = L;
        parameters[2]          = outputChannel;
        parameters[3]          = plane * unit * bytes;
        parameters[4]          = 0;
        parameters[5]          = 0;

        auto dstOrigin = output->host<uint8_t>();
        auto srcOrigin = input->host<uint8_t>();
        for (int x = (int)tId; x < tileCount; x += threadNumberFirst) {
            int start  = (int)x * eP;
            int remain = plane - start;
            int xC     = remain > eP ? eP : remain;
            /* Compute Pack position */
            int oyBegin   = start / width;
            int oxBegin   = start % width;
            int oyEnd     = (start + xC - 1) / width;
            remain        = xC;
            int number    = 0;
            bool needZero = false;
            int eStart    = 0;
            for (int oyb = oyBegin; oyb <= oyEnd; ++oyb) {
                int step    = std::min(width - oxBegin, remain);
                int oy      = oyb % height;
                int ob      = oyb / height;
                int sySta   = oy * strideY - padY;
                int kyStart = std::max(0, UP_DIV(-sySta, dilateY));
                int kyEnd   = std::min(kernel_height, UP_DIV(src_height - sySta, dilateY));
                if (kyEnd - kyStart < kernel_height) {
                    needZero = true;
                }
                auto srcStart = srcOrigin + ((ob * src_height + sySta) * src_width) * bytes * unit;
                for (int ky = kyStart; ky < kyEnd; ++ky) {
                    auto lKYOffset = ky * kernel_width * ic;
                    auto srcKy     = srcStart + ky * dilateY * src_width * bytes * unit;
                    for (int kx = 0; kx < kernel_width; ++kx) {
                        /* Compute x range:*/
                        /* 0 <= (oxBegin + x) * strideX - padX + dilateX * kx < src_width*/
                        /* 0 <= x <= step*/
                        int end = std::min(
                            step, (src_width - oxBegin * strideX - dilateX * kx + padX + strideX - 1) / strideX);
                        int sta = std::max(0, UP_DIV((padX - oxBegin * strideX - dilateX * kx), strideX));
                        if (end - sta < step) {
                            needZero = true;
                        }
                        if (end > sta) {
                            auto lOffset = lKYOffset + (kx * ic);
                            auto srcKx   = srcKy + ((oxBegin + sta) * strideX + dilateX * kx - padX) * bytes * unit;
                            srcPtr[number]     = (const float *)srcKx;
                            el[4 * number + 0] = end - sta;
                            el[4 * number + 1] = ic;
                            el[4 * number + 2] = eStart + sta;
                            el[4 * number + 3] = lOffset;
                            number++;
                        }
                    }
                }
                oxBegin = 0;
                remain -= step;
                eStart += step;
            }
            info[0] = number;
            if (needZero || lP != 1) {
                ::memset(gemmBuffer, 0, mTempBufferTranspose.stride(0));
            }
            if (number > 0) {
                packA((float *)gemmBuffer, srcPtr, info, el);
            }
            if (xC == eP) {
                matmulUnit((float*)(dstOrigin + start * unit * bytes), (float*)gemmBuffer, weightPtr, parameters,postParameters.data(), biasPtr);
            } else {
                matmulRemain((float*)(dstOrigin + start * unit * bytes), (float*)gemmBuffer, weightPtr, xC, parameters,postParameters.data(), biasPtr);
            }
        }
    };
    return NO_ERROR;
}

} // namespace MNN
