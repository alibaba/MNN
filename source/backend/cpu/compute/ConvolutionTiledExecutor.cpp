//
//  ConvolutionTiledExecutor.cpp
//  MNN
//
//  Created by MNN on 2018/07/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/compute/ConvolutionTiledExecutor.hpp"
#include <MNN/AutoTime.hpp>
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/Concurrency.h"
#include "backend/cpu/compute/ConvOpt.h"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "math/Vec4.hpp"

namespace MNN {
static bool _initWeight(float *dest, const float *source, int depth, int outputCount, int kernelSize) {
    // Swap k, ic
    // FIXME: Use tensor instead of vector
    std::vector<float> tempBuffer(depth * outputCount * kernelSize);
    if (nullptr == tempBuffer.data()) {
        return false;
    }
    for (int o=0; o<outputCount; ++o) {
        auto dO = tempBuffer.data() + o * depth * kernelSize;
        auto sO = source + o * depth * kernelSize;
        for (int y=0; y<depth; ++y) {
            for (int x=0; x<kernelSize; ++x) {
                dO[y+x*depth] = sO[y*kernelSize+x];
            }
        }
    }
    MNNPackForMatMul_B(dest, tempBuffer.data(), outputCount, kernelSize * depth, true);
    return true;
}
ErrorCode ConvolutionTiledExecutorMultiInput::onExecute(const std::vector<Tensor*>& inputs,
                                                        const std::vector<Tensor*>& outputs) {
    int depth       = inputs[1]->channel();
    int outputCount = inputs[1]->batch();
    ::memset(mTempWeight->host<float>(), 0, mTempWeight->size());
    if (nullptr != mTempBias) {
        ::memset(mTempBias->host<float>(), 0, mTempBias->size());
        if (inputs.size() > 2) {
            ::memcpy(mTempBias->host<float>(), inputs[2]->host<float>(), inputs[2]->size());
        }
    }
    _initWeight(mTempWeight->host<float>(), inputs[1]->host<float>(), depth, outputCount,
                                  inputs[1]->width() * inputs[1]->height());
    return mProxy->onExecute(mInputs, outputs);
}
ErrorCode ConvolutionTiledExecutorMultiInput::onResize(const std::vector<Tensor*>& inputs,
                                                       const std::vector<Tensor*>& outputs) {
    int depth       = inputs[1]->channel();
    int outputCount = outputs[0]->channel();
    int eP, lP, hP;
    MNNGetMatMulPackMode(&eP, &lP, &hP);

    mTempWeight.reset(Tensor::createDevice<float>(
        {UP_DIV(outputCount, hP), depth * inputs[1]->width() * inputs[1]->height(), hP}));
    backend()->onAcquireBuffer(mTempWeight.get(), Backend::DYNAMIC);
    mTempBias.reset();
    if (inputs.size() > 2 && inputs[2]->elementSize() % 4 == 0) {
        mInputs = {inputs[0], mTempWeight.get(), inputs[2]};
    } else {
        mTempBias.reset(Tensor::createDevice<float>({ALIGN_UP4(outputCount)}));
        backend()->onAcquireBuffer(mTempBias.get(), Backend::DYNAMIC);
        mInputs = {inputs[0], mTempWeight.get(), mTempBias.get()};
    }
    auto errorCode = mProxy->onResize(mInputs, outputs);
    backend()->onReleaseBuffer(mTempWeight.get(), Backend::DYNAMIC);
    if (nullptr != mTempBias) {
        backend()->onReleaseBuffer(mTempBias.get(), Backend::DYNAMIC);
    }
    return errorCode;
}

ConvolutionTiledExecutor::ConvolutionTiledExecutor(const Convolution2DCommon* common, Backend* b,
                                                   const float* originWeight, size_t originWeightSize,
                                                   const float* bias, size_t biasSize)
    : MNN::Execution(b) {
    auto outputCount = (int)biasSize;
    int eP, lP, hP;
    MNNGetMatMulPackMode(&eP, &lP, &hP);

    // Don't use common->inputCount for old model common->inputCount is zero
    auto srcCount    = (int)originWeightSize / outputCount / common->kernelX() / common->kernelY();
    mWeight.reset(Tensor::createDevice<float>(
        {UP_DIV(outputCount, hP), UP_DIV(srcCount, 4), (int)common->kernelX(), common->kernelY(), 4 * hP}));
    mValid = backend()->onAcquireBuffer(mWeight.get(), Backend::STATIC);
    if (!mValid) {
        return;
    }
    _initWeight(mWeight->host<float>(), originWeight, srcCount, outputCount, common->kernelX() * common->kernelY());
    mBias.reset(Tensor::createDevice<float>({ALIGN_UP4((int)biasSize)}));
    mValid = backend()->onAcquireBuffer(mBias.get(), Backend::STATIC);
    if (!mValid) {
        return;
    }
    ::memset(mBias->host<float>(), 0, mBias->size());
    ::memcpy(mBias->host<float>(), bias, biasSize * sizeof(float));
    mProxy.reset(new ConvolutionTiledExecutorBasic(common, b));
}
ConvolutionTiledExecutor::~ConvolutionTiledExecutor() {
    if (nullptr != mBias) {
        backend()->onReleaseBuffer(mBias.get(), Backend::STATIC);
    }
    if (nullptr != mWeight) {
        backend()->onReleaseBuffer(mWeight.get(), Backend::STATIC);
    }
}
ErrorCode ConvolutionTiledExecutorBasic::onResize(const std::vector<Tensor*>& inputs,
                                                  const std::vector<Tensor*>& outputs) {
    MNN_ASSERT(3 == inputs.size());
    CPUConvolution::onResize(inputs, outputs);
    auto layer  = mCommon;
    auto input  = inputs[0];
    auto weight = inputs[1];
    auto bias   = inputs[2];
    auto output = outputs[0];
    mFunctions.clear();
    CONV_SETUP_KERNELSIZE(4);
    int threadNumber    = ((CPUBackend*)backend())->threadNumber();
    auto biasPtr        = bias->host<float>();
    auto weightPtr      = weight->host<float>();
    int src_z_step      = input->width() * input->height() * 4;
    int eP, lP, hP;
    MNNGetMatMulPackMode(&eP, &lP, &hP);
    auto CONVOLUTION_TILED_NUMBER = eP;
    auto& tempBuffer = mTempBuffer.buffer();
    auto icC4 = UP_DIV(input->channel(), 4);
    auto ic = input->channel();
    auto L = input->channel() * mCommon->kernelY() * mCommon->kernelX();
    auto kernelSize = mCommon->kernelX() * mCommon->kernelY();

    tempBuffer.dim[0].extent = threadNumber;
    tempBuffer.dim[1].extent = CONVOLUTION_TILED_NUMBER;
    tempBuffer.dim[2].extent = icC4 * mCommon->kernelY() * mCommon->kernelX(); // srcCount * kx*ky
    tempBuffer.dim[3].extent = 4;
    TensorUtils::setLinearLayout(&mTempBuffer);
    
    mTempBufferTranspose.buffer().dimensions = 2;
    mTempBufferTranspose.buffer().dim[0].extent = threadNumber;
    mTempBufferTranspose.buffer().dim[1].extent = L * CONVOLUTION_TILED_NUMBER;
    TensorUtils::setLinearLayout(&mTempBufferTranspose);

    
    int count                             = UP_DIV(width*height, CONVOLUTION_TILED_NUMBER);
    int plane = width * height;

    bool success = backend()->onAcquireBuffer(&mTempBuffer, Backend::DYNAMIC) && backend()->onAcquireBuffer(&mTempBufferTranspose, Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }
    auto hDiv = MNNGetC4DivNumber(hP);
    auto outputChannel = output->channel();
    auto oC4 = UP_DIV(outputChannel, 4);
    std::shared_ptr<Tensor> cache;
    if (hP % 4 != 0) {
        cache.reset(Tensor::createDevice<float>({threadNumber, 4 * hDiv * eP + oC4 * 4 * eP}));
        success = backend()->onAcquireBuffer(cache.get(), Backend::DYNAMIC);
        if (!success) {
            return OUT_OF_MEMORY;
        }
        backend()->onReleaseBuffer(cache.get(), Backend::DYNAMIC);
    }

    backend()->onReleaseBuffer(&mTempBuffer, Backend::DYNAMIC);
    backend()->onReleaseBuffer(&mTempBufferTranspose, Backend::DYNAMIC);
    std::vector<size_t> parameters(6);
    parameters[0] = eP * sizeof(float);
    parameters[1] = L;
    parameters[2] = outputChannel;
    parameters[3] = plane * 4 * sizeof(float);
    parameters[4] = 0;
    parameters[5] = 0;
    auto threadNumberFirst                 = std::min(threadNumber, count);
    auto postParameters = getPostParameters();
    std::function<void(int)> firstFunction = [=](int tId) {
        auto colBuffer = mTempBuffer.host<float>() + mTempBuffer.stride(0) * tId;
        auto gemmBuffer = mTempBufferTranspose.host<float>() + mTempBufferTranspose.stride(0) * tId;
        float* cachePtr = nullptr;
        if (nullptr != cache) {
            cachePtr = cache->host<float>() + tId * cache->stride(0);
        }
        for (int batchIndex = 0; batchIndex < input->batch(); ++batchIndex) {
            auto dstOrigin = output->host<float>() + batchIndex * output->stride(0);
            auto srcOrigin = input->host<float>() + batchIndex * input->stride(0);

            for (int x = (int)tId; x < count; x += threadNumberFirst) {
                int start    = (int)x * CONVOLUTION_TILED_NUMBER;
                int remain   = plane - start;
                int xC        = remain > CONVOLUTION_TILED_NUMBER ? CONVOLUTION_TILED_NUMBER : remain;
                // Im2Col
                ::memset(colBuffer, 0, mTempBuffer.stride(0) * sizeof(float));
                int oyBegin = start / width;
                int oxBegin = start % width;
                int oyEnd = (start + xC-1) / width;
                remain = xC;
                auto colIndex = colBuffer;
                for (int oy=oyBegin; oy <= oyEnd; ++oy) {
                    int step = std::min(width - oxBegin, remain);
                    int sySta = oy * strideY - padY;
                    int kyStart = std::max(0, UP_DIV(-sySta, dilateY));
                    int kyEnd = std::min(kernel_height, UP_DIV(src_height - sySta, dilateY));
                    for (int i=0; i<step; ++i) {
                        int ox = i + oxBegin;
                        int sxSta = ox * strideX - padX;
                        int kxStart = std::max(0, UP_DIV(-sxSta, dilateX));
                        int kxEnd = std::min(kernel_width, UP_DIV(src_width - sxSta, dilateX));
                        // ivec2 sfxy = max(ivec2(0), (UP_DIV(-s0, uConstant.dilate)));
                        // ivec2 efxy = min(uConstant.kernelSize, UP_DIV(inputSize.xy-s0, uConstant.dilate));
                        auto srcStart = srcOrigin + sxSta * 4 + sySta * 4 * src_width;
                        auto dstStart = colIndex + 4 * i;
                        for (int sz=0; sz<icC4; ++sz) {
                            auto srcZ = srcStart + src_z_step * sz;
                            auto dstZ = dstStart + 4 * CONVOLUTION_TILED_NUMBER * kernel_height * kernel_width * sz;
                            for (int ky=kyStart; ky<kyEnd; ++ky) {
                                auto sy = ky * dilateY;
                                auto srcY = srcZ + sy * 4 * src_width;
                                auto dstY = dstZ + 4 * CONVOLUTION_TILED_NUMBER * (ky*kernel_width);
                                for (int kx=kxStart; kx<kxEnd; ++kx) {
                                    auto sx = kx * dilateX;
                                    auto srcX = srcY + sx * 4;
                                    auto dstX = dstY + 4 * CONVOLUTION_TILED_NUMBER * kx;
                                    Math::Vec4::save(dstX, Math::Vec4::load(srcX));
                                }
                            }
                        }
                    }
                    oxBegin = 0;
                    remain -= step;
                    colIndex += 4 * step;
                }

                // GEMM
                MNNPackC4ForMatMul_A(gemmBuffer, colBuffer, CONVOLUTION_TILED_NUMBER * kernelSize, ic, CONVOLUTION_TILED_NUMBER * kernelSize);
                if (xC == CONVOLUTION_TILED_NUMBER) {
                    MNNPackedMatMul(dstOrigin + start * 4, gemmBuffer, weightPtr, parameters.data(), cachePtr, postParameters.data(), biasPtr);
                } else {
                    MNNPackedMatMulRemain(dstOrigin + start * 4, gemmBuffer, weightPtr, xC, parameters.data(), cachePtr, postParameters.data(), biasPtr);
                }
            }
        }
    };
    mFunctions.emplace_back(std::make_pair(threadNumberFirst, firstFunction));
    return NO_ERROR;
}

ErrorCode ConvolutionTiledExecutorBasic::onExecute(const std::vector<Tensor*>& inputs,
                                                   const std::vector<Tensor*>& outputs) {
    for (auto& iter : mFunctions) {
        MNN_CONCURRENCY_BEGIN(tId, iter.first) {
            iter.second((int)tId);
        }
        MNN_CONCURRENCY_END();
    }
    return NO_ERROR;
}
} // namespace MNN
