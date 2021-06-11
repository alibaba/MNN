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
#include "core/MemoryFormater.h"

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
    auto kernelSize = inputs[1]->width() * inputs[1]->height();
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
    function->MNNPackForMatMul_B(mTempWeight->host<float>(), mTempWeightCache->host<float>(), outputCount, inputs[1]->width() * inputs[1]->height() * depth, true);
    return mProxy->onExecute(mInputs, outputs);
}
ErrorCode ConvolutionTiledExecutorMultiInput::onResize(const std::vector<Tensor*>& inputs,
                                                       const std::vector<Tensor*>& outputs) {
    int depth       = inputs[1]->channel();
    int outputCount = outputs[0]->channel();
    auto function = static_cast<CPUBackend*>(backend())->functions();
    int eP, lP, hP;
    function->MNNGetMatMulPackMode(&eP, &lP, &hP);
    auto kernelSize = depth * inputs[1]->width() * inputs[1]->height();
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

#define GENERATE_FUNCTOR()                     \
    auto matmulUnit   = core->MNNPackedMatMul; \
    auto matmulRemain = core->MNNPackedMatMulRemain;

#define GENERATE_WEIGHT()                      \
    auto weightPtr = weight->host<float>();

#define GENERATE_MM()                                                                                                  \
    if (xC == CONVOLUTION_TILED_NUMBER) {                                                                              \
        matmulUnit((float*)(dstOrigin + start * unit * bytes), (float*)gemmBuffer, weightPtr, parameters.data(),       \
                   postParameters.data(), biasPtr);                                                                    \
    } else {                                                                                                           \
        matmulRemain((float*)(dstOrigin + start * unit * bytes), (float*)gemmBuffer, weightPtr, xC, parameters.data(), \
                     postParameters.data(), biasPtr);                                                                  \
    }                                                                                                                  \
    /*MNN_PRINT("formatMatrix gemm. xC:%d, eP:%d\n", xC, eP);*/                                                        \
    /*formatMatrix((float*)(dstOrigin + start * 4 * bytes), {UP_DIV(outputChannel, hP), xC, hP});*/

ErrorCode DenseConvolutionTiledImpl::onResize(const std::vector<Tensor*>& inputs,
                                                  const std::vector<Tensor*>& outputs) {
    GENERATE_RESIZE();
}

#undef GENERATE_FUNCTOR
#undef GENERATE_WEIGHT
#undef GENERATE_MM

} // namespace MNN
