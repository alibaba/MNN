//
//  DenseConvolutionTiledExecutor.cpp
//  MNN
//
//  Created by MNN on 2018/07/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include <math.h>
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
#define PARAMETERSIZE 6

#define MNN_ALLOC_MEMORY_INDIRECTLY
using Vec4 = MNN::Math::Vec<float, 4>;
namespace MNN {

void DenseConvolutionTiledExecutor::initWeight(float *dest, const float *source, float* cache, int depth, int outputCount, int kernelSize, const CoreFunctions* function) {
    ConvolutionTiledExecutor::initWeight(source, cache, depth, outputCount, kernelSize, function);
    function->MNNPackForMatMul_B(dest, cache, outputCount, kernelSize * depth, true);

}
static bool _initQuantizeResource(std::shared_ptr<ConvolutionCommon::Int8Common> int8Info, std::shared_ptr<CPUConvolution::Resource> resource, int hU, int hP, int lU, int lP, int outputCount, int srcChannel, int kernelSize) {
    int weightLength = hU * lU * hP * lP;
    resource->mWeight.reset(Tensor::createDevice<uint8_t>(
        {weightLength}));
    auto res = resource->backend->onAcquireBuffer(resource->mWeight.get(), Backend::STATIC);
    if (!res) {
        return false;
    }
    resource->mDequantize.bits = 8;
    resource->lU = lU;
    resource->hU = hU;
    resource->lP = lP;
    resource->hP = hP;
    // Reorder weight
    MNN_ASSERT(lP == 1);
    auto dstWInt8 = resource->mWeight->host<int8_t>();
    auto srcWInt8 = int8Info->weight.get();
    for (int y=0; y<outputCount; ++y) {
        int yo = y / hP;
        int yi = y % hP;
        auto srcY = srcWInt8 + y * srcChannel * kernelSize;
        auto dstY = dstWInt8 + yo * lP * hP * lU + yi;
        for (int iz=0; iz<srcChannel; ++iz) {
            for (int k=0; k<kernelSize; ++k) {
                int sx = iz * kernelSize + k;
                int dx = iz + k * srcChannel;
                dstY[dx * hP] = srcY[sx];
            }
        }
    }
    // Save scale bias
    resource->mDequantize.mScaleBias.reset(MNN::Tensor::createDevice<float>({hU * hP * 2}));
    res = resource->backend->onAcquireBuffer(resource->mDequantize.mScaleBias.get(), Backend::STATIC);
    if (!res) {
        return false;
    }
    auto alphaPtr = resource->mDequantize.mScaleBias->host<float>();
    auto biasPtr = resource->mDequantize.mScaleBias->host<float>() + hU * hP;
    ::memset(alphaPtr, 0, 2 * hU * hP * sizeof(float));
    int h = int8Info->alpha.size();
    if (int8Info->asymmetric) {
        h = h / 2;
        for (int i=0; i<h; ++i) {
            alphaPtr[i] = int8Info->alpha.get()[2 * i + 1];
            biasPtr[i] = int8Info->alpha.get()[2 * i];
        }
    } else {
        for (int i=0; i<h; ++i) {
            alphaPtr[i] = int8Info->alpha.get()[i];
        }
    }
    if (int8Info->canUseInt4) {
        MNN_ASSERT(weightLength % 2 == 0);
        weightLength = UP_DIV(weightLength, 2);
        resource->mDequantize.bits = 4;
        resource->mDequantize.mLowBitWeightMap = int8Info->weightMap;
        std::shared_ptr<MNN::Tensor> weightLow(Tensor::createDevice<uint8_t>(
            {weightLength}));
        auto res = resource->backend->onAcquireBuffer(weightLow.get(), Backend::STATIC);
        if (!res) {
            return false;
        }
        auto srcPtr = resource->mWeight->host<int8_t>();
        auto dstPtr = weightLow->host<uint8_t>();
        for (int i=0; i<weightLength; ++i) {
            int s0 = srcPtr[2 * i + 0];
            int s1 = srcPtr[2 * i + 1];
            s0 = int8Info->weightReverseMap[(int)s0 + 128];
            s1 = int8Info->weightReverseMap[(int)s1 + 128];
            int d = s0 * 16 + s1;
            dstPtr[i] = d;
        }
        resource->mWeight = weightLow;
    }
    return true;
}

DenseConvolutionTiledExecutor::DenseConvolutionTiledExecutor(const Convolution2DCommon* common, Backend* b,
                                                   const float* originWeight, size_t originWeightSize,
                                                   const float* bias, size_t biasSize, std::shared_ptr<ConvolutionCommon::Int8Common> int8Info)
    : ConvolutionTiledExecutor(b, bias, biasSize) {

    auto outputCount = (int)biasSize;
    int eP, lP, hP;
    auto core = static_cast<CPUBackend*>(b)->functions();
    int bytes = core->bytes;
    core->MNNGetMatMulPackMode(&eP, &lP, &hP);
    bool useInt8Weight = 0 == originWeightSize;
    if (useInt8Weight) {
        MNN_ASSERT(nullptr != int8Info.get());
        originWeightSize = int8Info->weight.size();
    }
    // Don't use common->inputCount for old model common->inputCount is zero
    auto srcCount    = (int)originWeightSize / outputCount / common->kernelX() / common->kernelY();
    auto lSize = srcCount * common->kernelX() * common->kernelY();
    auto hU = UP_DIV(outputCount, hP);
    auto lU = UP_DIV(lSize, lP);
    if (useInt8Weight) {
        // Quantize weight to int8
        auto allocSuccess = _initQuantizeResource(int8Info, mResource, hU, hP, lU, lP, outputCount, srcCount, common->kernelX() * common->kernelY());
        if (!allocSuccess) {
            mValid = false;
            return;
        }
    } else {
        mResource->mWeight.reset(Tensor::createDevice<uint8_t>(
            {hU * lU * hP * lP * bytes}));
        mValid = mValid && backend()->onAcquireBuffer(mResource->mWeight.get(), Backend::STATIC);
        if (!mValid) {
            return;
        }
        std::shared_ptr<Tensor> cache(Tensor::createDevice<uint8_t>({outputCount * srcCount * common->kernelX() * common->kernelY() * (int)sizeof(float)})); // cache must be float
        mValid = mValid && backend()->onAcquireBuffer(cache.get(), Backend::STATIC);
        if (!mValid) {
            return;
        }
        initWeight(mResource->mWeight->host<float>(), originWeight, cache->host<float>(), srcCount, outputCount, common->kernelX() * common->kernelY(), core);
        // MNN_PRINT("srcCount:%d, outputCount:%d, dense weight matrix tile:", srcCount, outputCount);
        // formatMatrix(mResource->mWeight->host<float>(), {UP_DIV(outputCount, hP), lSize, hP});
        backend()->onReleaseBuffer(cache.get(), Backend::STATIC);
    }
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
    auto dense = new DenseConvolutionTiledExecutor(mResource, op->main_as_Convolution2D()->common(), bn);
    dense->mProxy->mConvPerfconfig = mProxy->mConvPerfconfig;
    *dst = dense;
    return true;
}

ErrorCode DenseConvolutionTiledExecutor::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    bool needDequantize = mResource->mDequantize.bits <= 8;
    if (needDequantize) {
#ifndef MNN_ALLOC_MEMORY_INDIRECTLY
        auto res = backend()->onAcquireBuffer(mWeightCache.weight.get(), Backend::STATIC);
        if (!res) {
            return OUT_OF_MEMORY;
        }
        if (nullptr != mWeightCache.weightInt8) {
            res = backend()->onAcquireBuffer(mWeightCache.weightInt8.get(), Backend::STATIC);
            if (!res) {
                return OUT_OF_MEMORY;
            }
        }
#endif
        auto hU = mResource->hU;
        auto hP = mResource->hP;
        auto mid = mResource->lU * mResource->lP;
        auto srcInt8 = mResource->mWeight->host<int8_t>();
        if (mResource->mDequantize.bits == 4) {
            int weightLength = hU * hP * mid;
            weightLength = UP_DIV(weightLength, 2);
            auto srcPtr = mResource->mWeight->host<uint8_t>();
            auto dstPtr = mWeightCache.weightInt8->host<int8_t>();
            for (int i=0; i<weightLength; ++i) {
                int d = srcPtr[i];
                int s0 = d / 16;
                int s1 = d % 16;
                s0 = mResource->mDequantize.mLowBitWeightMap[s0];
                s1 = mResource->mDequantize.mLowBitWeightMap[s1];
                dstPtr[2 * i + 0] = s0;
                dstPtr[2 * i + 1] = s1;
            }
            srcInt8 = mWeightCache.weightInt8->host<int8_t>();
        }
        auto alpha = mResource->mDequantize.mScaleBias->host<float>();
        auto bias = mResource->mDequantize.mScaleBias->host<float>() + hU * hP;
        auto dstFloat = mWeightCache.weight->host<float>();
        for (int yo=0; yo<hU; ++yo) {
            auto dstY = dstFloat + yo * mid * hP;
            auto srcY = srcInt8 + yo * mid * hP;
            auto k = alpha + yo * hP;
            auto b = bias + yo * hP;
            for (int x=0; x<mid; ++x) {
                auto dstX = dstY + x * hP;
                auto srcX = srcY + x * hP;
                for (int yi=0; yi<hP; ++yi) {
                    dstX[yi] = srcX[yi] * k[yi] + b[yi];
                }
            }
        }
#ifndef MNN_ALLOC_MEMORY_INDIRECTLY
        if (mWeightCache.weightInt8 != nullptr) {
            backend()->onReleaseBuffer(mWeightCache.weightInt8.get(), Backend::STATIC);
        }
#endif
    }
    auto code = mProxy->onExecute(mInputs, outputs);
#ifndef MNN_ALLOC_MEMORY_INDIRECTLY
    if (needDequantize) {
        backend()->onReleaseBuffer(mWeightCache.weight.get(), Backend::STATIC);
    }
    ((Runtime*)(static_cast<CPUBackend*>(backend())->getRuntime()))->onGabageCollect(0);
#endif
    return code;
}
ErrorCode DenseConvolutionTiledExecutor::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mInputs = {inputs[0], mResource->mWeight.get(), mResource->mBias.get()};
    bool needDequantize = mResource->mDequantize.bits <= 8;
    if (needDequantize) {
        if (mWeightCache.weight == nullptr) {
            int weightLength = mResource->hU * mResource->lU * mResource->hP * mResource->lP;
            mWeightCache.weight.reset(new Tensor);
            mWeightCache.weight->buffer().type = halide_type_of<float>();
            TensorUtils::getDescribe(mWeightCache.weight.get())->dimensionFormat = MNN_DATA_FORMAT_NCHW;
            mWeightCache.weight->buffer().dimensions = 1;
            mWeightCache.weight->setLength(0, weightLength);
            if (mWeightCache.weightInt8 == nullptr && mResource->mDequantize.bits == 4) {
                mWeightCache.weightInt8.reset(new Tensor);
                mWeightCache.weightInt8->buffer().type = halide_type_of<int8_t>();
                mWeightCache.weightInt8->buffer().dimensions = 1;
                mWeightCache.weightInt8->setLength(0, weightLength);
                TensorUtils::getDescribe(mWeightCache.weightInt8.get())->dimensionFormat = MNN_DATA_FORMAT_NCHW;
            }
        }
        mInputs[1] = mWeightCache.weight.get();
#ifdef MNN_ALLOC_MEMORY_INDIRECTLY
        bool res = false;
        if (nullptr != mWeightCache.weightInt8) {
            res = backend()->onAcquireBuffer(mWeightCache.weightInt8.get(), Backend::DYNAMIC);
            if (!res) {
                return OUT_OF_MEMORY;
            }
        }
        res = backend()->onAcquireBuffer(mWeightCache.weight.get(), Backend::DYNAMIC);
        if (!res) {
            return OUT_OF_MEMORY;
        }
        if (nullptr != mWeightCache.weightInt8) {
            backend()->onReleaseBuffer(mWeightCache.weightInt8.get(), Backend::DYNAMIC);
        }
#endif
    }
    auto code = mProxy->onResize(mInputs, outputs);
    if (NO_ERROR != code) {
        return code;
    }
    if (needDequantize) {
#ifdef MNN_ALLOC_MEMORY_INDIRECTLY
        backend()->onReleaseBuffer(mWeightCache.weight.get(), Backend::DYNAMIC);
#endif
    }
    return NO_ERROR;
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

// #define PROFILE_DETAIL

PerfConfig DenseConvolutionTiledImpl::bestTileConvolutionConfig(const Convolution2DCommon *common, const Tensor *inputTensor,
                                          const Tensor *outputTensor, int threadNumber, Backend* b) {
    auto input   = inputTensor;
    Tensor *bias = nullptr;
    auto core    = static_cast<CPUBackend *>(b)->functions();
    int bytes    = core->bytes;
    int unit     = core->pack;
    int ePMax, lP, hP;
    core->MNNGetMatMulPackMode(&ePMax, &lP, &hP);
    auto kernel_width      = common->kernelX();
    auto kernel_height     = common->kernelY();
    auto output      = outputTensor;
    auto batch       = output->batch();
    auto width       = output->width();
    auto height      = output->height();
    auto src_width                = input->width();
    auto icC4                     = UP_DIV(input->channel(), unit);
    auto ic                       = input->channel();
    auto L                        = ic * common->kernelY() * common->kernelX();

    auto outputChannel = output->channel();
    auto padX = ConvolutionCommon::convolutionPad(inputTensor, outputTensor, common).first;
    if (src_width == 1 && width == 1 && height > 1 && kernel_width == 1 && padX == 0) {
        /* Swap x, y*/
        width         = height;
        height        = 1;
        kernel_width  = kernel_height;
        kernel_height = 1;
    }
    auto kernelSize               = common->kernelX() * common->kernelY();
    auto plane    = width * height * batch;
    auto oC4           = UP_DIV(outputChannel, unit);

     //In next major version these would be read from microbenchmark result file.
     constexpr int roofLine = 20;
     constexpr int indexCalculate = 3000;
     constexpr int indexMem = 40;

    PerfConfig denseConfig(false, 0, 0, 0, std::numeric_limits<float>().max());

    for (int eP = ePMax; eP >= ePMax; eP -= 16) { // search space should be open after pack-free dense is available.
        int tileCount = UP_DIV(plane, eP);
        auto hTileCount = UP_DIV(outputChannel, hP);

        float outerFlops[3], innerFlops[3], outerBandwidth[3], innerBandwidth[3], outer[3], inner[3], outerAcc = 0, innerAcc = 0;
        float tailCost = 0.0, lastTail = 0.0;

        if (plane % eP == 0) {
            tailCost = 1.0f;
            lastTail = 1.0f;
        } else {
            bool moreThanOnetail = tileCount % threadNumber > 1;
            lastTail = (4.f * (plane % eP)) / eP;
            tailCost = moreThanOnetail ? (std::max(1.0f, lastTail)) : lastTail;
        }

        float outerCoefficient = tailCost + ((tileCount - 1) / threadNumber);
        float innerCoefficient = lastTail + ((plane - 1) / eP);

        int indexNumber = UP_DIV(eP, width) * kernel_width * kernel_height;
        outerFlops[0] = outerCoefficient * indexNumber * indexCalculate * unit;
        outerFlops[1] = 0;
        outerFlops[2] = outerCoefficient * eP * (2 * L) * oC4 * unit;
        outerBandwidth[0] = outerCoefficient * indexNumber * indexMem;
        outerBandwidth[1] = outerCoefficient * indexNumber * (2 * eP * ic);
        outerBandwidth[2] = outerCoefficient * (eP * 2 * L + oC4 * unit * 2 *  L + eP * oC4 * unit);

        innerFlops[0] = innerCoefficient * indexNumber * indexCalculate * unit;
        innerFlops[1] = 0;
        innerFlops[2] = innerCoefficient * eP * (2 * L) * UP_DIV(oC4, threadNumber) * unit;
        innerBandwidth[0] = innerCoefficient * indexNumber * indexMem;
        innerBandwidth[1] = innerCoefficient * (2 * eP * unit + 10 * sizeof(int) * unit) * UP_DIV(icC4 * indexNumber, threadNumber);
        innerBandwidth[2] = innerCoefficient * (eP * 2 * L + unit * 2*  L + eP * unit) * UP_DIV(oC4, threadNumber);

        for (int i = 0; i < sizeof(outerFlops) / sizeof(float); i++) {
             outer[i] = std::max(outerBandwidth[i] * roofLine, outerFlops[i]);
             inner[i] = std::max(innerBandwidth[i] * roofLine, innerFlops[i]);
             outerAcc += outer[i];
             innerAcc += inner[i];
        }
        PerfConfig thisConfig(false, eP, eP, 0,  -1);
        thisConfig.isParallelInner = outerAcc > innerAcc;
        thisConfig.instructionCosts = outerAcc > innerAcc ? innerAcc : outerAcc;

        if (thisConfig.instructionCosts < denseConfig.instructionCosts) {
            denseConfig = thisConfig;
#ifdef PROFILE_DETAIL
            MNN_PRINT("\nouterFlops:");
            formatMatrix(outerFlops, {sizeof(outerFlops) / sizeof(float)});
            MNN_PRINT("\ninnerFlops:");
            formatMatrix(innerFlops, {sizeof(innerFlops) / sizeof(float)});
            MNN_PRINT("\nouterBandwidth:");
            formatMatrix(outerBandwidth, {sizeof(outerBandwidth) / sizeof(float)});
            MNN_PRINT("\ninnerBandwidth:");
            formatMatrix(innerBandwidth, {sizeof(innerBandwidth) / sizeof(float)});

            MNN_PRINT("\nouter:");
            formatMatrix(outer, {sizeof(outer) / sizeof(float)});
            MNN_PRINT("\ninner:");
            formatMatrix(inner, {sizeof(inner) / sizeof(float)});

            MNN_PRINT("\ndense im2col mParallelInner:%d, ePack:%d, outerAcc:%.1f, innerAcc:%.1f, totalCount:%d, tileCount:%d, outerCoefficient:%.2f, innerCoefficient:%.2f, tailCost:%.2f, lastTail:%.2f, allowed thread:%d, omp thread:\n\n",
                denseConfig.isParallelInner, eP, outerAcc, innerAcc, plane, tileCount, outerCoefficient, innerCoefficient, tailCost, lastTail,  threadNumber);
#endif
        }
    }

    return denseConfig;

}

ErrorCode DenseConvolutionTiledImpl::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
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
    auto src_width                = input->width();
    auto src_height               = input->height();
    auto icC4                     = UP_DIV(input->channel(), unit);
    auto ic                       = input->channel();
    auto L                        = ic * mCommon->kernelY() * mCommon->kernelX();
    int  LRoundup = ROUND_UP(L, lP);
    int  LRoundupC4 = UP_DIV(LRoundup, unit);
    auto outputChannel = output->channel();
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
    auto oC4           = UP_DIV(outputChannel, unit);
    mConvPerfconfig = bestTileConvolutionConfig(mCommon, input, output, threadNumber, backend());

    auto threadNumberFirst = mConvPerfconfig.isParallelInner ? threadNumber : std::min(threadNumber, tileCount);
    bool success = backend()->onAcquireBuffer(&mTempBufferTranspose, Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }

    auto bufferAlloc   = static_cast<CPUBackend *>(backend())->getBufferAllocator();
    auto maxLine       = UP_DIV(eP, width) + 1;
    auto tempPtr = bufferAlloc->alloc(kernelSize * maxLine * threadNumber * (4 * sizeof(int32_t) + sizeof(float *)));
    if (nullptr == tempPtr.first) {
        return OUT_OF_MEMORY;
    }
    backend()->onReleaseBuffer(&mTempBufferTranspose, Backend::DYNAMIC);
    bufferAlloc->free(tempPtr);

    auto postParameters    = getPostParameters();
    mFunction.first        = threadNumberFirst;

    if (mConvPerfconfig.isParallelInner) {

        mFunction.second = [=](int placeholder) {
#ifdef PROFILE_DETAIL
        MNN_PRINT("dense conv: n:%d, ih:%d, iw:%d, ic:%d, oh:%d, ow:%d, oc:%d, kh:%d, kw:%d, plane:%d, threadNumberFirst:%d, tileCount:%d, ePack:%d, pack::%d, bytes:%d\n",
        batch, src_height, src_width, ic, height, width, outputChannel, kernel_width, kernel_height, plane, threadNumberFirst, tileCount, eP, unit, bytes);
#endif

        auto gemmBuffer = mTempBufferTranspose.host<uint8_t>() + mTempBufferTranspose.stride(0) * 0;
        auto srcPtr     = (float const **)((uint8_t *)tempPtr.first + tempPtr.second +
                                       0 * kernelSize * maxLine * (4 * sizeof(int32_t) + sizeof(float *)));
        auto el         = (int32_t *)(srcPtr + kernelSize * maxLine);
        auto weightPtr = weight->host<uint8_t>();

        constexpr int InfoSize = 4;
        int32_t shapeInfo[InfoSize];
        int32_t* info = shapeInfo;
        info[1] = src_width * src_height * batch;
        info[2] = eP;
        info[3] = strideX;
        size_t shapeParameters[PARAMETERSIZE];
        size_t* parameters = shapeParameters;
        parameters[0]          = eP * bytes;
        parameters[1]          = L;
        parameters[2]          = outputChannel;
        parameters[3]          = plane * unit * bytes;
        parameters[4]          = 0;
        parameters[5]          = 0;

#ifdef PROFILE_DETAIL
        uint64_t durationMul[threadNumberFirst] = {0};
        uint64_t packATime[threadNumberFirst] = {0};
        uint64_t indexTime[threadNumberFirst] = {0};
        Timer timer[threadNumberFirst];
        double macs[threadNumberFirst] = {0};
#endif

        auto dstOrigin = output->host<uint8_t>();
        auto srcOrigin = input->host<uint8_t>();

        for (int x = 0; x < tileCount; x += 1) {
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
            int indexThread = std::min(threadNumberFirst, oyEnd - oyBegin + 1);

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
                            srcPtr[number]     = (const float*)srcKx;
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

#ifdef PROFILE_DETAIL
            indexTime[0] += timer[0].durationInUs();
            timer[0].reset();
#endif

            info[0] = 1;
            int hw4Stride = info[1] * unit * bytes;
            MNN_CONCURRENCY_BEGIN(tId, threadNumberFirst) {
                int threadEL[4];
                for(int tic_inumber = tId; tic_inumber < number * icC4; tic_inumber+=threadNumberFirst) {
                        int inumber = tic_inumber / icC4;
                        int t_ic = tic_inumber % icC4;
                        memcpy(threadEL, el + 4 * inumber, 4 * sizeof(int));
                        threadEL[1] = std::min(ic - (t_ic * unit), unit);
                        const float* source = (const float*)((const uint8_t*)(srcPtr[inumber]) + t_ic * hw4Stride);
                        auto gemmDest = gemmBuffer + t_ic * unit * eP * bytes;
                        packA((float *)gemmDest, &source, info, threadEL);
                }
            }
            MNN_CONCURRENCY_END();

#ifdef PROFILE_DETAIL
            packATime[0] += timer[0].durationInUs();
            timer[0].reset();
#endif

            if (xC == eP) {
                MNN_CONCURRENCY_BEGIN(tId, threadNumberFirst) {
                    size_t paraParameters[PARAMETERSIZE];
                    memcpy(paraParameters, parameters, PARAMETERSIZE * sizeof(size_t));
                    for (int t_oc = tId; t_oc < oC4; t_oc += threadNumberFirst) {
                        auto _dstFloatPtr = (float*)(dstOrigin + (t_oc * plane + start) * unit * bytes);
                        int ocIndex = t_oc * unit;
                        auto _weightFloatPtr = (const float*)(weightPtr + ((ocIndex / hP) * LRoundup * hP + ocIndex % hP) * bytes);
                        paraParameters[2] = std::min(outputChannel - (t_oc * unit), unit);
                        matmulUnit(_dstFloatPtr, (float*)gemmBuffer, _weightFloatPtr, paraParameters, postParameters.data(), biasPtr + ocIndex);
                    }
                }
                MNN_CONCURRENCY_END();
            } else {
                MNN_CONCURRENCY_BEGIN(tId, threadNumberFirst) {
                    size_t paraParameters[PARAMETERSIZE];
                    memcpy(paraParameters, parameters, PARAMETERSIZE * sizeof(size_t));
                    for (int t_oc = tId; t_oc < oC4; t_oc += threadNumberFirst) {
                        auto _dstFloatPtr = (float*)(dstOrigin + (t_oc * plane + start) * unit * bytes);
                        int ocIndex = t_oc * unit;
                        auto _weightFloatPtr = (const float*)(weightPtr + ((ocIndex / hP) * LRoundup * hP + ocIndex % hP) * bytes);
                        paraParameters[2] = std::min(outputChannel - (t_oc * unit), unit);
                        matmulRemain(_dstFloatPtr, (float*)gemmBuffer, _weightFloatPtr, xC, paraParameters, postParameters.data(), biasPtr + ocIndex);
                    }
                }
                MNN_CONCURRENCY_END();
            }

#ifdef PROFILE_DETAIL
         macs[0] += 2.0 * xC * L * oC4 * unit / threadNumberFirst;
         durationMul[0] += timer[0].durationInUs();
         timer[0].reset();
#endif

        }

#ifdef PROFILE_DETAIL
        double gflops = macs[0] / 1000.0 / durationMul[0];
        MNN_PRINT("dense conv mParallelInner:%d, inside measure: indexTime:%lu us, packATime:%lu us, durationMul:%lu us, total:%lu us, %.3f GFLOPS\n",
            mConvPerfconfig.isParallelInner, indexTime[0], packATime[0], durationMul[0], indexTime[0] + packATime[0] + durationMul[0], gflops);

#endif

    };

    } else {
        mFunction.second       = [=](int tId) {

#ifdef PROFILE_DETAIL
            if (tId == 0) {
                MNN_PRINT("dense conv: n:%d, ih:%d, iw:%d, ic:%d, oh:%d, ow:%d, oc:%d, kh:%d, kw:%d, plane:%d, tileCount:%d, ePack:%d, pack::%d, bytes:%d\n",
                batch, src_height, src_width, ic, height, width, outputChannel, kernel_width, kernel_height, plane, tileCount, eP, unit, bytes);
            }
#endif

            auto gemmBuffer = mTempBufferTranspose.host<uint8_t>() + mTempBufferTranspose.stride(0) * tId;
            auto srcPtr     = (float const **)((uint8_t *)tempPtr.first + tempPtr.second +
                                           tId * kernelSize * maxLine * (4 * sizeof(int32_t) + sizeof(float *)));
            auto el         = (int32_t *)(srcPtr + kernelSize * maxLine);
            auto weightPtr = weight->host<float>();
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

#ifdef PROFILE_DETAIL
            uint64_t durationMul[threadNumberFirst] = {0};
            uint64_t packATime[threadNumberFirst] = {0};
            uint64_t indexTime[threadNumberFirst] = {0};
            Timer timer[threadNumberFirst];
            double macs[threadNumberFirst] = {0};
#endif

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

#ifdef PROFILE_DETAIL
                indexTime[tId] += timer[tId].durationInUs();
                timer[tId].reset();
#endif
                if (number > 0) {
                    packA((float *)gemmBuffer, srcPtr, info, el);
                }

#ifdef PROFILE_DETAIL
                packATime[tId] += timer[tId].durationInUs();
                timer[tId].reset();
#endif
                if (xC == eP) {
                    matmulUnit((float*)(dstOrigin + start * unit * bytes), (float*)gemmBuffer, (float*)weightPtr, parameters,postParameters.data(), biasPtr);
                } else {
                    matmulRemain((float*)(dstOrigin + start * unit * bytes), (float*)gemmBuffer, (float*)weightPtr, xC, parameters,postParameters.data(), biasPtr);
                }

#ifdef PROFILE_DETAIL
            macs[tId] += 2.0 * xC * L * oC4 * unit; // bias
             durationMul[tId] += timer[tId].durationInUs();
             timer[tId].reset();
#endif
            }

#ifdef PROFILE_DETAIL
            double gflops = macs[tId] / 1000.0 / durationMul[tId];
            MNN_PRINT("dense conv mParallelInner:%d, inside measure: indexTime:%lu us, packATime:%lu us, durationMul:%lu us, total:%lu us, %.3f GFLOPS\n",
                mConvPerfconfig.isParallelInner, indexTime[tId], packATime[tId], durationMul[tId], indexTime[tId] + packATime[tId] + durationMul[tId], gflops);

#endif
        };
    }
    return NO_ERROR;
}

ErrorCode DenseConvolutionTiledImpl::onExecute(const std::vector<Tensor*>& inputs,
                                          const std::vector<Tensor*>& outputs) {
#ifdef PROFILE_DETAIL
    Timer outsideTimer;
    outsideTimer.reset();
#endif
    if (mConvPerfconfig.isParallelInner) {
        mFunction.second(0);
    } else {
        MNN_CONCURRENCY_BEGIN(tId, mFunction.first) {
            mFunction.second((int)tId);
        }
        MNN_CONCURRENCY_END();
    }

#ifdef PROFILE_DETAIL
    MNN_PRINT("dense conv. mParallelInner:%d, outside measure: total cost %lu us\n", mConvPerfconfig.isParallelInner, outsideTimer.durationInUs());
#endif
    return NO_ERROR;
}

#undef PROFILE_DETAIL

} // namespace MNN
