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

using Vec4 = MNN::Math::Vec<float, 4>;
namespace MNN {

void DenseConvolutionTiledExecutor::initWeight(float *dest, const float *source, float* cache, int depth, int outputCount, int kernelSize, const CoreFunctions* function) {
    ConvolutionTiledExecutor::initWeight(source, cache, depth, outputCount, kernelSize, function);
    function->MNNPackForMatMul_B(dest, cache, outputCount, kernelSize * depth, true);

}
bool DenseConvolutionTiledExecutor::initQuantizeResource(std::shared_ptr<ConvolutionCommon::Int8Common> int8Info, std::shared_ptr<CPUConvolution::Resource> resource, int hU, int hP, int lU, int lP, int outputCount, int srcChannel, int kernelSize, int bytes) {
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
    MNN_ASSERT(lP == 1);
    // Reorder weight

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
    auto biasPtr = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(alphaPtr) + hU * hP * bytes);
    ::memset(alphaPtr, 0, 2 * hU * hP * bytes);
    int h = int8Info->alpha.size();
    if (bytes == 2) {
        auto core = static_cast<CPUBackend*>(resource->backend)->functions();
        if (int8Info->asymmetric) {
            std::unique_ptr<int16_t[]> tmp(new int16_t[h]);
            core->MNNFp32ToLowp(int8Info->alpha.get(), tmp.get(), h);
            for (int i=0; i< h/2; ++i) {
                reinterpret_cast<int16_t*>(alphaPtr)[i] = tmp[2 * i + 1];
                reinterpret_cast<int16_t*>(biasPtr)[i] = tmp[2 * i];
            }
        } else {
            core->MNNFp32ToLowp(int8Info->alpha.get(), reinterpret_cast<int16_t*>(alphaPtr), h);
        }
    } else {
        if (int8Info->asymmetric) {
            h = h / 2;
            for (int i=0; i<h; ++i) {
                alphaPtr[i] = int8Info->alpha.get()[2 * i + 1];
                biasPtr[i] = int8Info->alpha.get()[2 * i];
            }
        } else {
            for (int i=0; i<h; ++i) {
                alphaPtr[i] = int8Info->alpha.get()[i];
                biasPtr[i] = 0.f;
            }
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
            int d = (s0 + 8) * 16 + (s1 + 8);
            dstPtr[i] = d;
        }
        resource->mWeight = weightLow;
    }
    return true;
}

void DenseConvolutionTiledExecutor::selectLowMemoryMatmulFunc(lowMemoryMatmulUnit* matmulUnit, lowMemoryMatmulRemain* matmulRemain, float* weightBytes, int32_t weightQuantBits, const CoreFunctions* core) {
    if (weightQuantBits == 8) {
        *matmulUnit = core->MNNPackedMatMul_int8;
        *matmulRemain = core->MNNPackedMatMulRemain_int8;
        *weightBytes  = 1;
    }
    if (weightQuantBits == 4) {
        *matmulUnit   = core->MNNPackedMatMul_int4;
        *matmulRemain = core->MNNPackedMatMulRemain_int4;
        *weightBytes  = 0.5;
    }
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
        auto allocSuccess = DenseConvolutionTiledExecutor::initQuantizeResource(int8Info, mResource, hU, hP, lU, lP, outputCount, srcCount, common->kernelX() * common->kernelY(), bytes);
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
    mProxy.reset(new DenseConvolutionTiledImpl(common, b, mResource.get()));
}

DenseConvolutionTiledExecutor::DenseConvolutionTiledExecutor(std::shared_ptr<CPUConvolution::Resource> res, const Convolution2DCommon* common, Backend* b) : ConvolutionTiledExecutor(res, b) {
    mProxy.reset(new DenseConvolutionTiledImpl(common, b, mResource.get()));
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
    auto code = mProxy->onExecute(mInputs, outputs);
    return code;
}
ErrorCode DenseConvolutionTiledExecutor::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mInputs = {inputs[0], mResource->mWeight.get(), mResource->mBias.get()};
    auto code = mProxy->onResize(mInputs, outputs);
    if (NO_ERROR != code) {
        return code;
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
    if (inputs.size() > 2) {
        bias = inputs[2];
    }
    auto core    = static_cast<CPUBackend *>(backend())->functions();
    int bytes    = core->bytes;
    float weightBytes  = bytes;
    int unit     = core->pack;
    auto packA   = core->MNNPackC4ForMatMul_A;
    int eP, lP, hP;
    getPackParameter(&eP, &lP, &hP, core);
    auto matmulUnit   = core->MNNPackedMatMul;
    auto matmulRemain = core->MNNPackedMatMulRemain;
    auto weightType = weight->getType();
    const uint8_t* dequantAlpha = nullptr;
    const uint8_t* dequantBias = nullptr;
#ifdef MNN_LOW_MEMORY
    if (mResource && mResource->mDequantize.bits <= 8) {
        DenseConvolutionTiledExecutor::selectLowMemoryMatmulFunc(&matmulUnit, &matmulRemain, &weightBytes, mResource->mDequantize.bits, core);
        dequantAlpha = mResource->mDequantize.mScaleBias->host<uint8_t>();
        dequantBias = dequantAlpha + mResource->hU * mResource->hP * bytes;
    }
#endif
    auto kernel_width      = mCommon->kernelX();
    auto kernel_height     = mCommon->kernelY();
    auto output      = outputs[0];
    auto batch       = output->batch();
    int threadNumber = ((CPUBackend *)backend())->threadNumber();
    auto icC4                     = UP_DIV(input->channel(), unit);
    auto ic                       = input->channel();
    auto L                        = ic * mCommon->kernelY() * mCommon->kernelX();
    int  LRoundup = ROUND_UP(L, lP);
    int  LRoundupC4 = UP_DIV(LRoundup, unit);
    auto outputChannel = output->channel();
    auto tileC    = std::max(unit, hP);
    auto oC4      = UP_DIV(outputChannel, tileC);
    auto kernelSize               = mCommon->kernelX() * mCommon->kernelY();

    ConvolutionTiledExecutor::setIm2ColParameter(mIm2ColParameters, mCommon, input, output, mPadX, mPadY, core, nullptr);
    mTempBufferTranspose.buffer().type          = halide_type_of<uint8_t>();
    mTempBufferTranspose.buffer().dimensions    = 2;
    mTempBufferTranspose.buffer().dim[0].extent = threadNumber;
    mTempBufferTranspose.buffer().dim[1].extent = UP_DIV(L, lP) * lP * eP * bytes;
    TensorUtils::setLinearLayout(&mTempBufferTranspose);
    auto plane    = mIm2ColParameters.ow * mIm2ColParameters.oh * batch;
    int tileCount = UP_DIV(plane, eP);
    mConvPerfconfig = bestTileConvolutionConfig(mCommon, input, output, threadNumber, backend());

    auto threadNumberFirst = mConvPerfconfig.isParallelInner ? threadNumber : std::min(threadNumber, tileCount);
    bool success = backend()->onAcquireBuffer(&mTempBufferTranspose, Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }

    auto bufferAlloc   = static_cast<CPUBackend *>(backend())->getBufferAllocator();
    auto maxLine       = UP_DIV(eP, mIm2ColParameters.ow) + 1;
    auto tempPtr = bufferAlloc->alloc(kernelSize * maxLine * threadNumber * (4 * sizeof(int32_t) + sizeof(float *)));
    if (tempPtr.invalid()) {
        return OUT_OF_MEMORY;
    }
    backend()->onReleaseBuffer(&mTempBufferTranspose, Backend::DYNAMIC);
    bufferAlloc->free(tempPtr);

    auto postParameters    = getPostParameters();
    mFunction.first        = threadNumberFirst;

    if (mConvPerfconfig.isParallelInner) {

        mFunction.second = [=](int placeholder) {
#ifdef PROFILE_DETAIL
        MNN_PRINT("dense conv: n:%d, ic:%d, oc:%d, kh:%d, kw:%d, plane:%d, threadNumberFirst:%d, tileCount:%d, ePack:%d, pack::%d, bytes:%d\n",
        batch, ic, outputChannel, kernel_width, kernel_height, plane, threadNumberFirst, tileCount, eP, unit, bytes);
#endif
        const float* biasPtr = bias ? bias->host<float>() : nullptr;
        auto gemmBuffer = mTempBufferTranspose.host<uint8_t>() + mTempBufferTranspose.stride(0) * 0;
        auto srcPtr     = (float const **)(tempPtr.ptr() + 0 * kernelSize * maxLine * (4 * sizeof(int32_t) + sizeof(float *)));
        auto el         = (int32_t *)(srcPtr + kernelSize * maxLine);
        auto weightPtr = weight->host<uint8_t>();

        constexpr int InfoSize = 4;
        int32_t shapeInfo[InfoSize];
        int32_t* info = shapeInfo;
        info[1] = mIm2ColParameters.iw * mIm2ColParameters.ih * batch;
        info[2] = eP;
        info[3] = mIm2ColParameters.strideX;
        size_t shapeParameters[PARAMETERSIZE];
        size_t* parameters = shapeParameters;
        parameters[0]          = eP * bytes;
        parameters[1]          = L;
        parameters[2]          = outputChannel;
        parameters[3]          = plane * unit * bytes;
        parameters[4]          = 0;
        parameters[5]          = 0;

#ifdef PROFILE_DETAIL
        std::vector<uint64_t> durationMul(threadNumberFirst, 0);
        std::vector<uint64_t> packATime(threadNumberFirst, 0);
        std::vector<uint64_t> indexTime(threadNumberFirst, 0);
        Timer timer[threadNumberFirst];
        std::vector<double> macs(threadNumberFirst, 0);
#endif

        auto dstOrigin = output->host<uint8_t>();
        auto srcOrigin = input->host<uint8_t>();

        for (int x = 0; x < tileCount; x += 1) {
            int start  = (int)x * eP;
            int remain = plane - start;
            int xC     = remain > eP ? eP : remain;
            auto res = ConvolutionTiledExecutor::turnIm2ColToBlitInfo(srcPtr, el, start, xC, mIm2ColParameters, srcOrigin, bytes);
            int number    = res.first;
            bool needZero = res.second;
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
                        int ocIndex = t_oc * tileC;
                        auto _dstFloatPtr = reinterpret_cast<float*>(dstOrigin + (ocIndex / unit * plane + start) * unit * bytes);
                        auto _weightFloatPtr = reinterpret_cast<const float*>(weightPtr + int((ocIndex / hP * LRoundup * hP) * weightBytes));
                        auto _biasFloatPtr = reinterpret_cast<const float*>(reinterpret_cast<const uint8_t*>(biasPtr) + ocIndex * bytes);
                        paraParameters[2] = std::min(outputChannel - ocIndex, tileC);
                        auto k = reinterpret_cast<const float*>(dequantAlpha + ocIndex * bytes);
                        auto b = reinterpret_cast<const float*>(dequantBias + ocIndex * bytes);
                        matmulUnit(_dstFloatPtr, (float*)gemmBuffer, _weightFloatPtr, paraParameters, postParameters.data(), _biasFloatPtr, k, b);
                    }
                }
                MNN_CONCURRENCY_END();
            } else {
                MNN_CONCURRENCY_BEGIN(tId, threadNumberFirst) {
                    size_t paraParameters[PARAMETERSIZE];
                    memcpy(paraParameters, parameters, PARAMETERSIZE * sizeof(size_t));
                    for (int t_oc = tId; t_oc < oC4; t_oc += threadNumberFirst) {
                        int ocIndex = t_oc * tileC;
                        auto _dstFloatPtr = reinterpret_cast<float*>(dstOrigin + (ocIndex / unit * plane + start) * unit * bytes);
                        auto _weightFloatPtr = reinterpret_cast<const float*>(weightPtr + int((ocIndex / hP * LRoundup * hP) * weightBytes));
                        auto _biasFloatPtr = reinterpret_cast<const float*>(reinterpret_cast<const uint8_t*>(biasPtr) + ocIndex * bytes);
                        paraParameters[2] = std::min(outputChannel - ocIndex, tileC);
                        auto k = reinterpret_cast<const float*>(dequantAlpha + ocIndex * bytes);
                        auto b = reinterpret_cast<const float*>(dequantBias + ocIndex * bytes);
                        matmulRemain(_dstFloatPtr, (float*)gemmBuffer, _weightFloatPtr, xC, paraParameters, postParameters.data(), _biasFloatPtr, k, b);
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
                MNN_PRINT("dense conv: n:%d, ic:%d, oc:%d, kh:%d, kw:%d, plane:%d, tileCount:%d, ePack:%d, pack::%d, bytes:%d\n",
                batch, ic, outputChannel, kernel_width, kernel_height, plane, tileCount, eP, unit, bytes);
            }
#endif
            const float* biasPtr = bias ? bias->host<float>() : nullptr;
            auto gemmBuffer = mTempBufferTranspose.host<uint8_t>() + mTempBufferTranspose.stride(0) * tId;
            auto srcPtr     = (float const **)(tempPtr.ptr() + tId * kernelSize * maxLine * (4 * sizeof(int32_t) + sizeof(float *)));
            auto el         = (int32_t *)(srcPtr + kernelSize * maxLine);
            auto weightPtr = weight->host<float>();
            int32_t info[4];
            info[1] = mIm2ColParameters.iw * mIm2ColParameters.ih * batch;
            info[2] = eP;
            info[3] = mIm2ColParameters.strideX;
            size_t parameters[6];
            parameters[0]          = eP * bytes;
            parameters[1]          = L;
            parameters[2]          = outputChannel;
            parameters[3]          = plane * unit * bytes;
            parameters[4]          = 0;
            parameters[5]          = 0;

#ifdef PROFILE_DETAIL
            std::vector<uint64_t> durationMul(threadNumberFirst, 0);
            std::vector<uint64_t> packATime(threadNumberFirst, 0);
            std::vector<uint64_t> indexTime(threadNumberFirst, 0);
            Timer timer[threadNumberFirst];
            std::vector<double> macs(threadNumberFirst, 0);
#endif

            auto dstOrigin = output->host<uint8_t>();
            auto srcOrigin = input->host<uint8_t>();
            for (int x = (int)tId; x < tileCount; x += threadNumberFirst) {
                int start  = (int)x * eP;
                int remain = plane - start;
                int xC     = remain > eP ? eP : remain;
                auto res = ConvolutionTiledExecutor::turnIm2ColToBlitInfo(srcPtr, el, start, xC, mIm2ColParameters, srcOrigin, bytes);
                auto number = res.first;
                bool needZero = res.second;
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
                auto k =  reinterpret_cast<const float*>(dequantAlpha);
                auto b =  reinterpret_cast<const float*>(dequantBias);
                auto _dstFloatPtr = reinterpret_cast<float*>(dstOrigin + start * unit * bytes);
                if (xC == eP) {
                    matmulUnit(_dstFloatPtr, (float*)gemmBuffer, (float*)weightPtr, parameters, postParameters.data(), biasPtr, k, b);
                } else {
                    matmulRemain(_dstFloatPtr, (float*)gemmBuffer, (float*)weightPtr, xC, parameters, postParameters.data(), biasPtr, k, b);
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
