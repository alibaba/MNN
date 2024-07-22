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
#include "core/MemoryFormater.h"
#define PARAMETERSIZE 7

using Vec4 = MNN::Math::Vec<float, 4>;
namespace MNN {

void DenseConvolutionTiledExecutor::initWeight(float *dest, const float *source, float* cache, int depth, int outputCount, int kernelSize, const CoreFunctions* function) {
    ConvolutionTiledExecutor::initWeight(source, cache, depth, outputCount, kernelSize, function);
    function->MNNPackForMatMul_B(dest, cache, outputCount, kernelSize * depth, true);

}
bool DenseConvolutionTiledExecutor::initQuantizeResource(std::shared_ptr<ConvolutionCommon::Int8Common> int8Info, std::shared_ptr<CPUConvolution::Resource> resource, int hU, int hP, int lU, int lP, int outputCount, int srcChannel, int kernelSize, int bytes) {
    int weightLength = hU * lU * hP * lP;
    resource->mWeight.reset(Tensor::createDevice<int8_t>(std::vector<int>{hU, lU * lP, hP}));
//    resource->mWeight.reset(Tensor::createDevice<uint8_t>({weightLength}));
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
    ::memset(dstWInt8, 0, resource->mWeight->usize());
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
    int dequantCnt = int8Info->alpha.size();
    int scaleSize = dequantCnt; // real size
    if (int8Info->asymmetric) {
        scaleSize = dequantCnt / 2;
        
    }
    int blockNum = scaleSize / outputCount;
    scaleSize = blockNum * hU * hP; // pack size
    resource->mDequantize.mScaleBias.reset(MNN::Tensor::createDevice<uint8_t>({scaleSize * 2 * bytes}));
    res = resource->backend->onAcquireBuffer(resource->mDequantize.mScaleBias.get(), Backend::STATIC);
    if (!res) {
        return false;
    }
    int originOffset = 0;
    if (int8Info->canUseInt4) {
        MNN_ASSERT(weightLength % 2 == 0);
        weightLength = UP_DIV(weightLength, 2);
        resource->mDequantize.bits = 4;
        std::shared_ptr<MNN::Tensor> weightLow(Tensor::createDevice<int8_t>(
            {weightLength}));
        auto res = resource->backend->onAcquireBuffer(weightLow.get(), Backend::STATIC);
        if (!res) {
            return false;
        }
        auto srcPtr = resource->mWeight->host<int8_t>();
        auto dstPtr = weightLow->host<unsigned char>();
        for (int i=0; i<weightLength; ++i) {
            int s0 = srcPtr[2 * i + 0] + 8;
            int s1 = srcPtr[2 * i + 1] + 8;
            int d = s0 * 16 + s1;
            dstPtr[i] = d;
        }
        originOffset = -8;
        resource->mWeight = weightLow;
    }
    auto alphaPtr = resource->mDequantize.mScaleBias->host<float>();
    auto biasPtr = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(alphaPtr) + scaleSize * bytes);
    ::memset(alphaPtr, 0, 2 * scaleSize * bytes);
    int h = int8Info->alpha.size();
    if (bytes == 2) {
        auto core = static_cast<CPUBackend*>(resource->backend)->functions();
        std::vector<float> tmpAlpha(scaleSize * 2, 0.0f);
        if (int8Info->asymmetric) {
            for (int i = 0; i < blockNum; ++i) {
                auto dstAlpha = tmpAlpha.data() + i * hU * hP;
                auto srcAlpha = int8Info->alpha.get();
                for (int j = 0; j < outputCount; ++j) {
                    int scaleIndex = j * blockNum + i;
                    dstAlpha[j] = srcAlpha[2 * scaleIndex + 1];
                    dstAlpha[j + scaleSize] = srcAlpha[2 * scaleIndex] + (float)originOffset * dstAlpha[j];
                }
            }
        } else {
            for (int i = 0; i < blockNum; ++i) {
                auto dstAlpha = tmpAlpha.data() + i * hU * hP;
                auto srcAlpha = int8Info->alpha.get();
                for (int j = 0; j < outputCount; ++j) {
                    int scaleIndex = j * blockNum + i;
                    dstAlpha[j] = srcAlpha[scaleIndex];
                    dstAlpha[j + scaleSize] = (float)originOffset * dstAlpha[j];
                }
            }
        }
        core->MNNFp32ToLowp(tmpAlpha.data(), reinterpret_cast<int16_t*>(alphaPtr), scaleSize * 2);
    } else {
        if (int8Info->asymmetric) {
            for (int i = 0; i < blockNum; ++i) {
                auto dstAlpha = alphaPtr + i * hU * hP;
                auto dstBias  = biasPtr + i * hU * hP;
                auto srcAlpha = int8Info->alpha.get();
                for (int j = 0; j < outputCount; ++j) {
                    int scaleIndex = j * blockNum + i;
                    dstAlpha[j] = srcAlpha[2 * scaleIndex + 1];
                    dstBias[j] = srcAlpha[2 * scaleIndex] + (float)originOffset * dstAlpha[j];
                }
            }
        } else {
            for (int i = 0; i < blockNum; ++i) {
                auto dstAlpha = alphaPtr + i * hU * hP;
                auto dstBias  = biasPtr + i * hU * hP;
                auto srcAlpha = int8Info->alpha.get();
                for (int j = 0; j < outputCount; ++j) {
                    int scaleIndex = j * blockNum + i;
                    dstAlpha[j] = srcAlpha[scaleIndex];
                    dstBias[j] = (float)originOffset * dstAlpha[j];
                }
            }
        }
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
        if (core->matmulBytes != 0) {
            bytes = core->matmulBytes;
        }
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
        thisConfig.isParallelInner = outerAcc > innerAcc && 0 == core->matmulBytes;
        thisConfig.instructionCosts = outerAcc > innerAcc ? innerAcc : outerAcc;

        if (thisConfig.instructionCosts < denseConfig.instructionCosts) {
            denseConfig = thisConfig;
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
    int matmulBytes = bytes;
    if (core->matmulBytes != 0) {
        matmulBytes = core->matmulBytes;
    }
    auto packA   = core->MNNPackC4ForMatMul_A;
    int eP, lP, hP;
    getPackParameter(&eP, &lP, &hP, core);
    auto matmulUnit   = core->MNNPackedMatMul;
    auto matmulRemain = core->MNNPackedMatMulRemain;
    const uint8_t* dequantAlpha = nullptr;
    const uint8_t* dequantBias = nullptr;
    auto ic       = input->channel();
    auto icC4     = UP_DIV(ic, unit);
    auto L        = ic * mCommon->kernelY() * mCommon->kernelX();
    auto tileC    = std::max(unit, hP);
    int blockSize = L;
    int blockNum  = 1;
    float halfStride = 1;
    size_t weightStride = 0;
#ifdef MNN_LOW_MEMORY
    if (mResource && mResource->mDequantize.bits <= 8) {
        DenseConvolutionTiledExecutor::selectLowMemoryMatmulFunc(&matmulUnit, &matmulRemain, &weightBytes, mResource->mDequantize.bits, core);
        int scaleSize = mResource->mDequantize.mScaleBias->size() / (2 * bytes);
        blockNum = scaleSize / (mResource->hU * mResource->hP);
        blockSize /= blockNum;
        dequantAlpha = mResource->mDequantize.mScaleBias->host<uint8_t>();
        dequantBias = dequantAlpha + scaleSize * bytes;
        weightStride = (L - blockSize) * hP;
        if (mResource->mDequantize.bits == 4) {
            halfStride = 0.5;
            weightStride = static_cast<size_t>(weightStride * halfStride);
        }
    }
#endif
    auto kernel_width      = mCommon->kernelX();
    auto kernel_height     = mCommon->kernelY();
    auto output      = outputs[0];
    auto batch       = output->batch();
    int threadNumber = ((CPUBackend *)backend())->threadNumber();
    
    int  LRoundup = ROUND_UP(L, lP);
    int  LRoundupC4 = UP_DIV(LRoundup, unit);
    auto outputChannel = output->channel();
    auto oC4      = UP_DIV(outputChannel, tileC);
    auto ocUp4    = ROUND_UP(outputChannel, hP);
    auto kernelSize               = mCommon->kernelX() * mCommon->kernelY();

    ConvolutionTiledExecutor::setIm2ColParameter(mIm2ColParameters, mCommon, input, output, mPadX, mPadY, core, nullptr);
    mTempBufferTranspose.buffer().type          = halide_type_of<uint8_t>();
    mTempBufferTranspose.buffer().dimensions    = 2;
    mTempBufferTranspose.buffer().dim[0].extent = threadNumber;
    mTempBufferTranspose.buffer().dim[1].extent = UP_DIV(L, lP) * lP * eP * matmulBytes;
    TensorUtils::setLinearLayout(&mTempBufferTranspose);
    auto plane    = mIm2ColParameters.ow * mIm2ColParameters.oh * batch;
    int tileCount = UP_DIV(plane, eP);
    mConvPerfconfig = bestTileConvolutionConfig(mCommon, input, output, threadNumber, backend());
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
    mFunction.first        = threadNumber;

    if (mConvPerfconfig.isParallelInner) {
        auto rt = static_cast<const CPURuntime*>(backend()->getRuntime());
        std::vector<int> ocC4ParralSize(threadNumber + 1);
        ocC4ParralSize[0] = 0;
        rt->computeDivideSizes(oC4, ocC4ParralSize.data()+1);
        mFunction.second = [=](int placeholder) {
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
        parameters[1]          = blockSize;
        parameters[2]          = outputChannel;
        parameters[3]          = plane * unit * bytes;
        parameters[4]          = 0;
        parameters[5]          = weightStride; // Only used when block quant
        parameters[6]          = 0;

        auto dstOrigin = output->host<uint8_t>();
        auto srcOrigin = input->host<uint8_t>();
        std::vector<int> im2colParallelSize(threadNumber + 1);
        im2colParallelSize[0] = 0;

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
            info[0] = 1;
            int hw4Stride = info[1] * unit * bytes;
            rt->computeDivideSizes(number * icC4, im2colParallelSize.data() + 1);
            im2colParallelSize[0] = 0;
            MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
                int threadEL[4];
                int ticSta = im2colParallelSize[tId];
                int ticEnd = im2colParallelSize[tId+1];
                for(int tic_inumber = ticSta; tic_inumber < ticEnd; tic_inumber++) {
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

            if (xC == eP) {
                MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
                    size_t paraParameters[PARAMETERSIZE];
                    memcpy(paraParameters, parameters, PARAMETERSIZE * sizeof(size_t));
                    for (int t_oc = ocC4ParralSize[tId]; t_oc < ocC4ParralSize[tId+1]; ++t_oc) {
                        int ocIndex = t_oc * tileC;
                        auto _dstFloatPtr = reinterpret_cast<float*>(dstOrigin + (ocIndex / unit * plane + start) * unit * bytes);
                        auto _weightFloatPtr = reinterpret_cast<const float*>(weightPtr + int((ocIndex / hP * LRoundup * hP) * weightBytes));
                        auto _biasFloatPtr = reinterpret_cast<const float*>(reinterpret_cast<const uint8_t*>(biasPtr) + ocIndex * bytes);
                        paraParameters[2] = std::min(outputChannel - ocIndex, tileC);
                        auto k = reinterpret_cast<const uint8_t*>(dequantAlpha + ocIndex * bytes);
                        auto b = reinterpret_cast<const uint8_t*>(dequantBias + ocIndex * bytes);
                        const float* relufp32 = nullptr;
                        const float* exeBiasPtr = nullptr;
                        int finishedL = 0;
                        int wquantStride = 0;
                        auto _weightPtr = reinterpret_cast<const int8_t*>(_weightFloatPtr);
                        uint8_t*  _APtr      = reinterpret_cast<uint8_t*>(gemmBuffer);
                        for (int bk = 0; bk < blockNum; ++bk) {
                            paraParameters[6] = bk;
                            if (bk == blockNum - 1) {
                                relufp32 = postParameters.data();
                                exeBiasPtr = _biasFloatPtr;
                            }
                            finishedL = blockSize * bk;
                            wquantStride = static_cast<int32_t>(blockSize * bk * hP * halfStride);
                            matmulUnit(_dstFloatPtr, (float*)(_APtr + eP * finishedL * bytes), (float*)(_weightPtr + wquantStride), paraParameters, relufp32, exeBiasPtr, (float*)(k + bk * ocUp4 * bytes), (float*)(b + bk * ocUp4 * bytes));
                        }
                    }
                }
                MNN_CONCURRENCY_END();
            } else {
                MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
                    size_t paraParameters[PARAMETERSIZE];
                    memcpy(paraParameters, parameters, PARAMETERSIZE * sizeof(size_t));
                    for (int t_oc = ocC4ParralSize[tId]; t_oc < ocC4ParralSize[tId+1]; ++t_oc) {
                        int ocIndex = t_oc * tileC;
                        auto _dstFloatPtr = reinterpret_cast<float*>(dstOrigin + (ocIndex / unit * plane + start) * unit * bytes);
                        auto _weightFloatPtr = reinterpret_cast<const float*>(weightPtr + int((ocIndex / hP * LRoundup * hP) * weightBytes));
                        auto _biasFloatPtr = reinterpret_cast<const float*>(reinterpret_cast<const uint8_t*>(biasPtr) + ocIndex * bytes);
                        paraParameters[2] = std::min(outputChannel - ocIndex, tileC);
                        auto k = reinterpret_cast<const uint8_t*>(dequantAlpha + ocIndex * bytes);
                        auto b = reinterpret_cast<const uint8_t*>(dequantBias + ocIndex * bytes);
                        const float* relufp32 = nullptr;
                        const float* exeBiasPtr = nullptr;
                        int finishedL = 0;
                        int wquantStride = 0;
                        const int8_t* _weightPtr = reinterpret_cast<const int8_t*>(_weightFloatPtr);
                        uint8_t*  _APtr      = reinterpret_cast<uint8_t*>(gemmBuffer);
                        for (int bk = 0; bk < blockNum; ++bk) {
                            paraParameters[6] = bk;
                            if (bk == blockNum - 1) {
                                relufp32 = postParameters.data();
                                exeBiasPtr = _biasFloatPtr;
                            }
                            finishedL = blockSize * bk;
                            wquantStride = static_cast<int32_t>(blockSize * bk * hP * halfStride);
                            matmulRemain(_dstFloatPtr, (float*)(_APtr + eP * finishedL * bytes), (float*)(_weightPtr + wquantStride), xC, paraParameters, relufp32, exeBiasPtr, (float*)(k + bk * ocUp4 * bytes), (float*)(b + bk * ocUp4 * bytes));
                        }
                    }
                }
                MNN_CONCURRENCY_END();
            }

        }
    };

    } else {
        std::vector<int> divides(threadNumber + 1);
        divides[0] = 0;

        static_cast<const CPURuntime*>(static_cast<CPUBackend*>(backend())->getRuntime())->computeDivideSizes(tileCount, divides.data() + 1);

        mFunction.second       = [=](int tId) {
            const float* biasPtr = bias ? bias->host<float>() : nullptr;
            auto gemmBuffer = mTempBufferTranspose.host<uint8_t>() + mTempBufferTranspose.stride(0) * tId;
            auto srcPtr     = (float const **)(tempPtr.ptr() + tId * kernelSize * maxLine * (4 * sizeof(int32_t) + sizeof(float *)));
            auto el         = (int32_t *)(srcPtr + kernelSize * maxLine);
            auto weightPtr = weight->host<float>();
            int32_t info[4];
            info[1] = mIm2ColParameters.iw * mIm2ColParameters.ih * batch;
            info[2] = eP;
            info[3] = mIm2ColParameters.strideX;
            size_t parameters[PARAMETERSIZE];
            parameters[0]          = eP * bytes;
            parameters[1]          = blockSize;
            parameters[2]          = outputChannel;
            parameters[3]          = plane * unit * bytes;
            parameters[4]          = 0;
            parameters[5]          = weightStride; // Only used when block quant
            parameters[6]          = 0;

            auto dstOrigin = output->host<uint8_t>();
            auto srcOrigin = input->host<uint8_t>();
            int tEnd = divides[tId+1];
            int tStart = divides[tId];
            for (int x = (int)tStart; x < tEnd; ++x) {
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

                if (number > 0) {
                    packA((float *)gemmBuffer, srcPtr, info, el);
                }

                int finishedL = 0;
                int wquantStride = 0;
                int8_t* _weightPtr = reinterpret_cast<int8_t*>(weightPtr);
                auto _dstFloatPtr = reinterpret_cast<float*>(dstOrigin + start * unit * bytes);
                const float* relufp32 = nullptr;
                const float* exeBiasPtr = nullptr;
                if (xC == eP) {
                    // matmulUnit(_dstFloatPtr, (float*)gemmBuffer, (float*)weightPtr, parameters, postParameters.data(), biasPtr, k, b);
                    for (int bk = 0; bk < blockNum; ++bk) {
                        parameters[6] = bk;
                        if (bk == blockNum - 1) {
                            relufp32 = postParameters.data();
                            exeBiasPtr = biasPtr;
                        }
                        finishedL = blockSize * bk;
                        wquantStride = static_cast<int32_t>(blockSize * bk * hP * halfStride);
                        
                        matmulUnit(_dstFloatPtr, (float*)(gemmBuffer + bytes * eP * finishedL), (float*)(_weightPtr + wquantStride), parameters, relufp32, exeBiasPtr, (float*)(dequantAlpha + bk * ocUp4 * bytes), (float*)(dequantBias + bk * ocUp4 * bytes));
                    }
                } else {
                    for (int bk = 0; bk < blockNum; ++bk) {
                        parameters[6] = bk;
                        if (bk == blockNum - 1) {
                            relufp32 = postParameters.data();
                            exeBiasPtr = biasPtr;
                        }
                        finishedL = blockSize * bk;
                        wquantStride = static_cast<int32_t>(blockSize * bk * hP * halfStride);
                        
                        matmulRemain(_dstFloatPtr, (float*)(gemmBuffer + eP * bytes * finishedL), (float*)(_weightPtr + wquantStride), xC, parameters, relufp32, exeBiasPtr, (float*)(dequantAlpha + bk * ocUp4 * bytes), (float*)(dequantBias + bk * ocUp4 * bytes ));
                    }
                    // matmulRemain(_dstFloatPtr, (float*)gemmBuffer, (float*)weightPtr, xC, parameters, postParameters.data(), biasPtr, k, b);
                }
            }
        };
    }
    return NO_ERROR;
}

ErrorCode DenseConvolutionTiledImpl::onExecute(const std::vector<Tensor*>& inputs,
                                          const std::vector<Tensor*>& outputs) {
    if (mConvPerfconfig.isParallelInner) {
        mFunction.second(0);
    } else {
        MNN_CONCURRENCY_BEGIN(tId, mFunction.first) {
            mFunction.second((int)tId);
        }
        MNN_CONCURRENCY_END();
    }

    return NO_ERROR;
}


} // namespace MNN
