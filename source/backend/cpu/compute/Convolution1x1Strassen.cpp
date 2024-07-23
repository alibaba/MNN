//
//  Convolution1x1Strassen.cpp
//  MNN
//
//  Created by MNN on 2019/02/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Convolution1x1Strassen.hpp"
#include "DenseConvolutionTiledExecutor.hpp"
#include <string.h>
#include "core/BufferAllocator.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "core/Concurrency.h"
#include "ConvOpt.h"
#include "core/Macro.h"
#include "CommonOptFunction.h"
#include "core/TensorUtils.hpp"

namespace MNN {
Convolution1x1Strassen::Convolution1x1Strassen(const Convolution2DCommon *common, Backend *b, const float *originWeight,
                                               size_t originWeightSize, const float *bias, size_t biasSize, std::shared_ptr<ConvolutionCommon::Int8Common> quantInfo)
    : CPUConvolution(common, b) {
    auto outputCount = (int)biasSize;
    int ePack, lPack, hPack;
    auto core = static_cast<CPUBackend*>(b)->functions();
    core->MNNGetMatMulPackMode(&ePack, &lPack, &hPack);
    mResource.reset(new CPUConvolution::Resource);
    mResource->backend = b;
    auto mSrcCount   = (int)originWeightSize / outputCount;
    if (!mResource->copyBiasAlign(bias, (int)biasSize)) {
        MNN_ERROR("Not Enough Memory\n");
        mValid = false;
        return;
    }
#ifdef MNN_LOW_MEMORY
    if ((originWeightSize == 0 || nullptr == originWeight) && nullptr != quantInfo.get()) { // Use Int8 Weight.
        originWeightSize = quantInfo->weight.size();
        int lSize = (int)originWeightSize / (int)biasSize * common->kernelX() * common->kernelY();
        auto hU = UP_DIV(outputCount, hPack);
        auto lU = UP_DIV(lSize, lPack);
        mSrcCount   = (int)originWeightSize / outputCount;

        mResource->mWeight.reset(Tensor::createDevice<int8_t>(std::vector<int>{UP_DIV(outputCount, hPack), UP_DIV(mSrcCount, lPack) * lPack, hPack}));
        mValid = b->onAcquireBuffer(mResource->mWeight.get(), Backend::STATIC);
        if (!mValid) {
            MNN_ERROR("Not Enough Memory\n");
            return;
        }

        DenseConvolutionTiledExecutor::initQuantizeResource(quantInfo, mResource, hU, hPack, lU, lPack, outputCount, (int)originWeightSize / (int)biasSize, common->kernelX() * common->kernelY(), core->bytes);
        return;
    }
#endif
    // Use Float Weight.
    mResource->mWeight.reset(Tensor::createDevice<float>(std::vector<int>{UP_DIV(outputCount, hPack), UP_DIV(mSrcCount, lPack) * lPack, hPack}));
    mValid = b->onAcquireBuffer(mResource->mWeight.get(), Backend::STATIC);
    if (!mValid) {
        MNN_ERROR("Not Enough Memory\n");
        return;
    }
    if (core->bytes < 4) {
        AutoRelease<Tensor> tempTensor(Tensor::createDevice<float>({outputCount * mSrcCount}));
        mValid = b->onAcquireBuffer(tempTensor.get(), Backend::STATIC);
        if (!mValid) {
            MNN_ERROR("Not Enough Memory\n");
            return;
        }
        core->MNNFp32ToLowp(originWeight, tempTensor->host<int16_t>(), outputCount * mSrcCount);
        core->MNNPackForMatMul_B(mResource->mWeight->host<float>(), tempTensor->host<float>(), outputCount, mSrcCount, true);
        b->onReleaseBuffer(tempTensor.get(), Backend::STATIC);
    } else {
        core->MNNPackForMatMul_B(mResource->mWeight->host<float>(), originWeight, outputCount, mSrcCount, true);
    }
}
Convolution1x1Strassen::Convolution1x1Strassen(std::shared_ptr<CPUConvolution::Resource> resource, const Convolution2DCommon *common, Backend* b) : CPUConvolution(common, b) {
    mResource = resource;
}

Convolution1x1Strassen::~Convolution1x1Strassen() {
    // Do nothing
}

bool Convolution1x1Strassen::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    *dst = new Convolution1x1Strassen(mResource, op->main_as_Convolution2D()->common(), bn);
    return true;
}

ErrorCode Convolution1x1Strassen::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    CPUConvolution::onResize(inputs, outputs);
    auto core = static_cast<CPUBackend*>(backend())->functions();
    int ePack, lPack, hPack;
    core->MNNGetMatMulPackMode(&ePack, &lPack, &hPack);
    int bytes = core->bytes;
    auto CONVOLUTION_TILED_NUMBER = ePack;
    auto input       = inputs[0];
    auto output      = outputs[0];
    const int numberThread = ((CPUBackend *)backend())->threadNumber();
    auto ic = input->channel();
    auto oc = output->channel();
    auto icC4        = UP_DIV(ic, core->pack);
    auto ocC4        = UP_DIV(oc, core->pack);
    auto batch       = input->batch();
    auto matrixSizeE = output->height() * output->width() * input->batch();
    auto outputPlane = output->height() * output->width();
    mUnits.clear();
    std::shared_ptr<char> __autoFunction;
    auto padY     = mPadY;
    auto padX     = mPadX;
    auto strideX  = mCommon->strideX();
    auto strideY  = mCommon->strideY();
    auto postParameters = getPostParameters();
    auto memoryPool = ((CPUBackend *)backend())->getBufferAllocator();
    memoryPool->barrierBegin();
    std::shared_ptr<void> __a(nullptr, [memoryPool](void *) { memoryPool->barrierEnd(); });
    int maxDepth = 5;
    auto icAlign = UP_DIV(ic, lPack) * lPack;
    auto weightTensor = mResource->mWeight.get();
    uint8_t* dequantAlpha = nullptr;
    uint8_t* dequantBias = nullptr;
    int dequantBits = bytes * 8; // fp16:16, fp32:32
#ifdef MNN_LOW_MEMORY
    if (mResource && mResource->mDequantize.bits <= 8) {
        dequantAlpha = mResource->mDequantize.mScaleBias->host<uint8_t>();
        dequantBias = dequantAlpha + mResource->hU * mResource->hP * bytes;
        dequantBits = mResource->mDequantize.bits;
    }
#endif
    mWeightBytes = static_cast<float>(dequantBits) / 8.0f;
    auto rt = static_cast<const CPURuntime*>(backend()->getRuntime());
    if (matrixSizeE > CONVOLUTION_TILED_NUMBER * 8 * numberThread && matrixSizeE > ocC4) {
        std::vector<int> divides(numberThread+1);
        divides[0] = 0;
        rt->computeDivideSizes(matrixSizeE, divides.data()+1);
        mUnits.resize(numberThread);
        for (int i = 0; i < numberThread; ++i) {
            int planeStart = divides[i];
            int planeEnd   = divides[i+1];
            int planeSize  = planeEnd - planeStart;
            Unit &unit     = mUnits[i];
            if (planeSize <= 0) {
                unit.mValid = false;
                continue;
            }
            unit.offset[1] = 0;
            unit.offset[2] = 0;
            unit.offset[0] = core->pack * planeStart * bytes;
            unit.offset[3] = core->pack * planeStart * bytes;
            unit.mStracssenComputor.reset(new StrassenMatrixComputor(backend(), false, maxDepth, dequantAlpha, dequantBias, dequantBits));
            int e = planeSize;
            int l = ic;
            int h = oc;
            uint8_t* aPtr = nullptr;
            auto bPtr = TensorUtils::getDescribeOrigin(weightTensor)->mem->chunk();;
            uint8_t* cPtr = nullptr;
            auto biasPtr = TensorUtils::getDescribeOrigin(mResource->mBias.get())->mem->chunk();
            memoryPool->beginGroup();
            auto code = unit.mStracssenComputor->onEncode(e, l, h, matrixSizeE * core->pack, UP_DIV(l, lPack) * lPack * hPack, matrixSizeE * core->pack, aPtr, bPtr, cPtr, true, biasPtr, postParameters);
            if (NO_ERROR != code) {
                memoryPool->endGroup();
                return code;
            }
            memoryPool->endGroup();
        }
    } else {
        // Divide in ocC4
        auto hDiv = 1;
        if (hPack > core->pack) {
            hDiv = hPack / core->pack;
        }
        auto ocDiv = UP_DIV(ocC4, hDiv);
        std::vector<int> divides(numberThread+1);
        divides[0] = 0;
        rt->computeDivideSizes(ocDiv, divides.data()+1);
        mUnits.resize(numberThread);
        for (int i = 0; i < numberThread; ++i) {
            int ocStart = divides[i] * hDiv;
            int ocEnd = divides[i+1] * hDiv;
            if (ocEnd >= ocC4) {
                ocEnd = ocC4;
            }
            int ocSize  = ocEnd - ocStart;
            Unit &unit  = mUnits[i];
            if (ocSize <= 0) {
                unit.mValid = false;
                continue;
            }
            auto ocStartWeight = (ocStart * core->pack) / hPack;
            auto ocWeightSize = std::min(UP_DIV((ocSize * core->pack), hPack), mResource->mWeight->length(0) - ocStartWeight);
            unit.offset[1] = hPack * icAlign * ocStartWeight * mWeightBytes;
            unit.offset[2] = core->pack * ocStart * bytes;
            unit.offset[0] = 0;
            unit.offset[3] = core->pack * matrixSizeE * ocStart * bytes;

            unit.mStracssenComputor.reset(new StrassenMatrixComputor(backend(), false, maxDepth, dequantAlpha, dequantBias, dequantBits));
            int e = matrixSizeE;
            int l = ic;
            int h = std::min(ocSize * core->pack, ocWeightSize * hPack);
            uint8_t* aPtr = nullptr;
            auto bPtr = TensorUtils::getDescribeOrigin(mResource->mWeight.get())->mem->chunk() + hPack * icAlign * ocStartWeight * mWeightBytes;
            uint8_t* cPtr = nullptr;
            auto biasPtr = TensorUtils::getDescribeOrigin(mResource->mBias.get())->mem->chunk() + core->pack * ocStart * bytes;
            memoryPool->beginGroup();
            auto code = unit.mStracssenComputor->onEncode(e, l, h, matrixSizeE * core->pack, UP_DIV(l, lPack) * lPack * hPack, matrixSizeE * core->pack, aPtr, bPtr, cPtr, true, biasPtr, postParameters);
            if (NO_ERROR != code) {
                memoryPool->endGroup();
                return code;
            }
            memoryPool->endGroup();
        }
    }
    return NO_ERROR;
}

ErrorCode Convolution1x1Strassen::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto size   = mUnits.size();
    auto input  = inputs[0];
    auto output = outputs[0];
    auto core = static_cast<CPUBackend*>(backend())->functions();
    auto inputPtr = input->host<uint8_t>();
    auto outputPtr = output->host<uint8_t>();
    auto weightPtr = mResource->mWeight->host<uint8_t>();
    auto biasPtr = mResource->mBias->host<uint8_t>();

    MNN_CONCURRENCY_BEGIN(tId, size) {
        auto &unit = mUnits[tId];
        if (unit.mValid) {
            unit.mStracssenComputor->onExecute(inputPtr + unit.offset[0], weightPtr + unit.offset[1], biasPtr + unit.offset[2], outputPtr + unit.offset[3]);
        }
    }
    MNN_CONCURRENCY_END();
    return NO_ERROR;
}
} // namespace MNN
