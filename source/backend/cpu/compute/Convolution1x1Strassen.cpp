//
//  Convolution1x1Strassen.cpp
//  MNN
//
//  Created by MNN on 2019/02/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Convolution1x1Strassen.hpp"
#include <string.h>
#include "core/BufferAllocator.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "CommonOptFunction.h"
#include "core/Concurrency.h"
#include "ConvOpt.h"
#include "core/Macro.h"
namespace MNN {
void Convolution1x1Strassen::_init(const Convolution2DCommon *common, Backend *b, const float *originWeight, size_t originWeightSize, const float *bias, size_t biasSize) {
    mPostFunction    = CPUConvolution::getPostFunction();
    auto outputCount = (int)biasSize;
    auto mSrcCount   = (int)originWeightSize / outputCount;
    int ePack, lPack, hPack;
    MNNGetMatMulPackMode(&ePack, &lPack, &hPack);
    mWeight.reset(Tensor::createDevice<float>(std::vector<int>{UP_DIV(outputCount, hPack), UP_DIV(mSrcCount, lPack), lPack * hPack}));
    mValid = b->onAcquireBuffer(mWeight.get(), Backend::STATIC);
    if (!mValid) {
        MNN_ERROR("Not Enough Memory\n");
        return;
    }
    MNNPackForMatMul_B(mWeight->host<float>(), originWeight, outputCount, mSrcCount, true);
    mBias.reset(Tensor::createDevice<float>(std::vector<int>{UP_DIV(outputCount, 4), 4}));
    mValid = b->onAcquireBuffer(mBias.get(), Backend::STATIC);
    if (!mValid) {
        MNN_ERROR("Not Enough Memory\n");
        return;
    }
    ::memset(mBias->host<float>(), 0, mBias->size());
    ::memcpy(mBias->host<float>(), bias, biasSize * sizeof(float));
}

Convolution1x1Strassen::Convolution1x1Strassen(const Convolution2DCommon *common, Backend *b, const float *originWeight,
                                               size_t originWeightSize, const float *bias, size_t biasSize)
    : CPUConvolution(common, b) {
    _init(common, b, originWeight, originWeightSize, bias, biasSize);
}

Convolution1x1Strassen::~Convolution1x1Strassen() {
    if (nullptr != mWeight) {
        backend()->onReleaseBuffer(mWeight.get(), Backend::STATIC);
    }
    backend()->onReleaseBuffer(mBias.get(), Backend::STATIC);
}

ErrorCode Convolution1x1Strassen::onReleaseCache() {
    return NO_ERROR;
}

ErrorCode Convolution1x1Strassen::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    CPUConvolution::onResize(inputs, outputs);
    auto input       = inputs[0];
    auto output      = outputs[0];
    auto ic        = input->channel();
    auto oc        = output->channel();
    auto l = ic;
    auto h = oc;
    int ePack, lPack, hPack;
    MNNGetMatMulPackMode(&ePack, &lPack, &hPack);
    auto threadNumber = static_cast<CPUBackend*>(backend())->threadNumber();
    mTempInputPack.reset(Tensor::createDevice<float>({threadNumber, UP_DIV(l, lPack), ePack * lPack}));
    mTempOutputPack.reset(Tensor::createDevice<float>({threadNumber, UP_DIV(h, hPack), ePack * hPack}));
    bool res = true;
    res = res && backend()->onAcquireBuffer(mTempInputPack.get(), Backend::DYNAMIC);
    res = res && backend()->onAcquireBuffer(mTempOutputPack.get(), Backend::DYNAMIC);

    if (!res) {
        return OUT_OF_MEMORY;
    }
    mParameters.resize(6);
    mParameters[0] = 1;
    mParameters[1] = UP_DIV(l, lPack);
    mParameters[2] = UP_DIV(h, hPack);
    mParameters[5] = 0;
    res = res && backend()->onReleaseBuffer(mTempInputPack.get(), Backend::DYNAMIC);
    res = res && backend()->onReleaseBuffer(mTempOutputPack.get(), Backend::DYNAMIC);

    return NO_ERROR;
}

ErrorCode Convolution1x1Strassen::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];
    auto ic        = input->channel();
    auto oc        = output->channel();
    auto outputPlane = output->height() * output->width();
    auto e = outputPlane;
    auto l = ic;
    auto h = oc;
    auto ocC4 = UP_DIV(oc, 4);
    int ePack, lPack, hPack;
    MNNGetMatMulPackMode(&ePack, &lPack, &hPack);
    auto tileCount = UP_DIV(e, ePack);
    auto threadNumber = static_cast<CPUBackend*>(backend())->threadNumber();

    for (int batchIndex = 0; batchIndex < input->batch(); ++batchIndex) {
        MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
            auto gemmSrc = mTempInputPack->host<float>() + tId * mTempInputPack->stride(0);
            auto gemmDst = mTempOutputPack->host<float>() + tId * mTempOutputPack->stride(0);
            for (int index=tId; index < tileCount; index += threadNumber) {
                auto inputSrc = input->host<float>() + batchIndex * input->stride(0) + index * 4 * ePack;
                auto eSize = std::min(e - index * ePack, ePack);
                MNNPackC4ForMatMul_A(gemmSrc, inputSrc, eSize, l, e);
                MNNPackedMatMul(gemmDst, gemmSrc, mWeight->host<float>(), mParameters.data());
                auto outputSrc = output->host<float>() + batchIndex * output->stride(0) + index * 4 * ePack;
                MNNUnPackC4ForMatMul_C(outputSrc, gemmDst, eSize, h, e);
            }
        }
        MNN_CONCURRENCY_END();
        
        MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
            for (int dz=tId; dz<ocC4; dz+=threadNumber) {
                mPostFunction(output->host<float>() + batchIndex * output->stride(0) + dz * outputPlane * 4, mBias->host<float>() + dz * 4, outputPlane, 1);
            }
        }
        MNN_CONCURRENCY_END();
    }
    return NO_ERROR;
}
} // namespace MNN
