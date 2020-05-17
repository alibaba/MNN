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
    mStracssenComputor.reset(new StrassenMatrixComputor(b, true, 5));
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
    auto outputPlane = output->height() * output->width();
    auto e = outputPlane;
    auto l = ic;
    auto h = oc;
    mTempOutputBatch.reset();
    mTempInputBatch.reset();
    int ePack, lPack, hPack;
    MNNGetMatMulPackMode(&ePack, &lPack, &hPack);
    mTempInputPack.reset(Tensor::createDevice<float>({UP_DIV(e, ePack), UP_DIV(l, lPack), ePack * lPack}));
    mTempOutputPack.reset(Tensor::createDevice<float>({UP_DIV(e, ePack), UP_DIV(h, hPack), ePack * hPack}));
    mTempInputBatch.reset(Tensor::createDevice<float>({l, e}));
    mTempOutputBatch.reset(Tensor::createDevice<float>({h, e}));
    
    bool res = backend()->onAcquireBuffer(mTempInputBatch.get(), Backend::DYNAMIC);
    res = res && backend()->onAcquireBuffer(mTempInputPack.get(), Backend::DYNAMIC);
    res = res && backend()->onAcquireBuffer(mTempOutputPack.get(), Backend::DYNAMIC);
    res = res && backend()->onAcquireBuffer(mTempOutputBatch.get(), Backend::DYNAMIC);

    if (!res) {
        return OUT_OF_MEMORY;
    }
    mStracssenComputor->onReset();
    auto code = mStracssenComputor->onEncode({mTempInputPack.get(), mWeight.get()}, {mTempOutputPack.get()});
    if (NO_ERROR != code) {
        return code;
    }
    res = backend()->onReleaseBuffer(mTempInputBatch.get(), Backend::DYNAMIC);
    res = res && backend()->onReleaseBuffer(mTempInputPack.get(), Backend::DYNAMIC);
    res = res && backend()->onReleaseBuffer(mTempOutputPack.get(), Backend::DYNAMIC);
    res = res && backend()->onReleaseBuffer(mTempOutputBatch.get(), Backend::DYNAMIC);

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

    for (int batchIndex = 0; batchIndex < input->batch(); ++batchIndex) {
        MNNPackTranspose(mTempInputBatch->host<float>(), input->host<float>() + batchIndex * input->stride(0), e, l);
        MNNPackForMatMul_A(mTempInputPack->host<float>(), mTempInputBatch->host<float>(), e, l, false);
        mStracssenComputor->onExecute();
        MNNUnpackForMatMul_C(mTempOutputBatch->host<float>(), mTempOutputPack->host<float>(), e, h);
        MNNUnpackTranspose(output->host<float>() + batchIndex * output->stride(0), mTempOutputBatch->host<float>(), outputPlane, oc);
        mPostFunction(output->host<float>() + batchIndex * output->stride(0), mBias->host<float>(), outputPlane, ocC4);
    }
    return NO_ERROR;
}
} // namespace MNN
