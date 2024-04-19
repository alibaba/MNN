//
//  CPUScale.cpp
//  MNN
//
//  Created by MNN on 2023/05/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include "math.h"
#include "CPUScaleInt8.hpp"
#include "CPUBackend.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "core/Concurrency.h"
#include "compute/CommonOptFunction.h"
#include "backend/cpu/compute/Int8FunctionsOpt.h"

namespace MNN {

CPUScaleInt8::CPUScaleInt8(const Op* op, Backend* bn) : MNN::Execution(bn) {
    auto scale      = op->main_as_Scale();
    auto core = static_cast<CPUBackend*>(bn)->functions();
    int outputCount = scale->scaleData()->size();
    mScaleBias.reset(Tensor::createDevice<uint8_t>({2, UP_DIV(outputCount, core->pack) * core->pack * core->bytes}));
    auto res = bn->onAcquireBuffer(mScaleBias.get(), Backend::STATIC);
    if (!res) {
        MNN_ERROR("Error for alloc buffer for CPUScale\n");
        mScaleBias = nullptr;
        mValid = false;
        return;
    }
    ::memset(mScaleBias->host<float>(), 0, mScaleBias->size());
    std::vector<float> scaleDataQuant(outputCount);
    for (int i = 0; i < outputCount; ++i) {
        scaleDataQuant[i] = 1.0 / scale->scaleData()->data()[i];
    }
    if (core->bytes < 4) {
        core->MNNFp32ToLowp(scale->scaleData()->data(), mScaleBias->host<int16_t>(), outputCount);
    } else {
        ::memcpy(mScaleBias->host<float>(), scale->scaleData()->data(), outputCount * sizeof(float));
    }
    if (nullptr != scale->biasData() && nullptr != scale->biasData()->data()) {
        auto biasPtr = mScaleBias->host<uint8_t>() + mScaleBias->length(1);
        if (core->bytes < 4) {
            core->MNNFp32ToLowp(scale->biasData()->data(), reinterpret_cast<int16_t*>(biasPtr), outputCount);
        } else {
            ::memcpy(biasPtr, scale->biasData()->data(), outputCount * sizeof(float));
        }
    }
}
CPUScaleInt8::~CPUScaleInt8() {
    if (nullptr != mScaleBias) {
        backend()->onReleaseBuffer(mScaleBias.get(), Backend::STATIC);
    }
}

ErrorCode CPUScaleInt8::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input = inputs[0];
    auto output = outputs[0];
    auto core = static_cast<CPUBackend*>(backend())->functions();
    int outputCount = output->channel();

    mInputQuantInfo = TensorUtils::getQuantInfo(input);
    mOutputQuantInfo = TensorUtils::getQuantInfo(output);
    float inputScale = mInputQuantInfo[0], outputScale = mOutputQuantInfo[0];
    outputScale = (outputScale == 0.f ? 0.f : 1.f / outputScale);

    std::vector<int32_t> scales_(outputCount, 0);
    std::vector<int32_t> bias_(outputCount, 0);
    auto scalePtr = (float*)mScaleBias->host<uint8_t>();
    auto biasPtr  = (float*)(mScaleBias->host<uint8_t>() + mScaleBias->length(1));

    mShiftBits = 15;
    for (int i = 0; i < outputCount; ++i) {
        int32_t scaleInt32 = static_cast<int32_t>(roundf(scalePtr[i] * inputScale * outputScale * (1 << mShiftBits)));
        scales_[i] = scaleInt32;
        int32_t biasInt32  = static_cast<int32_t>(roundf(biasPtr[i] * outputScale* (1 << mShiftBits)));
        bias_[i]  = biasInt32;
    }

    auto scalePtr_ = mScaleBias->host<uint8_t>();
    auto biasPtr_  = scalePtr_ + mScaleBias->length(1);
    ::memcpy(scalePtr_, scales_.data(), outputCount * sizeof(int32_t));
    ::memcpy(biasPtr_, bias_.data(), outputCount * sizeof(int32_t));

    mOutputQuantInfo[0] = outputScale;

    return NO_ERROR;
}


ErrorCode CPUScaleInt8::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];
    auto core = static_cast<CPUBackend*>(backend())->functions();
    auto gcore = static_cast<CPUBackend*>(backend())->int8Functions();
    auto scalePtr = mScaleBias->host<uint8_t>();
    auto biasPtr = mScaleBias->host<uint8_t>() + 1 * mScaleBias->length(1);

    auto batch       = input->buffer().dim[0].extent;
    auto depthQuad   = UP_DIV(input->channel(), core->pack);
    int planeNumber = 1;
    for (int i = 2; i < input->buffer().dimensions; ++i) {
        planeNumber *= input->length(i);
    }
    auto depthStride = planeNumber * core->pack;
    auto totalDepth = batch * depthQuad;
    int numberThread = ((CPUBackend*)backend())->threadNumber();

    MNN_CONCURRENCY_BEGIN(tId, numberThread) {
        int8_t inputZeroPoint = (int8_t)mInputQuantInfo[1];
        int8_t outputZeroPoint = (int8_t)mOutputQuantInfo[1];
        for (int i = tId; i < totalDepth; i+=numberThread) {
            auto depthIndex = i / batch;
            const int8_t* inputPtr      = input->host<int8_t>() + depthStride * i;
            const int32_t* biasPtr_      = (const int32_t*)(biasPtr + core->pack * core->bytes * depthIndex);
            const int32_t* scalePtr_     = (const int32_t*)(scalePtr + core->pack * core->bytes * depthIndex);
            MNNScaleAndAddBiasInt8(output->host<int8_t>() + depthStride * i, inputPtr, biasPtr_, scalePtr_, mShiftBits, (ssize_t)mOutputQuantInfo[2], (ssize_t)mOutputQuantInfo[3], &inputZeroPoint, &outputZeroPoint, planeNumber, 1, core->pack);
        }
    }
    MNN_CONCURRENCY_END();
    return NO_ERROR;
}

} // namespace MNN
