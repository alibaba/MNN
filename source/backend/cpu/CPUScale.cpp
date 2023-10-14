//
//  CPUScale.cpp
//  MNN
//
//  Created by MNN on 2018/08/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUScale.hpp"
#include "CPUScaleInt8.hpp"
#include "CPUBackend.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "core/Concurrency.h"
#include "core/OpCommonUtils.hpp"
#include "compute/CommonOptFunction.h"

namespace MNN {
CPUScale::CPUScale(const Op* op, Backend* bn) : MNN::Execution(bn) {
    auto scale      = op->main_as_Scale();
    auto core = static_cast<CPUBackend*>(bn)->functions();
    bool external = USE_EXTERNAL_DATA(scale);
    int outputCount = 0;
    if (external) {
        outputCount = static_cast<int>(scale->external()->Get(1) / sizeof(float));
    } else {
        outputCount = scale->scaleData()->size();
    }
    mScaleBias.reset(Tensor::createDevice<uint8_t>({2, UP_DIV(outputCount, core->pack) * core->pack * core->bytes}));
    auto res = bn->onAcquireBuffer(mScaleBias.get(), Backend::STATIC);
    if (!res) {
        MNN_ERROR("Error for alloc buffer for CPUScale\n");
        mScaleBias = nullptr;
        mValid = false;
        return;
    }
    ::memset(mScaleBias->host<float>(), 0, mScaleBias->size());
    if (external) {
        bool hasBias = scale->external()->size() > 2;
        if (hasBias) {
            if (core->bytes < 4) {
                std::unique_ptr<Tensor> tmpTensor(Tensor::createDevice<float>({outputCount * 2}));
                auto status = backend()->onAcquireBuffer(tmpTensor.get(), Backend::STATIC);
                if (!status) {
                    MNN_ERROR("Out of memory when tmpTensor is acquired in CPUScale.\n");
                    return;
                }
                char* scalePtr = tmpTensor->host<char>();
                char* biasPtr = scalePtr + outputCount * sizeof(float);
                OpCommonUtils::loadExternalDatas(bn, {scalePtr, biasPtr}, scale->external()->data());
                core->MNNFp32ToLowp(tmpTensor->host<float>(), mScaleBias->host<int16_t>(), outputCount * 2);
            } else {
                OpCommonUtils::loadExternalDatas(bn, {mScaleBias->host<char>(), mScaleBias->host<char>() + mScaleBias->length(1)}, scale->external()->data());
            }
        } else {
            if (core->bytes < 4) {
                std::unique_ptr<Tensor> tmpTensor(Tensor::createDevice<float>({outputCount}));
                auto status = backend()->onAcquireBuffer(tmpTensor.get(), Backend::STATIC);
                if (!status) {
                    MNN_ERROR("Out of memory when tmpTensor is acquired in CPUScale.\n");
                    return;
                }
                OpCommonUtils::loadExternalDatas(bn, {tmpTensor->host<char>()}, scale->external()->data());
                core->MNNFp32ToLowp(tmpTensor->host<float>(), mScaleBias->host<int16_t>(), outputCount);
            } else {
                OpCommonUtils::loadExternalDatas(bn, {mScaleBias->host<char>()}, scale->external()->data());
            }
        }
    } else {
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
}
CPUScale::~CPUScale() {
    if (nullptr != mScaleBias) {
        backend()->onReleaseBuffer(mScaleBias.get(), Backend::STATIC);
    }
}
ErrorCode CPUScale::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];
    auto core = static_cast<CPUBackend*>(backend())->functions();
    auto scalePtr = mScaleBias->host<uint8_t>();
    auto biasPtr = mScaleBias->host<uint8_t>() + 1 * mScaleBias->length(1);
    //FUNC_PRINT(TensorUtils::getDescribe(input)->dimensionFormat);
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
        for (int i = tId; i < totalDepth; i+=numberThread) {
            auto depthIndex = i / batch;
            core->MNNScaleAndAddBias((float*)(output->host<uint8_t>() + depthStride * i * core->bytes), (const float*)(input->host<uint8_t>() + depthStride * i * core->bytes), (const float*)(biasPtr + core->pack * core->bytes * depthIndex),
                                     (const float*)(scalePtr + core->pack * core->bytes * depthIndex), planeNumber, 1);
        }
    }
    MNN_CONCURRENCY_END();
    return NO_ERROR;
}
class CPUScaleCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        if (CPUBackend::getDataType(inputs[0]) == DataType_DT_INT8 || inputs[0]->getType().bytes() == 1) {
            return new CPUScaleInt8(op, backend);
        }
        return new CPUScale(op, backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPUScaleCreator, OpType_Scale);
} // namespace MNN
