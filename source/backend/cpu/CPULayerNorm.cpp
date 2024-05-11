//
//  CPULayerNorm.cpp
//  MNN
//
//  Created by MNN on 2020/07/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <cmath>
#include "backend/cpu/CPULayerNorm.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/Execution.hpp"
#include "core/Concurrency.h"
#include "core/TensorUtils.hpp"
#include "MNN_generated.h"

namespace MNN {

CPULayerNorm::CPULayerNorm(std::shared_ptr<Resource> res, Backend* backend) : Execution(backend) {
    mResource = res;
}

std::shared_ptr<CPULayerNorm::Resource> CPULayerNorm::makeResource(const MNN::Op* op, Backend* backend) {
    const auto* layer_norm_param = op->main_as_LayerNorm();
    std::shared_ptr<CPULayerNorm::Resource> res(new Resource);
    res->mAxis = 0;
    if (nullptr != layer_norm_param->axis()) {
        res->mAxis = layer_norm_param->axis()->size();
    }
    res->mGroup = layer_norm_param->group();
    res->mEpsilon = layer_norm_param->epsilon();
    res->mRMSNorm = layer_norm_param->useRMSNorm();
    if (layer_norm_param->gamma() && layer_norm_param->beta()) {
        int size = layer_norm_param->gamma()->size();
        res->mIniGammaBeta = true;
        res->mGamma.reset(Tensor::createDevice<float>({size}));
        auto status = backend->onAcquireBuffer(res->mGamma.get(), Backend::STATIC);
        if (!status) {
            MNN_ERROR("Out of memory when gamma is acquired in CPULayerNorm.\n");
            return nullptr;
        }
        res->mBeta.reset(Tensor::createDevice<float>({size}));
        status = backend->onAcquireBuffer(res->mBeta.get(), Backend::STATIC);
        if (!status) {
            MNN_ERROR("Out of memory when beta is acquired in CPULayerNorm.\n");
            return nullptr;
        }

        if (layer_norm_param->beta()->size() != size) {
            MNN_ERROR("Size of gamma and beta are not match in CPULayerNorm.\n");
        }
        const float* gamma_data = layer_norm_param->gamma()->data();
        memcpy(res->mGamma->host<float>(), gamma_data, size * sizeof(float));
        const float* beta_data = layer_norm_param->beta()->data();
        memcpy(res->mBeta->host<float>(), beta_data, size * sizeof(float));
    }
    return res;
}

ErrorCode CPULayerNorm::onExecute(const std::vector<Tensor*> &inputs,
                                  const std::vector<Tensor*> &outputs) {
    const float* gamma = mResource->mIniGammaBeta ? mResource->mGamma->host<float>() : nullptr;
    const float* beta = mResource->mIniGammaBeta ? mResource->mBeta->host<float>() : nullptr;
    
    if (mInpZero.data()) {
        const int8_t* input = inputs[0]->host<int8_t>();
        int8_t* output = outputs[0]->host<int8_t>();
        MNN_CONCURRENCY_BEGIN(tId, mOutterSize) {
            auto core = static_cast<CPUBackend*>(backend())->int8Functions();
            QuanPrePostParameters params;
            params.maxValue = mMaxMinValue[0];
            params.minValue = mMaxMinValue[1];
            params.inputScale = mInpScale.data();
            params.outputScale = mOutScale.data();
            params.inputZeroPoint = mInpZero.data();
            params.outputZeroPoint = mOutZero.data();
            const int8_t* inner_input = input + tId * mInnerSize;
            int8_t* inner_output = output + tId * mInnerSize;
            core->MNNNormInt8(inner_output, inner_input, gamma, beta, mResource->mEpsilon, mInnerSize, &params, mResource->mRMSNorm);
        }
        MNN_CONCURRENCY_END();
        return NO_ERROR;
    }

    const float* input = inputs.at(0)->host<float>();
    float* output = outputs.at(0)->host<float>();
    MNN_CONCURRENCY_BEGIN(tId, mOutterSize) {
        const float* inner_input = input + tId * mInnerSize;
        float* inner_output = output + tId * mInnerSize;
        MNNNorm(inner_output, inner_input, gamma, beta, mResource->mEpsilon, mInnerSize, mResource->mRMSNorm);
    }
    MNN_CONCURRENCY_END();
    return NO_ERROR;
}

ErrorCode CPULayerNorm::onResize(const std::vector<Tensor*> &inputs,
                                 const std::vector<Tensor*> &outputs) {
    mOutterSize = 1;
    mInnerSize = 1;
    int rank = inputs.at(0)->dimensions();
    if (mResource->mGroup > 1) {
        mOutterSize = inputs.at(0)->length(0) * mResource->mGroup;
        for (int i = 1; i < rank; i++) {
            mInnerSize *= inputs.at(0)->length(i);
        }
        mInnerSize /= mResource->mGroup;
        if (mResource->mIniGammaBeta) {
            MNN_ASSERT(mResource->mGamma->size() == mInnerSize * sizeof(float));
        }
    
        return NO_ERROR;
    }
    for (int i = 0; i < rank - mResource->mAxis; ++i) {
        mOutterSize *= inputs.at(0)->length(i);
    }
    for (int i = rank - mResource->mAxis; i < rank; ++i) {
        mInnerSize *= inputs.at(0)->length(i);
    }
    if (mResource->mIniGammaBeta) {
        MNN_ASSERT(mResource->mGamma->size() == mInnerSize * sizeof(float));
    }
    if (CPUBackend::getDataType(inputs[0]) == DataType_DT_INT8 || inputs[0]->getType().bytes() == 1) {
        mInpZero.resize(1);
        mOutZero.resize(1);
        mInpScale.resize(1);
        mOutScale.resize(1);
        mMaxMinValue.resize(2);
        auto inpQuantAttr = TensorUtils::getDescribe(inputs[0])->quantAttr;
        auto outQuantAttr = TensorUtils::getDescribe(outputs[0])->quantAttr;
        mInpZero[0] = inpQuantAttr->zero;
        mOutZero[0] = outQuantAttr->zero;
        mInpScale[0] = inpQuantAttr->scale;
        mOutScale[0] = outQuantAttr->scale == 0.f? 0.f : 1.0f / outQuantAttr->scale;
        mMaxMinValue[0] = outQuantAttr->max;
        mMaxMinValue[1] = outQuantAttr->min;
    }
    return NO_ERROR;
}

CPULayerNorm::~CPULayerNorm() {
    // Do nothing
}
bool CPULayerNorm::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (nullptr == dst) {
        return true;
    }
    *dst = new CPULayerNorm(mResource, bn);
    return true;
}

class CPULayerNormCreator : public CPUBackend::Creator {
public:
    Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op, Backend* backend) const override {
        auto res = CPULayerNorm::makeResource(op, backend);
        if (nullptr == res.get()) {
            return nullptr;
        }
        return new CPULayerNorm(res, backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPULayerNormCreator, OpType_LayerNorm);

}  // namespace MNN
