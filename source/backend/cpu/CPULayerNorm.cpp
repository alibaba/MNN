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
#include "CPUCast.hpp"
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
        // Use uint8_t to avoid lowp reduce float bytes
        res->mGamma.reset(Tensor::createDevice<uint8_t>({size * 4}));
        auto status = backend->onAcquireBuffer(res->mGamma.get(), Backend::STATIC);
        if (!status) {
            MNN_ERROR("Out of memory when gamma is acquired in CPULayerNorm.\n");
            return nullptr;
        }
        // Use uint8_t to avoid lowp reduce float bytes
        res->mBeta.reset(Tensor::createDevice<uint8_t>({size * 4}));
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
    auto input = inputs[0]->host<uint8_t>();
    auto output = outputs[0]->host<uint8_t>();
    auto bn = static_cast<CPUBackend*>(backend());
    auto core = bn->functions();
    auto threadNumber = bn->threadNumber();
    threadNumber = ALIMIN(threadNumber, mOutterSize);
    auto int8core = bn->int8Functions();
    int bytes = core->bytes;
    auto inputQuan = TensorUtils::getDescribe(inputs[0])->quantAttr.get();
    auto outputQuan = TensorUtils::getDescribe(outputs[0])->quantAttr.get();

    if (CPUBackend::getDataType(inputs[0]) == DataType_DT_INT8 || inputs[0]->getType().bytes() == 1) {
        bytes = 1;
    }

    MNN_CONCURRENCY_BEGIN(ttId, threadNumber) {
        for (int tId=ttId; tId < mOutterSize; tId += threadNumber) {
            const float* inner_input = (const float*)(input + tId * mInnerSize * bytes);
            float* inner_output = (float*)(output + tId * mInnerSize * bytes);
            if (bytes != 4) {
                auto tmpInput = (float*)(mTmpInputFloat.ptr() + ttId * mInnerSize * sizeof(float));
                auto tmpOutput = (float*)(mTmpOutputFloat.ptr() + ttId * mInnerSize * sizeof(float));
                if (bytes == 1) {
                    CPUCastCreator::cast(inner_input, tmpInput, CPUCastCreator::INT8_TO_FlOAT, mInnerSize, inputQuan->scale, inputQuan->zero, inputQuan->min, inputQuan->max, bn);
                } else {
                    core->MNNLowpToFp32((const int16_t*)inner_input, tmpInput, mInnerSize);
                }
                MNNNorm(tmpOutput, tmpInput, gamma, beta, mResource->mEpsilon, mInnerSize, mResource->mRMSNorm);
                if (bytes == 1) {
                    CPUCastCreator::cast(tmpOutput, inner_output, CPUCastCreator::FlOAT_TO_INT8, mInnerSize, outputQuan->scale, outputQuan->zero, outputQuan->min, outputQuan->max, bn);
                } else {
                    core->MNNFp32ToLowp(tmpOutput, (int16_t*)inner_output, mInnerSize);
                }
            } else {
                MNNNorm(inner_output, inner_input, gamma, beta, mResource->mEpsilon, mInnerSize, mResource->mRMSNorm);
            }
        }
    }
    MNN_CONCURRENCY_END();
    return NO_ERROR;
}

ErrorCode CPULayerNorm::onResize(const std::vector<Tensor*> &inputs,
                                 const std::vector<Tensor*> &outputs) {
    mOutterSize = 1;
    mInnerSize = 1;
    do {
        // Compute outter and inner
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
            break;
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
    } while (false);
    auto bn = static_cast<CPUBackend*>(backend());
    auto threadNumber = ALIMIN(bn->threadNumber(), mOutterSize);
    auto buf = bn->getBufferAllocator();

    if (CPUBackend::getDataType(inputs[0]) == DataType_DT_INT8 || inputs[0]->getType().bytes() == 1 || bn->functions()->bytes != 4) {
        mTmpInputFloat = buf->alloc(threadNumber * mInnerSize * sizeof(float));
        mTmpOutputFloat = buf->alloc(threadNumber * mInnerSize * sizeof(float));
        buf->free(mTmpInputFloat);
        buf->free(mTmpOutputFloat);
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
