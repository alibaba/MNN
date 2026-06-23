//
//  CPURoPE.cpp
//  MNN
//
//  Created by MNN on 2018/08/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPURoPE.hpp"
#include "CPUBackend.hpp"
#include "MNN_generated.h"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/Concurrency.h"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {
CPURoPE::CPURoPE(const Op* op, Backend* bn) : MNN::Execution(bn) {
    const Op* qLayernorm = nullptr;
    const Op* kLayernorm = nullptr;
    if (nullptr != op && OpParameter_Extra == op->main_type()) {
        auto extra = op->main_as_Extra();
        if (nullptr != extra && nullptr != extra->attr()) {
            for (int i = 0; i < extra->attr()->size(); ++i) {
                auto attr = extra->attr()->GetAs<Attribute>(i);
                if (nullptr == attr || nullptr == attr->key()) {
                    continue;
                }
                if (attr->key()->str() == "rope_cut_head_dim") {
                    mRopeCutHeadDim = attr->i();
                    continue;
                }
                if (attr->key()->str() == "q_norm") {
                    qLayernorm = flatbuffers::GetRoot<Op>(attr->tensor()->int8s()->data());
                    mQNorm = CPULayerNorm::makeResource(qLayernorm, bn);
                    continue;
                }
                if (attr->key()->str() == "k_norm") {
                    kLayernorm = flatbuffers::GetRoot<Op>(attr->tensor()->int8s()->data());
                    mKNorm = CPULayerNorm::makeResource(kLayernorm, bn);
                    continue;
                }
            }
        }
    }
}

CPURoPE::~CPURoPE() {
    // Do nothing.
}

CPURoPE::CPURoPE(Backend* bn) : Execution(bn) {
    // Do nothing.
}

ErrorCode CPURoPE::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto bn = static_cast<CPUBackend*>(backend());
    auto threadNumber = bn->threadNumber();
    auto buf = bn->getBufferAllocator();
    if (bn->functions()->bytes != 4) {
        if (mQNorm) {
            auto Q = inputs[0];
            int numHead = Q->length(2);
            int headDim = Q->length(3);
            mTmpQFloat = buf->alloc(threadNumber * numHead * headDim * sizeof(float));
            buf->free(mTmpQFloat);
        }
        if (mKNorm) {
            auto K = inputs[1];
            int kvnumHead = K->length(2);
            int headDim = K->length(3);
            mTmpKFloat = buf->alloc(threadNumber * kvnumHead * headDim * sizeof(float));
            buf->free(mTmpKFloat);
        }
    }
    return NO_ERROR;
}

bool CPURoPE::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (nullptr == dst) {
        return true;
    }
    auto rope = new CPURoPE(bn);
    rope->mRopeCutHeadDim = mRopeCutHeadDim;
    rope->mQNorm = mQNorm;
    rope->mKNorm = mKNorm;
    *dst = rope;
    return true;
}

ErrorCode CPURoPE::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto Q = inputs[0];
    auto K = inputs[1];
    auto cosEven = inputs[2];
    auto cosOdd = inputs[3];
    auto sinEven = inputs[4];
    auto sinOdd = inputs[5];

    auto QOutput = outputs[0];
    auto KOutput = outputs[1];
    int batch = Q->length(0);
    int seqLen = Q->length(1);
    int numHead = Q->length(2);
    int headDim = Q->length(3);
    int kvnumHead = K->length(2);
    auto halfHeadDim = headDim / 2;
    int threadNum = static_cast<CPUBackend*>(backend())->threadNumber();
    int totalWork = batch * seqLen;
    auto core = static_cast<CPUBackend*>(backend())->functions();
    MNN_ASSERT(core->MNNRoPECompute != nullptr);

    MNN_CONCURRENCY_BEGIN(tId, threadNum) {
        int start = tId * totalWork / threadNum;
        int end = (tId + 1) * totalWork / threadNum;
        for (int i = start; i < end; ++i) {
            auto cosEvenPtr = static_cast<const uint8_t*>(cosEven->host<void>()) + i * halfHeadDim * core->bytes;
            auto cosOddPtr = static_cast<const uint8_t*>(cosOdd->host<void>()) + i * halfHeadDim * core->bytes;
            auto sinEvenPtr = static_cast<const uint8_t*>(sinEven->host<void>()) + i * halfHeadDim * core->bytes;
            auto sinOddPtr = static_cast<const uint8_t*>(sinOdd->host<void>()) + i * halfHeadDim * core->bytes;
            auto qPtr = static_cast<const uint8_t*>(Q->host<void>()) + i * numHead * headDim * core->bytes;
            auto qPtrOut = static_cast<uint8_t*>(QOutput->host<void>()) + i * numHead * headDim * core->bytes;

            if (mQNorm) {
                int size = headDim;
                const float* gamma = mQNorm->mIniGammaBeta ? mQNorm->mGamma->host<float>() : nullptr;
                const float* beta = mQNorm->mIniGammaBeta ? mQNorm->mBeta->host<float>() : nullptr;
                if (core->bytes == 4) {
                    for (int h = 0; h < numHead; ++h) {
                        MNNNorm(reinterpret_cast<float*>(qPtrOut) + h * headDim,
                                reinterpret_cast<const float*>(qPtr) + h * headDim, gamma, beta, mQNorm->mEpsilon, size,
                                mQNorm->mRMSNorm);
                    }
                    qPtr = qPtrOut;
                } else {
                    int totalSize = numHead * headDim;
                    auto tmpQ = reinterpret_cast<float*>(mTmpQFloat.ptr() + tId * totalSize * sizeof(float));
                    core->MNNLowpToFp32(reinterpret_cast<const int16_t*>(qPtr), tmpQ, totalSize);
                    for (int h = 0; h < numHead; ++h) {
                        MNNNorm(tmpQ + h * headDim, tmpQ + h * headDim, gamma, beta, mQNorm->mEpsilon, size,
                                mQNorm->mRMSNorm);
                    }
                    core->MNNFp32ToLowp(tmpQ, reinterpret_cast<int16_t*>(qPtrOut), totalSize);
                    qPtr = qPtrOut;
                }
            }
            core->MNNRoPECompute(qPtrOut, qPtr, cosEvenPtr, cosOddPtr, sinEvenPtr, sinOddPtr, numHead, headDim,
                                 mRopeCutHeadDim);

            qPtr = static_cast<const uint8_t*>(K->host<void>()) + i * kvnumHead * headDim * core->bytes;
            qPtrOut = static_cast<uint8_t*>(KOutput->host<void>()) + i * kvnumHead * headDim * core->bytes;

            if (mKNorm) {
                int size = headDim;
                const float* gamma = mKNorm->mIniGammaBeta ? mKNorm->mGamma->host<float>() : nullptr;
                const float* beta = mKNorm->mIniGammaBeta ? mKNorm->mBeta->host<float>() : nullptr;
                if (core->bytes == 4) {
                    for (int h = 0; h < kvnumHead; ++h) {
                        MNNNorm(reinterpret_cast<float*>(qPtrOut) + h * headDim,
                                reinterpret_cast<const float*>(qPtr) + h * headDim, gamma, beta, mKNorm->mEpsilon, size,
                                mKNorm->mRMSNorm);
                    }
                    qPtr = qPtrOut;
                } else {
                    int totalSize = kvnumHead * headDim;
                    auto tmpK = reinterpret_cast<float*>(mTmpKFloat.ptr() + tId * totalSize * sizeof(float));
                    core->MNNLowpToFp32(reinterpret_cast<const int16_t*>(qPtr), tmpK, totalSize);
                    for (int h = 0; h < kvnumHead; ++h) {
                        MNNNorm(tmpK + h * headDim, tmpK + h * headDim, gamma, beta, mKNorm->mEpsilon, size,
                                mKNorm->mRMSNorm);
                    }
                    core->MNNFp32ToLowp(tmpK, reinterpret_cast<int16_t*>(qPtrOut), totalSize);
                    qPtr = qPtrOut;
                }
            }
            core->MNNRoPECompute(qPtrOut, qPtr, cosEvenPtr, cosOddPtr, sinEvenPtr, sinOddPtr, kvnumHead, headDim,
                                 mRopeCutHeadDim);
        }
    }
    MNN_CONCURRENCY_END();
    return NO_ERROR;
}

class CPURoPECreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new CPURoPE(op, backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPURoPECreator, OpType_RoPE);
} // namespace MNN
