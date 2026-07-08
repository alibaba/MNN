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
#include <cstring>

namespace MNN {
static std::shared_ptr<CPULayerNorm::Resource> makeRopeNormResource(const LayerNorm* layerNorm, Backend* backend) {
    if (nullptr == layerNorm || nullptr == layerNorm->gamma()) {
        return nullptr;
    }
    int gammaSize = layerNorm->gamma()->size();
    if (gammaSize <= 0) {
        return nullptr;
    }
    auto res = std::make_shared<CPULayerNorm::Resource>();
    res->mGroup = layerNorm->group();
    res->mEpsilon = layerNorm->epsilon();
    res->mRMSNorm = layerNorm->useRMSNorm();
    res->mAxis = layerNorm->axis() == nullptr ? 1 : layerNorm->axis()->size();
    res->mIniGammaBeta = true;
    res->mGamma.reset(Tensor::createDevice<uint8_t>({gammaSize * (int)sizeof(float)}));
    res->mBeta.reset(Tensor::createDevice<uint8_t>({gammaSize * (int)sizeof(float)}));
    auto status = backend->onAcquireBuffer(res->mGamma.get(), Backend::STATIC) &&
                  backend->onAcquireBuffer(res->mBeta.get(), Backend::STATIC);
    if (!status) {
        MNN_ERROR("CPURoPE: alloc q/k norm gamma buffer error.\n");
        return nullptr;
    }
    ::memcpy(res->mGamma->host<float>(), layerNorm->gamma()->data(), gammaSize * sizeof(float));
    if (layerNorm->beta() != nullptr && layerNorm->beta()->size() == gammaSize) {
        ::memcpy(res->mBeta->host<float>(), layerNorm->beta()->data(), gammaSize * sizeof(float));
    } else {
        ::memset(res->mBeta->host<float>(), 0, gammaSize * sizeof(float));
    }
    return res;
}

static void unpackC4Token(const uint8_t* src, uint8_t* dst, int token, int seqLen, int channel, int bytes, int pack,
                          int channelOffset = 0) {
    for (int c = 0; c < channel; ++c) {
        int srcChannel = channelOffset + c;
        int c4 = srcChannel / pack;
        int ci = srcChannel % pack;
        ::memcpy(dst + c * bytes, src + (c4 * seqLen * pack + token * pack + ci) * bytes, bytes);
    }
}

static bool validRopeC4Input(const Tensor* q, const Tensor* k, int numHead, int kvNumHead, int headDim) {
    if (q == nullptr || k == nullptr || numHead <= 0 || kvNumHead <= 0 || headDim <= 0) {
        return false;
    }
    if (TensorUtils::getDescribe(q)->dimensionFormat != MNN_DATA_FORMAT_NC4HW4 ||
        TensorUtils::getDescribe(k)->dimensionFormat != MNN_DATA_FORMAT_NC4HW4) {
        return false;
    }
    if (q->dimensions() < 2 || k->dimensions() < 2) {
        return false;
    }
    return q->length(1) == numHead * headDim && k->length(1) == kvNumHead * headDim;
}

CPURoPE::CPURoPE(const Op* op, Backend* bn) : MNN::Execution(bn) {
    auto param = op == nullptr ? nullptr : op->main_as_RoPEParam();
    if (param != nullptr) {
        mRopeCutHeadDim = param->rope_cut_head_dim();
        mNumHead = param->num_head();
        mKvNumHead = param->kv_num_head();
        mHeadDim = param->head_dim();
        mQNorm = makeRopeNormResource(param->q_norm(), bn);
        mKNorm = makeRopeNormResource(param->k_norm(), bn);
    }
}

CPURoPE::~CPURoPE() {
    // Do nothing.
}

CPURoPE::CPURoPE(Backend* bn) : Execution(bn) {
    // Do nothing.
}

ErrorCode CPURoPE::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto Q = inputs[0];
    auto K = inputs[1];
    if (!validRopeC4Input(Q, K, mNumHead, mKvNumHead, mHeadDim)) {
        MNN_ERROR("CPURoPE: invalid C4 head config, numHead=%d, kvNumHead=%d, headDim=%d.\n", mNumHead, mKvNumHead,
                  mHeadDim);
        return NOT_SUPPORT;
    }
    auto bn = static_cast<CPUBackend*>(backend());
    auto threadNumber = bn->threadNumber();
    auto buf = bn->getBufferAllocator();
    auto bytes = bn->functions()->bytes;
    mTmpQC4 = buf->alloc(threadNumber * mNumHead * mHeadDim * bytes);
    buf->free(mTmpQC4);
    mTmpKC4 = buf->alloc(threadNumber * mKvNumHead * mHeadDim * bytes);
    buf->free(mTmpKC4);
    if (bytes != 4) {
        if (mQNorm) {
            mTmpQFloat = buf->alloc(threadNumber * mNumHead * mHeadDim * sizeof(float));
            buf->free(mTmpQFloat);
        }
        if (mKNorm) {
            mTmpKFloat = buf->alloc(threadNumber * mKvNumHead * mHeadDim * sizeof(float));
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
    rope->mNumHead = mNumHead;
    rope->mKvNumHead = mKvNumHead;
    rope->mHeadDim = mHeadDim;
    rope->mQNorm = mQNorm;
    rope->mKNorm = mKNorm;
    *dst = rope;
    return true;
}

ErrorCode CPURoPE::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto Q = inputs[0];
    auto K = inputs[1];
    if (!validRopeC4Input(Q, K, mNumHead, mKvNumHead, mHeadDim)) {
        MNN_ERROR("CPURoPE: invalid C4 input, numHead=%d, kvNumHead=%d, headDim=%d.\n", mNumHead, mKvNumHead, mHeadDim);
        return NOT_SUPPORT;
    }
    auto cos = inputs[2];
    auto sin = inputs[3];

    auto QOutput = outputs[0];
    auto KOutput = outputs[1];
    int batch = 1;
    int seqLen = Q->length(0);
    int numHead = mNumHead;
    int headDim = mHeadDim;
    int kvnumHead = mKvNumHead;
    auto halfHeadDim = headDim / 2;
    int threadNum = static_cast<CPUBackend*>(backend())->threadNumber();
    int totalWork = batch * seqLen;
    auto core = static_cast<CPUBackend*>(backend())->functions();
    MNN_ASSERT(core->MNNRoPECompute != nullptr);

    MNN_CONCURRENCY_BEGIN(tId, threadNum) {
        int start = tId * totalWork / threadNum;
        int end = (tId + 1) * totalWork / threadNum;
        for (int i = start; i < end; ++i) {
            auto cosPtr = static_cast<const uint8_t*>(cos->host<void>()) + i * headDim * core->bytes;
            auto sinPtr = static_cast<const uint8_t*>(sin->host<void>()) + i * headDim * core->bytes;
            auto cosEvenPtr = cosPtr;
            auto cosOddPtr = cosPtr + halfHeadDim * core->bytes;
            auto sinEvenPtr = sinPtr;
            auto sinOddPtr = sinPtr + halfHeadDim * core->bytes;
            auto qPtr = static_cast<const uint8_t*>(Q->host<void>());
            auto qPtrOut = static_cast<uint8_t*>(QOutput->host<void>()) + i * numHead * headDim * core->bytes;
            auto qTmp = static_cast<uint8_t*>(mTmpQC4.ptr()) + tId * numHead * headDim * core->bytes;
            unpackC4Token(qPtr, qTmp, i, seqLen, numHead * headDim, core->bytes, core->pack);
            qPtr = qTmp;

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

            qPtr = static_cast<const uint8_t*>(K->host<void>());
            qPtrOut = static_cast<uint8_t*>(KOutput->host<void>()) + i * kvnumHead * headDim * core->bytes;
            auto kTmp = static_cast<uint8_t*>(mTmpKC4.ptr()) + tId * kvnumHead * headDim * core->bytes;
            unpackC4Token(qPtr, kTmp, i, seqLen, kvnumHead * headDim, core->bytes, core->pack);
            qPtr = kTmp;

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
