//
//  MNNRvvAttentionFunctions.cpp
//  MNN
//
//  Created by MNN on 2026/07/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNNRvvAttention.hpp"

#include <utility>

#include "MNNRvvFastPathUtils.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/CPUKVCacheManager.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

void MNNPackedMatMulRemainFP32_RVV_WithAStride(float* C, const float* A, const float* B, size_t eSize,
                                               const size_t* parameter, const float* postParameters, const float* bias,
                                               const float* k, const float* b, size_t aStride);

namespace MNN {

MNNRvvAttention::MNNRvvAttention(Backend* backend, bool kvCache) : CPUAttention(backend, kvCache) {}

MNNRvvAttention::~MNNRvvAttention() {
    releaseDecodeScratch();
}

ErrorCode MNNRvvAttention::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    releaseDecodeScratch();
    return CPUAttention::onResize(inputs, outputs);
}

void MNNRvvAttention::releaseDecodeScratch() {
    if (mDecodeScratch) {
        backend()->onReleaseBuffer(mDecodeScratch.get(), Backend::STATIC);
        mDecodeScratch.reset();
        mDecodeScratchCapacity = 0;
    }
}

bool MNNRvvAttention::acquireDecodeScratch(int capacity) {
    if (mDecodeScratch && capacity <= mDecodeScratchCapacity) {
        return true;
    }
    auto scratch =
        std::shared_ptr<Tensor>(Tensor::createDevice<int8_t>({mThreadNum, UP_DIV(capacity, mPack), 1, mPack * mBytes}));
    if (!backend()->onAcquireBuffer(scratch.get(), Backend::STATIC)) {
        return false;
    }
    releaseDecodeScratch();
    mDecodeScratch = std::move(scratch);
    mDecodeScratchCapacity = capacity;
    return true;
}

bool MNNRvvAttention::tryExecuteFastPath(const int8_t* query, int8_t* output, int seqLen, int kvSeqLen,
                                         int paddingLength, float qScale, float attentionScale, bool lowerTriangular,
                                         bool hasSinks, bool outputC4, bool directC4Output) {
    if (seqLen != 1 || mUseFlashAttention || mKeyQuantMode != KVQuantMode::None ||
        mValueQuantMode != KVQuantMode::None || mBytes != 4 || mPack != 4 || hP != 4 || lP != 1 || qScale != 1.0f ||
        !lowerTriangular || hasSinks || kvSeqLen != mKvBlockSize || mThreadNum <= 0 || mQNumHead <= 0 || mKvNumHead <= 0 ||
        mQNumHead % mKvNumHead != 0 || query == nullptr || output == nullptr) {
        return false;
    }

    const size_t requiredScoreBytes = static_cast<size_t>(ROUND_UP(kvSeqLen, mPack)) * mBytes;
    int8_t* scoreBase = mPackQ ? mPackQ->host<int8_t>() : nullptr;
    size_t scoreStride = mPackQ ? mPackQ->stride(0) : 0;
    std::shared_ptr<Tensor> transientScratch;
    if (scoreBase == nullptr || scoreStride < requiredScoreBytes) {
        constexpr int kMaxPersistentDecodeScratch = 4096;
        const int requestedCapacity = ALIMAX(mKvBlockSize, mKVCacheManager->maxLength());
        if (mKvBlockSize <= kMaxPersistentDecodeScratch) {
            const int persistentCapacity = ALIMIN(requestedCapacity, kMaxPersistentDecodeScratch);
            if (!acquireDecodeScratch(persistentCapacity)) {
                return false;
            }
            scoreBase = mDecodeScratch->host<int8_t>();
            scoreStride = mDecodeScratch->stride(0);
        } else {
            releaseDecodeScratch();
            transientScratch.reset(
                Tensor::createDevice<int8_t>({mThreadNum, UP_DIV(requestedCapacity, mPack), 1, mPack * mBytes}));
            if (!backend()->onAcquireBuffer(transientScratch.get(), Backend::STATIC)) {
                return false;
            }
            scoreBase = transientScratch->host<int8_t>();
            scoreStride = transientScratch->stride(0);
        }
    }

    const size_t requiredOutputBytes = static_cast<size_t>(ROUND_UP(mHeadDim, mPack) * mBytes);
    int8_t* outputWorkspace = mPackQKV ? mPackQKV->host<int8_t>() : nullptr;
    size_t outputWorkspaceStride = mPackQKV ? mPackQKV->stride(0) : 0;
    if (!directC4Output && (outputWorkspace == nullptr || outputWorkspaceStride < requiredOutputBytes)) {
        if (transientScratch) {
            backend()->onReleaseBuffer(transientScratch.get(), Backend::STATIC);
        }
        return false;
    }

    auto core = static_cast<CPUBackend*>(backend())->functions();
    const int headsPerThread = UP_DIV(mQNumHead, mThreadNum);
    const int groupSize = mQNumHead / mKvNumHead;
    MNNRvvFastPathParallelFor(backend(), mThreadNum, [&](int tId) {
        const int headBegin = tId * headsPerThread;
        const int headEnd = ALIMIN(mQNumHead, headBegin + headsPerThread);
        auto score = reinterpret_cast<float*>(scoreBase + static_cast<size_t>(tId) * scoreStride);
        auto outputScratch =
            directC4Output ? nullptr : outputWorkspace + static_cast<size_t>(tId) * outputWorkspaceStride;
        for (int head = headBegin; head < headEnd; ++head) {
            const int kvHead = head / groupSize;
            auto key = reinterpret_cast<const float*>(mKVCacheManager->addrOfKey(kvHead));
            auto value = reinterpret_cast<const float*>(mKVCacheManager->addrOfValue(kvHead));
            auto headQuery = reinterpret_cast<const float*>(query + static_cast<size_t>(head) * mHeadDim * mBytes);

            size_t qkParameters[7] = {
                static_cast<size_t>(eP * lP * mBytes),
                static_cast<size_t>(ROUND_UP(mHeadDim, lP)),
                static_cast<size_t>(kvSeqLen),
                static_cast<size_t>(mPack * mBytes),
                0,
                0,
                0,
            };
            MNNPackedMatMulRemainFP32_RVV_WithAStride(score, headQuery, key, 1, qkParameters, nullptr, nullptr, nullptr,
                                                      nullptr, 1);
            for (int i = 0; i < kvSeqLen; ++i) {
                score[i] *= attentionScale;
            }
            core->MNNSoftmax(score, score, nullptr, nullptr, nullptr, 1, kvSeqLen, 0, kvSeqLen - 1, mPack, true);

            int8_t* packedOutput =
                directC4Output ? output + static_cast<size_t>(head) * mHeadDim * mBytes : outputScratch;
            size_t pvParameters[7] = {
                static_cast<size_t>(eP * lP * mBytes),
                static_cast<size_t>(ROUND_UP(kvSeqLen, lP)),
                static_cast<size_t>(mHeadDim),
                static_cast<size_t>(mPack * mBytes),
                0,
                0,
                0,
            };
            pvParameters[5] =
                (ROUND_UP(mKVCacheManager->getFlashAttentionBlockKv(), lP) - ROUND_UP(kvSeqLen, lP)) * hP * mBytes;
            MNNPackedMatMulRemainFP32_RVV_WithAStride(reinterpret_cast<float*>(packedOutput), score, value, 1,
                                                      pvParameters, nullptr, nullptr, nullptr, nullptr, 1);

            if (!outputC4) {
                int offset[2] = {1, mQNumHead * mHeadDim};
                auto dst = output + static_cast<size_t>(head) * mHeadDim * mBytes;
                core->MNNUnpackCUnitTranspose(reinterpret_cast<float*>(dst),
                                              reinterpret_cast<const float*>(packedOutput), 1, mHeadDim, offset);
            } else if (!directC4Output) {
                for (int d = 0; d < mHeadDim; ++d) {
                    const int channel = head * mHeadDim + d;
                    const size_t srcOffset = (static_cast<size_t>(d / mPack) * mPack + d % mPack) * mBytes;
                    const size_t dstOffset = (static_cast<size_t>(channel / mPack) * mPack + channel % mPack) * mBytes;
                    ::memcpy(output + dstOffset, packedOutput + srcOffset, mBytes);
                }
            }
        }
    });

    if (transientScratch) {
        backend()->onReleaseBuffer(transientScratch.get(), Backend::STATIC);
    }
    return true;
}

CPUAttention* MNNRvvAttention::createClone(Backend* backend) const {
    return new MNNRvvAttention(backend, mKVCache);
}

Execution* MNNRvvCreateAttentionExecution(Backend* backend, bool kvCache) {
    return new MNNRvvAttention(backend, kvCache);
}

} // namespace MNN

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
