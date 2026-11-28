//
//  MNNSpacemitIme2AttentionFunctions.cpp
//  MNN
//
//  Created by MNN on 2026/07/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNNSpacemitIme2Attention.hpp"

#include <atomic>

#include "backend/cpu/CPUKVCacheManager.hpp"
#include "backend/cpu/CPUBackend.hpp"

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

extern "C" int MNNSpacemitIme2FlashAttentionFp32C4(float* dst, const float* query, const float* key, const float* value,
                                                   int seqLen, int numHeads, int headDim, float scale, void* qkvScratch,
                                                   size_t qkvScratchBytes, float* scoreScratch, size_t scoreCount,
                                                   float* outputScratch, size_t outputCount);
extern "C" int MNNSpacemitIme2FlashAttentionFp32C4Pair(float* dst, const float* query, const float* key,
                                                       const float* value, int seqLen, int numHeads, int headDim,
                                                       float scale, void* scratch, size_t scratchBytes);
extern "C" int MNNSpacemitIme2FlashAttentionFp32C4Supported();

#include "MNNSpacemitIme2Half.hpp"

using MNNRvvAttentionTask = void (*)(size_t taskIndex, void* taskScratch, size_t taskScratchBytes, void* context);
extern "C" size_t MNNSpacemitIme2RunTcmTasks(size_t taskCount, MNNRvvAttentionTask task, void* context);

#include <vector>

namespace MNN {

struct MNNSpacemitIme2FusedContext {
    CPUKVCacheManager* kvCache = nullptr;
    const int8_t* query = nullptr;
    int8_t* output = nullptr;
    int seqLen = 0;
    int taskCount = 8;
    float attentionScale = 0.0f;
    std::atomic<int> success{1};
};

extern "C" size_t MNNSpacemitIme2TcmBlockSize();

// Scratch the pair kernel needs for a given sequence length (64-row query tiles, 64-key KV tiles,
// two query heads per task, K/V packed as fp16 for every KV block up front).
static size_t MNNSpacemitIme2PairScratchBytes(int seqLen) {
    constexpr int queryTile = 64;
    constexpr int kvTile = 64;
    constexpr int pairCount = 2;
    constexpr int headDim = 128;
    const size_t halfTile = (size_t) queryTile * headDim;
    const size_t kvBlockNums = ((size_t) seqLen + kvTile - 1) / kvTile;
    return ((2 * kvBlockNums + pairCount) * halfTile) * sizeof(_Float16) +
           (queryTile * kvTile + pairCount * queryTile * headDim) * sizeof(float);
}

static void MNNSpacemitIme2RunFusedAttentionTask(size_t taskIndex, void* taskScratch, size_t taskScratchBytes,
                                                 void* context) {
    constexpr int kNumHeads = 16;
    constexpr int kHeadDim = 128;
    constexpr int kBytes = 4;
    constexpr size_t kFusedScratchBytes = 96 * 1024;
    constexpr size_t kQkvScratchBytes = 48 * 1024;
    constexpr size_t kScoreScratchOffset = 48 * 1024;
    constexpr size_t kOutputScratchOffset = 64 * 1024;
    auto fusedContext = static_cast<MNNSpacemitIme2FusedContext*>(context);
    // One task owns every head pair with index >= taskIndex that is congruent to it modulo taskCount,
    // so the same kernel serves any worker count (a pair is two query heads sharing one KV head).
    static thread_local std::vector<int8_t> heapScratch;
    const int pairCount = kNumHeads / 2;
    for (int pair = static_cast<int>(taskIndex); pair < pairCount; pair += fusedContext->taskCount) {
        const int headIndex = pair * 2;
        const int kvHeadIndex = headIndex / 2;
        auto dst = reinterpret_cast<float*>(fusedContext->output +
                                            static_cast<size_t>(headIndex) * kHeadDim * fusedContext->seqLen * kBytes);
        auto query =
            reinterpret_cast<const float*>(fusedContext->query + static_cast<size_t>(headIndex) * kHeadDim * kBytes);
        auto key = reinterpret_cast<const float*>(fusedContext->kvCache->addrOfKey(kvHeadIndex));
        auto value = reinterpret_cast<const float*>(fusedContext->kvCache->addrOfValue(kvHeadIndex));
        // The pack scratch holds K/V for every KV block of the sequence, so long sequences exceed the
        // per-task TCM block. Fall back to a per-worker heap buffer instead of the generic attention path.
        void* scratchPtr = taskScratch;
        size_t scratchSize = taskScratchBytes;
        const size_t needed = MNNSpacemitIme2PairScratchBytes(fusedContext->seqLen);
        if (needed > scratchSize) {
            if (heapScratch.size() < needed) {
                heapScratch.resize(needed);
            }
            scratchPtr = heapScratch.data();
            scratchSize = heapScratch.size();
        }
        int status = MNNSpacemitIme2FlashAttentionFp32C4Pair(dst, query, key, value, fusedContext->seqLen, kNumHeads,
                                                            kHeadDim, fusedContext->attentionScale, scratchPtr,
                                                            scratchSize);
        if (status > 0) {
            continue;
        }

        if (taskScratch == nullptr || taskScratchBytes < kFusedScratchBytes) {
            fusedContext->success.store(0, std::memory_order_relaxed);
            return;
        }
        auto scratch = static_cast<int8_t*>(taskScratch);
        for (int localHead = 0; localHead < 2; ++localHead) {
            const int head = headIndex + localHead;
            auto singleDst =
                reinterpret_cast<float*>(fusedContext->output + static_cast<size_t>(head) * kHeadDim * fusedContext->seqLen * kBytes);
            auto singleQuery =
                reinterpret_cast<const float*>(fusedContext->query + static_cast<size_t>(head) * kHeadDim * kBytes);
            status = MNNSpacemitIme2FlashAttentionFp32C4(
                singleDst, singleQuery, key, value, fusedContext->seqLen, kNumHeads, kHeadDim,
                fusedContext->attentionScale, taskScratch, kQkvScratchBytes,
                reinterpret_cast<float*>(scratch + kScoreScratchOffset), 4096,
                reinterpret_cast<float*>(scratch + kOutputScratchOffset), 8192);
            if (status <= 0) {
                fusedContext->success.store(0, std::memory_order_relaxed);
                return;
            }
        }
    }
}

MNNSpacemitIme2Attention::MNNSpacemitIme2Attention(Backend* backend, bool kvCache)
    : MNNRvvAttention(backend, kvCache) {}

bool MNNSpacemitIme2Attention::tryExecuteFastPath(const int8_t* query, int8_t* output, int seqLen, int kvSeqLen,
                                                  int paddingLength, float qScale, float attentionScale,
                                                  bool lowerTriangular, bool hasSinks, bool outputC4,
                                                  bool directC4Output) {
#if defined(__riscv_zvfh)
    if (MNNIme2::halfPrefillEnabled() && static_cast<CPUBackend*>(backend())->precisionMode() == BackendConfig::Precision_Low && mUseFlashAttention && mKeyQuantMode == KVQuantMode::None && mValueQuantMode == KVQuantMode::None &&
        mBytes == 4 && mPack == 4 && hP == 4 && lP == 1 && mHeadDim == 256 && mQNumHead >= mKvNumHead &&
        mKvNumHead > 0 && mQNumHead % mKvNumHead == 0 && directC4Output && lowerTriangular && !hasSinks &&
        paddingLength == 0 && qScale == 1.0f && seqLen == kvSeqLen && seqLen >= 16 &&
        mKvBlockSize == 64 && mKVCacheManager->getFlashAttentionBlockKv() == 64 && mThreadNum > 0 &&
        mThreadNum <= 8 && query && output) {
        const float* localHeads[16];
        std::unique_ptr<const float*[]> extra;
        if (mKvNumHead > 8) {
            extra.reset(new (std::nothrow) const float*[2 * size_t(mKvNumHead)]);
        }
        auto keys = mKvNumHead <= 8 ? localHeads : extra.get();
        if (keys) {
            auto values = keys + mKvNumHead;
            for (int h = 0; h < mKvNumHead; ++h) {
                keys[h] = reinterpret_cast<const float*>(mKVCacheManager->addrOfKey(h));
                values[h] = reinterpret_cast<const float*>(mKVCacheManager->addrOfValue(h));
            }
            if (MNNIme2CausalAttention(reinterpret_cast<const float*>(query), keys, values,
                                       reinterpret_cast<float*>(output), seqLen, mHeadDim, mQNumHead, mKvNumHead,
                                       mThreadNum, attentionScale))
                return true;
        }
    }
#endif
    if (MNNSpacemitIme2FlashAttentionFp32C4Supported() && mUseFlashAttention && mKeyQuantMode == KVQuantMode::None &&
        mValueQuantMode == KVQuantMode::None && mBytes == 4 && mPack == 4 && hP == 4 && lP == 1 && mQNumHead == 16 &&
        mKvNumHead == 8 && mThreadNum >= 2 && mThreadNum <= 8 && directC4Output && seqLen == kvSeqLen && seqLen >= 64 &&
        seqLen <= 4096 && mKvBlockSize == 64 && mHeadDim == 128 && paddingLength == 0 &&
        lowerTriangular && !hasSinks &&
        qScale == 1.0f && query != nullptr && output != nullptr) {
        MNNSpacemitIme2FusedContext context;
        context.kvCache = mKVCacheManager.get();
        context.query = query;
        context.output = output;
        context.seqLen = seqLen;
        context.attentionScale = attentionScale;
        context.taskCount = mThreadNum;
        const size_t completed = MNNSpacemitIme2RunTcmTasks(mThreadNum, MNNSpacemitIme2RunFusedAttentionTask, &context);
        if (completed == static_cast<size_t>(mThreadNum) && context.success.load(std::memory_order_relaxed) != 0) {
            return true;
        }
    }
    return MNNRvvAttention::tryExecuteFastPath(query, output, seqLen, kvSeqLen, paddingLength, qScale, attentionScale,
                                               lowerTriangular, hasSinks, outputC4, directC4Output);
}

CPUAttention* MNNSpacemitIme2Attention::createClone(Backend* backend) const {
    return new MNNSpacemitIme2Attention(backend, mKVCache);
}

Execution* MNNSpacemitIme2CreateAttentionExecution(Backend* backend, bool kvCache) {
    return new MNNSpacemitIme2Attention(backend, kvCache);
}

} // namespace MNN

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
