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

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

extern "C" int MNNSpacemitIme2FlashAttentionFp32C4(float* dst, const float* query, const float* key, const float* value,
                                                   int seqLen, int numHeads, int headDim, float scale, void* qkvScratch,
                                                   size_t qkvScratchBytes, float* scoreScratch, size_t scoreCount,
                                                   float* outputScratch, size_t outputCount);
extern "C" int MNNSpacemitIme2FlashAttentionFp32C4Pair(float* dst, const float* query, const float* key,
                                                       const float* value, int seqLen, int numHeads, int headDim,
                                                       float scale, void* scratch, size_t scratchBytes);
extern "C" int MNNSpacemitIme2FlashAttentionFp32C4Supported();

using MNNRvvAttentionTask = void (*)(size_t taskIndex, void* taskScratch, size_t taskScratchBytes, void* context);
extern "C" size_t MNNSpacemitIme2RunTcmTasks(size_t taskCount, MNNRvvAttentionTask task, void* context);

namespace MNN {

struct MNNSpacemitIme2FusedContext {
    CPUKVCacheManager* kvCache = nullptr;
    const int8_t* query = nullptr;
    int8_t* output = nullptr;
    int seqLen = 0;
    float attentionScale = 0.0f;
    std::atomic<int> success{1};
};

static void MNNSpacemitIme2RunFusedAttentionTask(size_t taskIndex, void* taskScratch, size_t taskScratchBytes,
                                                 void* context) {
    constexpr int kNumHeads = 16;
    constexpr int kHeadDim = 128;
    constexpr int kBytes = 4;
    auto fusedContext = static_cast<MNNSpacemitIme2FusedContext*>(context);
    const int headIndex = static_cast<int>(taskIndex) * 2;
    const int kvHeadIndex = headIndex / 2;
    auto dst = reinterpret_cast<float*>(fusedContext->output +
                                        static_cast<size_t>(headIndex) * kHeadDim * fusedContext->seqLen * kBytes);
    auto query =
        reinterpret_cast<const float*>(fusedContext->query + static_cast<size_t>(headIndex) * kHeadDim * kBytes);
    auto key = reinterpret_cast<const float*>(fusedContext->kvCache->addrOfKey(kvHeadIndex));
    auto value = reinterpret_cast<const float*>(fusedContext->kvCache->addrOfValue(kvHeadIndex));
    int status =
        MNNSpacemitIme2FlashAttentionFp32C4Pair(dst, query, key, value, fusedContext->seqLen, kNumHeads, kHeadDim,
                                                fusedContext->attentionScale, taskScratch, taskScratchBytes);
    if (status > 0) {
        return;
    }

    constexpr size_t kFusedScratchBytes = 96 * 1024;
    constexpr size_t kQkvScratchBytes = 48 * 1024;
    constexpr size_t kScoreScratchOffset = 48 * 1024;
    constexpr size_t kOutputScratchOffset = 64 * 1024;
    if (taskScratch == nullptr || taskScratchBytes < kFusedScratchBytes) {
        fusedContext->success.store(0, std::memory_order_relaxed);
        return;
    }
    auto scratch = static_cast<int8_t*>(taskScratch);
    for (int localHead = 0; localHead < 2; ++localHead) {
        const int head = headIndex + localHead;
        auto singleDst = reinterpret_cast<float*>(fusedContext->output +
                                                  static_cast<size_t>(head) * kHeadDim * fusedContext->seqLen * kBytes);
        auto singleQuery =
            reinterpret_cast<const float*>(fusedContext->query + static_cast<size_t>(head) * kHeadDim * kBytes);
        status = MNNSpacemitIme2FlashAttentionFp32C4(
            singleDst, singleQuery, key, value, fusedContext->seqLen, kNumHeads, kHeadDim, fusedContext->attentionScale,
            taskScratch, kQkvScratchBytes, reinterpret_cast<float*>(scratch + kScoreScratchOffset), 4096,
            reinterpret_cast<float*>(scratch + kOutputScratchOffset), 8192);
        if (status <= 0) {
            fusedContext->success.store(0, std::memory_order_relaxed);
            return;
        }
    }
}

MNNSpacemitIme2Attention::MNNSpacemitIme2Attention(Backend* backend, bool kvCache)
    : MNNRvvAttention(backend, kvCache) {}

bool MNNSpacemitIme2Attention::tryExecuteFastPath(const int8_t* query, int8_t* output, int seqLen, int kvSeqLen,
                                                  int paddingLength, float qScale, float attentionScale,
                                                  bool lowerTriangular, bool hasSinks, bool outputC4,
                                                  bool directC4Output) {
    if (MNNSpacemitIme2FlashAttentionFp32C4Supported() && mUseFlashAttention && mKeyQuantMode == KVQuantMode::None &&
        mValueQuantMode == KVQuantMode::None && mBytes == 4 && mPack == 4 && hP == 4 && lP == 1 && mQNumHead == 16 &&
        mKvNumHead == 8 && mThreadNum == 8 && directC4Output && seqLen == kvSeqLen && seqLen >= 64 && seqLen <= 512 &&
        seqLen % 64 == 0 && mKvBlockSize == 64 && mHeadDim == 128 && paddingLength == 0 && lowerTriangular && !hasSinks &&
        qScale == 1.0f && query != nullptr && output != nullptr) {
        MNNSpacemitIme2FusedContext context;
        context.kvCache = mKVCacheManager.get();
        context.query = query;
        context.output = output;
        context.seqLen = seqLen;
        context.attentionScale = attentionScale;
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
