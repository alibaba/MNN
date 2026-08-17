//
//  MetalLinearAttention.hpp
//  MNN
//
//  Created by MNN on 2026/02/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalLinearAttention_hpp
#define MetalLinearAttention_hpp

#import "core/Macro.h"
#import "MetalExecution.hpp"
#import "MetalBackend.hpp"
#include "MNN_generated.h"
#include "core/KVMeta.hpp"

#if MNN_METAL_ENABLED
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

namespace MNN {

struct MetalStateCache {
    std::shared_ptr<Tensor> mConvState;      // Conv1D padding state: [B, D, kernel_size - 1]
    std::shared_ptr<Tensor> mRecurrentState; // Gated Delta Rule recurrent state S: [B, H, d_k, d_v]
    // Pending verify block: the states above always hold the committed prefix only.
    std::shared_ptr<Tensor> mPendingQKVRaw;  // [B, D, cap] raw qkv (pre-conv) for conv-state commit
    std::shared_ptr<Tensor> mPendingK;       // [B, cap, H, d_k] prepped k
    std::shared_ptr<Tensor> mPendingV;       // [B, cap, H, d_v] prepped v
    std::shared_ptr<Tensor> mPendingGate;    // [B, cap, H] gate (unpacked)
    std::shared_ptr<Tensor> mPendingBeta;    // [B, cap, H] beta (unpacked)
    // Ping-pong set B, so in-kernel writes of the new block never alias reads of the previous one.
    std::shared_ptr<Tensor> mPendingK2;
    std::shared_ptr<Tensor> mPendingV2;
    std::shared_ptr<Tensor> mPendingGate2;
    std::shared_ptr<Tensor> mPendingBeta2;
    int mPendingIdx = 0;                     // which set holds the CURRENT pending block
    int mPendingLen = 0;                     // tokens in pending block (0 = none)
    int mPendingCap = 0;                     // allocated capacity (0 = not allocated)
};

class MetalLinearAttention : public MetalExecution {
public:
    MetalLinearAttention(Backend* backend, const MNN::Op* op);
    virtual ~MetalLinearAttention() = default;
    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual void onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                          id<MTLComputeCommandEncoder> encoder) override;
    // Prefill encode performs per-token CPU state management (state reset
    // memset) that binding replay cannot capture, so only the decode path
    // (seqLen == 1: no CPU-side state logic, constant param buffer) is
    // recordable. mLastSeqLen is set in onResize, which always precedes the
    // encode whose shape it describes. A live pending block is excluded too: it may emit a flush.
    virtual bool canRecordEncode() const override {
        return mLastSeqLen == 1 && mAttentionType == "gated_delta_rule" &&
               mStateCache->mPendingLen == 0;
    }
    // onResize re-creates mConvOut (and may re-plan the allocator), leaving
    // dangling Tensor* in recorded bindings. Bail out of replay whenever a
    // resize happened after the recording; the normal encode re-records.
    virtual bool onReplayUpdate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        return mRecordedGeneration == mResizeGeneration;
    }
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;

private:
    std::string mAttentionType;
    int mNumKHeads;
    int mNumVHeads;
    int mHeadKDim;
    int mHeadVDim;
    bool mUseQKL2Norm;

    // Persistent state buffers shared between prefill and decode via onClone
    std::shared_ptr<MetalStateCache> mStateCache;

    KVMeta* mMeta = nullptr;

    // Replay guards (see canRecordEncode/onReplayUpdate above)
    int mLastSeqLen = 0;
    int mResizeGeneration = 0;
    int mRecordedGeneration = -1;

    // gate/beta elementwise-chain fold (see MetalBackend::matchLinearAttnGateFolds).
    // Results persist across the per-token forced re-resize; invalidated when
    // the gate/beta input tensors change.
    MetalBackend::LinearAttnFoldRequest mFoldReq;
    // Export-time gate/beta fold: when true, mFoldReq is populated from
    // LinearAttentionParam directly (no runtime chain matching needed).
    bool mGateFold = false;
    std::vector<float> mGateCoef;
    std::vector<float> mGateBias;

    // Temporary buffer (DYNAMIC)
    std::shared_ptr<Tensor> mConvOut; // [B, D, L]
    std::shared_ptr<Tensor> mQ;       // [B, L, H, d_k]
    std::shared_ptr<Tensor> mK;       // [B, L, H, d_k]
    std::shared_ptr<Tensor> mV;       // [B, L, H, d_v]
    // Param buffer for shader
    id<MTLBuffer> mParamBuffer;
    // Flushes a pending block outside the verify path: seq_len = 0 plus the commit fields.
    id<MTLBuffer> mParamBufferFlush;

    // Pipeline states
    id<MTLComputePipelineState> mConvSiluPipeline;
    id<MTLComputePipelineState> mConvSiluStateDecodePipeline = nil;
    id<MTLComputePipelineState> mConvStateUpdatePipeline;
    id<MTLComputePipelineState> mQKVPrepPipeline;
    id<MTLComputePipelineState> mQKVPrepSGPipeline        = nil;
    id<MTLComputePipelineState> mGatedDeltaRulePipeline;
    id<MTLComputePipelineState> mGatedDeltaRuleSGPipeline;
    id<MTLComputePipelineState> mGatedDeltaRuleSGV4Pipeline = nil;
    id<MTLComputePipelineState> mGatedDeltaRuleFusedSGPipeline;
    id<MTLComputePipelineState> mFusedSGAlignPipeline     = nil;
    int mFusedSGAlignSimds = 4;

    id<MTLComputePipelineState> mFusedSGTGPipeline        = nil;
    int mFusedSGTGSimds = 4;
    id<MTLComputePipelineState> mFusedChunkSGPipeline     = nil;
    id<MTLComputePipelineState> mFlashChunkPrepPipeline   = nil;
    id<MTLComputePipelineState> mFlashChunkScanPipeline   = nil;
    id<MTLComputePipelineState> mFlashChunkSGMMPipeline   = nil;
    // Speculative-verify rollback pipelines
    id<MTLComputePipelineState> mConvCommitPipeline     = nil; // commit conv state from pending raw qkv
    id<MTLComputePipelineState> mQKVRawSavePipeline     = nil; // save raw qkv (unpacked) to pending
    // prep + delta + pending_save in one dispatch; seq_len = 0 runs the commit prologue alone
    id<MTLComputePipelineState> mVerifyFusedSGPipeline  = nil;
    id<MTLComputePipelineState> mShortConvPipeline;
    id<MTLComputePipelineState> mShortConvStateUpdatePipeline;
    id<MTLComputePipelineState> mShortConvOutputPipeline;
    bool mUseSimdGroupOpt   = false;
    bool mUseFusedChunkSG   = false;
    bool mUseFlashChunk     = false;
    bool mUseFlashChunkSGMM = false;
    int  mChunkTGThreads    = 0;
    int  mFlashDvBlock      = 32;
    int  mFlashSimdsPerTG   = 4;
    int  mSgmmDvBlock       = 16;
    int  mSgmmSimdsPerTG    = 16;
};

} // namespace MNN
#endif /* MNN_SUPPORT_TRANSFORMER_FUSE */
#endif /* MNN_METAL_ENABLED */
#endif /* MetalLinearAttention_hpp */
