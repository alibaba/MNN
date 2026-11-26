//
//  MetalAttention.mm
//  MNN
//
//  Created by MNN on b'2024/04/29'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalAttention_hpp
#define MetalAttention_hpp

#import "core/Macro.h"
#import "MetalBackend.hpp"
#include "MNN_generated.h"
#include "core/OpCommonUtils.hpp"
#include "MetalKVCacheManager.hpp"

#if MNN_METAL_ENABLED
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

namespace MNN {

// All three are passed with setBytes rather than bound from a persistent MTLBuffer --
// see _makeCopyParam in the .mm for the in-flight-overwrite hazard that forces this.
struct AttentionCopyParam {
    int head_count;
    int kv_seq_len;
    int max_kv_len;
    int dst_k_offset;
    int dst_v_offset;
    int batch;
    int value_c4;
    float v_scale;
    float k_scale;
};

struct AttentionQKVParam {
    int query_seq_len;
    int q_seq_piece_len;
    int key_seq_len;
    int head_num;
    int group;
    int head_dim;
    float scale;
    int max_kv_len;
    int batch;
    int kv_align_len;
    int mask_batch;
    int mask_head_num;
    int mask_q_len;
    int mask_k_len;
    float v_scale;
    float k_scale;
};

// Matches softmax_shape: inside, axis, outside, axis_align, q_piece_len, causal_base, head_count.
struct AttentionSoftmaxParam {
    int v[7];
};

class AttentionBufExecution : public MetalExecution {
public:
    AttentionBufExecution(Backend* backend, bool kvCache, bool outputC4,
                          std::shared_ptr<KVQuantParameter> kvQuantParam);
    ~AttentionBufExecution() override;
    void invalidateReplay() override;
    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;

    virtual void onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                          id<MTLComputeCommandEncoder> encoder) override;
    virtual bool onReplayUpdate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override {
        if (nullptr == dst) {
            return true;
        }
        auto param = op->main_as_AttentionParam();
        const bool kvShared = nullptr != param && param->kv_shared_layer_index() >= 0;
        const bool canShare = mKVCache && bn->getMetaPtr() == mMeta && mMeta != nullptr;
        if (kvShared && !canShare) {
            // A private cache would hold the placeholder K/V instead of the owner's.
            return false;
        }
        auto exe = new AttentionBufExecution(bn, mKVCache, mOutputC4, mKVQuantParameter);
        if (canShare) {
            exe->mKVCacheManager->removeReplayUser(exe);
            exe->mKVCacheManager = mKVCacheManager;
            exe->mKVCacheManager->addReplayUser(exe);
            exe->mIsKVShared = kvShared;
        }
        *dst = exe;
        MNN_METAL_PROFILE_REGISTER_CLONE(bn, op, *dst);
        return true;
    }

private:
    enum class AttnRoute : uint8_t {
        DecodeSdpa = 0,        // decode_splitkv
        DecodeFusedQkSoftmax,  // decode_qk_softmax + decode_qkv
        DecodeStaged,          // decode_qk -> softmax -> decode_qkv
        PrefillFaTc,           // prefill_flash_attn_tc
        PrefillFaSg,           // prefill_flash_attn_sg
        PrefillStaged,         // prefill_qk -> softmax -> prefill_qkv
    };
    // Compile outcomes only (build failure or threadgroup limit), never routing inputs.
    enum AttnVeto : uint8_t {
        AttnVetoNone      = 0,
        AttnVetoTensorApi = 1 << 0,
        AttnVetoFaTc      = 1 << 1,
        AttnVetoFaSg      = 1 << 2,
        AttnVetoSdpa      = 1 << 3,
    };
    // Re-snapshotted per resolve: kvLen changes per token.
    struct AttnFacts {
        int seqLen = 0, kvLen = 0, kvMaxLen = 0;
        bool decode = false;
        bool simdReduce = false, simdMatrix = false, tensorApi = false, fp16 = false;
        bool highBandwidth = false;
        int gpuGen = 0;
        bool kvCache = true, kvInDisk = false;
        bool causal = false, bidirectionalMask = false;
        bool sinks = false;
        bool trivialDecodeMask = false;
        bool quantKey = false, quantValue = false, dynamicScaleBuffer = false;
        int batch = 1, numHead = 1, kvNumHead = 1, headDim = 1, groupSize = 1;
        bool groupDivides = false;
    };
    struct AttnPlan {
        AttnRoute route = AttnRoute::DecodeStaged;
        bool isDecode() const {
            return route == AttnRoute::DecodeSdpa || route == AttnRoute::DecodeFusedQkSoftmax ||
                   route == AttnRoute::DecodeStaged;
        }
        bool isStaged() const {
            return route == AttnRoute::DecodeStaged || route == AttnRoute::PrefillStaged;
        }
        bool isFusedPrefill() const {
            return route == AttnRoute::PrefillFaTc || route == AttnRoute::PrefillFaSg;
        }
        bool qkSimdReduce = false, qkSimdMatrix = false, qkTensorMatrix = false;
        bool qkvSimdReduce = false, qkvSimdMatrix = false;
        bool sftmSimdReduce = false, copySimdReduce = false;
        // Staged prefill: QK dispatches the causal trapezoid, softmax/AV only the valid row prefix.
        bool causalTri = false, causalBound = false;
        // Power of two; bounds mTempQK + mTempSoftMax bytes, changes dispatch only.
        int qseqSplit = 1;
        // sdpaNsg/sdpaTwoPass are compile outcomes, so they stay out of the route key.
        int sdpaNsg = 8, sdpaKvb = 1, sdpaQhPerTg = 1, sdpaKvBlocks = 0;
        bool sdpaTwoPass = false;
        bool qkQsplit = false;
        int faTcWN = 1; // Head-dimension SIMD-groups per query tile.
        int faSgBq = 64, faSgNsg = 8;
    };

    void _init();
    ErrorCode compilePipelines(const std::vector<Tensor*>& inputs);
    ErrorCode preparePipelines(const std::vector<Tensor*>& inputs);
    ErrorCode updateExecutionState(const std::vector<Tensor*>& inputs);
    ErrorCode updateKVCache();
    ErrorCode handleKVAllocMemory();
    // Prepare passes a representative kv length, execute the live one after the cache update.
    void _resolveRoute(const std::vector<Tensor*>& inputs, int totalKv);
    AttnFacts _collectFacts(const std::vector<Tensor*>& inputs, int totalKv) const;
    AttnPlan _resolvePlan(const AttnFacts& facts) const;
    static bool _resolveSdpa(const AttnFacts& facts, uint8_t veto, AttnPlan& plan);
    int _resolveQseqSplit(const AttnFacts& facts, AttnRoute route) const;
    int _pastKvLength() const;
    uint64_t pipelineSignature() const;
    // Pipelines only; routing decisions are never cached here.
    struct PipelinePlan {
        id<MTLComputePipelineState> qk = nil, qkv = nil, copy = nil, softmax = nil;
        id<MTLComputePipelineState> prefillQk = nil, prefillQkv = nil, qkSoftmax = nil;
        id<MTLComputePipelineState> sdpa = nil, flashTc = nil, flashSg = nil;
        id<MTLComputePipelineState> sdpaPartial = nil, sdpaReduce = nil;
        uint8_t veto = AttnVetoNone;
        bool sdpaTwoPass = false;
        int sdpaNsg = 0;
    };
    std::map<uint64_t, PipelinePlan> mPipelinePlans;
    std::vector<int> mPreparedShape;
    AttentionCopyParam _makeCopyParam(const Tensor* key, const Tensor* value) const;
    AttentionQKVParam _makeQKVParam(const std::vector<Tensor*>& inputs, int seqLenPiece) const;
    AttentionSoftmaxParam _makeSoftmaxParam(int seqLenPiece) const;
    // Structural fingerprint of the current encode path; replay is only valid
    // while it matches the value captured at the end of the last onEncode.
    // forReplay=false drops fields that only affect recorded event layout or dispatch.
    uint32_t _pathSignature(bool forReplay) const;
    uint32_t mLastEncodeSig = 0;
    id<MTLBuffer> mLastKScaleBuffer = nil;
    id<MTLBuffer> mLastVScaleBuffer = nil;
    // K/V cache tensor identities at last encode; onReplayUpdate detects a
    // KV-expansion swap by pointer identity before touching the recording.
    const Tensor* mLastKTensor = nullptr;
    const Tensor* mLastVTensor = nullptr;
    const Tensor* mLastSdpaPartials = nullptr;

    AttnFacts mFacts;
    AttnPlan mPlan;
    uint8_t mVeto = AttnVetoNone;
    bool mKVCache = true;
    std::shared_ptr<MetalKVCacheManager> mKVCacheManager = nullptr;
    // Reuses another layer's kvcache: reads the shared buffers, never writes or resizes them.
    bool mIsKVShared = false;
    float mScale;
    bool mOutputC4 = false;
    std::shared_ptr<Tensor> mTempQK, mTempSoftMax;
    int mNumHead = 0, mHeadDim = 0, mValueH = 0, mKvNumHead = 0;
    int mSeqLen;
    int mKvAlignNum = 32;

    id<MTLComputePipelineState> mKernel_copy = nil;
    id<MTLComputePipelineState> mKernel_softmax = nil;
    id<MTLComputePipelineState> mKernel_qk = nil;
    id<MTLComputePipelineState> mKernel_qkv = nil;
    id<MTLComputePipelineState> mKernelPrefill_qk = nil;
    id<MTLComputePipelineState> mKernelPrefill_qkv = nil;
    id<MTLComputePipelineState> mKernel_qk_softmax = nil;
    id<MTLComputePipelineState> mKernel_sdpa = nil;
    id<MTLComputePipelineState> mKernelSdpaPartial = nil;
    id<MTLComputePipelineState> mKernelSdpaReduce = nil;
    id<MTLComputePipelineState> mKernel_faTc = nil;
    id<MTLComputePipelineState> mKernel_faSg = nil;
    // Unnormalized two-pass partials: [blocks x B x H x (hd+2)] fp32.
    std::shared_ptr<Tensor> mSdpaPartials;

    KVMeta* mMeta;
    // Dense mask (dims >= 2), bound at buffer 7 for QK/softmax kernels and buffer 8 for
    // the fused prefill kernels; lower ranks are one-element sentinels (core/AttentionCommon.hpp).
    bool mHasTensorMask = false;
    bool mIsAddMask = false;
    bool mHasSinks = false;
    int mBatch, mKvSeqLen, mKvMaxLen, mCurrentKvLen = 0;
    std::shared_ptr<Tensor> mTempK, mTempV;
    bool mKvInDisk;
    // Prefix-cache disk use is resolved per forward, not from this flag.
    bool mKvDirOnDisk = false;
    bool _predictKvInDisk() const;
    void _resolveKvFormat();

    std::shared_ptr<KVQuantParameter> mKVQuantParameter = nullptr;
    bool mQuantValue = false;
    bool mQuantKey = false;
};

} // namespace MNN
#endif /* MNN_SUPPORT_TRANSFORMER_FUSE */
#endif /* MNN_METAL_ENABLED */
#endif /* MetalAttention_hpp */
