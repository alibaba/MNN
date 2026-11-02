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

// Uniform structs for the attention kernels. All three are passed with setBytes
// rather than bound from a persistent MTLBuffer -- see _makeCopyParam in the .mm
// for the in-flight-overwrite hazard that forces this.

// KV-append / format-convert kernel.
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

// QK / softmax-fused / QKV kernels.
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

// Softmax kernel. Kept as a raw int sextuple to match the shader's layout exactly;
// slots are [inside, axis, outside, axis_align, causal_base, causal_bound_pad].
struct AttentionSoftmaxParam {
    int v[6];
};

class AttentionBufExecution : public MetalExecution {
public:
    AttentionBufExecution(Backend* backend, bool kvCache, bool outputC4,
                          std::shared_ptr<KVQuantParameter> kvQuantParam);
    virtual ~AttentionBufExecution() = default;
    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;

    virtual void onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                          id<MTLComputeCommandEncoder> encoder) override;
    // Encode replay patches param buffers and kv grids per token; structural
    // changes bail out to a normal encode + re-record.
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
            exe->mKVCacheManager = mKVCacheManager;
            exe->mIsKVShared = kvShared;
        }
        *dst = exe;
        MNN_METAL_PROFILE_REGISTER_CLONE(bn, op, *dst);
        return true;
    }

private:
    void _init();
    void compilerShader(const std::vector<Tensor*>& inputs);
    void handleKVAllocMemory();
    // How many q-sequence pieces the three-stage prefill splits into, so that
    // mTempQK + mTempSoftMax stay inside a byte budget. Requires mSeqLen and
    // mKvMaxLen to be current. Always a power of two; 1 means no split.
    int _resolveQseqSplit() const;
    // Per-token encode-path decisions (split-kv / fused decode-qk-softmax /
    // simd flags / causal flags) + KV memory bookkeeping. Shared by onEncode
    // and onReplayUpdate so replayed tokens recompute identical state.
    void _computePathFlags(const std::vector<Tensor*>& inputs);
    AttentionCopyParam _makeCopyParam(const Tensor* key, const Tensor* value) const;
    AttentionQKVParam _makeQKVParam(const std::vector<Tensor*>& inputs, int seqLenPiece) const;
    AttentionSoftmaxParam _makeSoftmaxParam(int seqLenPiece) const;
    // Structural fingerprint of the current encode path; replay is only valid
    // while it matches the value captured at the end of the last onEncode.
    uint32_t _pathSignature() const;
    uint32_t mLastEncodeSig = 0;
    id<MTLBuffer> mLastKScaleBuffer = nil;
    id<MTLBuffer> mLastVScaleBuffer = nil;
    // K/V cache tensor identities at last encode; onReplayUpdate detects a
    // KV-expansion swap by pointer identity before touching the recording.
    const Tensor* mLastKTensor = nullptr;
    const Tensor* mLastVTensor = nullptr;
    bool mKVCache = true;
    std::shared_ptr<MetalKVCacheManager> mKVCacheManager = nullptr;
    // Reuses another layer's kvcache: reads the shared buffers, never writes or resizes them.
    bool mIsKVShared = false;
    int _pastKvLength() const;
    float mScale;
    bool mOutputC4 = false;
    bool mShortSeq = false;
    std::shared_ptr<Tensor> mTempQK, mTempSoftMax;
    int mNumHead = 0, mHeadDim = 0, mValueH = 0, mKvNumHead = 0;
    int mSeqLen;
    // for simd/tensor maxtrix load alignment
    int mKvAlignNum = 32;
    id<MTLComputePipelineState> mKernel_softmax = nil;

    id<MTLComputePipelineState> mKernel_qk = nil;
    id<MTLComputePipelineState> mKernel_qkv = nil;
    id<MTLComputePipelineState> mKernel_copy = nil;
    id<MTLComputePipelineState> mKernel_qk_softmax = nil;
    id<MTLComputePipelineState> mKernelPrefill_qk = nil;
    id<MTLComputePipelineState> mKernelPrefill_qkv = nil;
    id<MTLComputePipelineState> mKernel_flashAttn = nil;

private:
    KVMeta* mMeta;
    bool mQkSimdReduce = false;
    bool mQkSimdMatrix = false;
    bool mQkTensorMatrix = false;
    bool mSftmSimdReduce = false;
    bool mQkvSimdReduce = false;
    bool mQkvSimdMatrix = false;
    bool mDecodeQkSoftmax = false;
    // Q-head-split decode_qk_softmax (auto: group_size==2, non-tensor-API
    // device, kv>=512): grid.z = group_size, one threadgroup per q-head.
    bool mQkQsplit = false;
    bool mCopySimdReduce = false;
    // Fused decode SDPA (decode_splitkv): one threadgroup per q-head group
    // sweeps the whole kv range. Gates the SDPA path as a whole; small kv
    // falls back to decode_qk_softmax / three-stage decode_qk.
    bool mSdpaSinglePass = false;
    int mSdpaNsg = 8;
    // q heads per threadgroup. 1 = one TG per q head; >1 shares each
    // KV row across a slice of the GQA group. Auto-resolved by group_size.
    int mSdpaQhPerTg = 1;
    // kv tokens per stream-loop iteration (SDPA_KVB kernel variant). Auto 2
    // at kv >= 2048 when the dispatch has < 16 threadgroups, else 1;
    // MNN_METAL_DECODE_SDPA_KVB forces.
    int mSdpaKvb = 1;
    // Auto mode flips mSdpaKvb as kv crosses its threshold; pre-build both
    // pipeline variants so the flip never compiles mid-decode.
    bool mSdpaKvbPrebuild = false;
    id<MTLComputePipelineState> mKernel_sdpa = nil;
    // Fused prefill attention on the tensor API: S/O stay in registers across
    // the whole KV sweep; causal-only.
    bool mFaTcPrefill = false;
    bool mFaTcUnavailable = false;
    // Head_dim slices spread over the dispatch grid's z axis (1 = whole head_dim
    // in one threadgroup). Splitting shrinks the persistent O register footprint
    // at the cost of recomputing QK per slice (2 at head_dim 256, 1 otherwise).
    int mFaTcDSplit = 1;
    id<MTLComputePipelineState> mKernel_faTc = nil;
    // Simdgroup-matrix fused prefill (prefill_flash_attn_sg): Q=32 / KV=16@D>=128 tiling,
    // S/O in simdgroup fragments, no mTempQK.
    bool mFaSgPrefill = false;
    bool mFaSgUnavailable = false;
    // FA-SG dispatch geometry, kept in sync with the compiled macro set:
    // q rows per threadgroup and simdgroups per threadgroup.
    int mFaSgBq = 32;
    int mFaSgNsg = 4;
    id<MTLComputePipelineState> mKernel_faSg = nil;
    // Causal triangular dispatch for prefill_qk: launch only the trapezoid
    // of tiles at or below the diagonal (needs mCausalLayout).
    bool mQkCausalTri = false;
    // CAUSAL_BOUND softmax + prefill_qkv AV early-exit. Independent of the
    // QK-side CAUSAL_TRI trapezoid dispatch and gated on mCausalLayout.
    bool mCausalBound = false;
    // True iff the mask is standard lower-triangular causal; required for the
    // causal-tri / causal-bound / FA / faTc paths. A real-tensor mask forces
    // this false so every element goes through ADD_MASK/SET_MASK.
    bool mCausalLayout = false;
    // Fused prefill flash-attention; opt-in via attention_mode/8 >= 1, gated
    // to head_dim in {64,128,256}, causal-only.
    bool mFlashAttnPrefill = false;

private:
    // Dense mask (dims >= 2), bound at buffer 7 for QK/softmax kernels and buffer 8 for
    // flash-attn. Lower ranks are one-element masks, classified by shape only (see
    // core/AttentionMaskConvention.hpp).
    bool mHasTensorMask = false;
    bool mIsAddMask = false;
    // Rank-1 one-element mask: fully visible, suppresses DEFAULT_MASK. Exclusive with mCausalLayout.
    bool mBidirectionalMask = false;
    int mBatch, mKvSeqLen, mKvMaxLen, mCurrentKvLen = 0;
    int mQseqSplitNum = 1;
    std::shared_ptr<Tensor> mTempK, mTempV;
    bool mKvInDisk;

    // KV static quantization (only V is quantized on Metal)
    std::shared_ptr<KVQuantParameter> mKVQuantParameter = nullptr;
    bool mQuantValue = false; // whether V cache is stored as int8
    bool mQuantKey = false;   // whether K cache is stored as int8
};

} // namespace MNN
#endif /* MNN_SUPPORT_TRANSFORMER_FUSE */
#endif /* MNN_METAL_ENABLED */
#endif /* MetalAttention_hpp */
