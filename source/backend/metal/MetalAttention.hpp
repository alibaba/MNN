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
class AttentionBufExecution : public MetalExecution {
public:
    AttentionBufExecution(Backend* backend, bool kvCache, bool outputC4, float attnScale,
                          std::shared_ptr<KVQuantParameter> kvQuantParam);
    virtual ~AttentionBufExecution() = default;
    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;

    virtual void onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                          id<MTLComputeCommandEncoder> encoder) override;
    // Encode replay: param-buffer contents and kv-dependent grids/bytes are
    // patched per token; structural changes (decode-path switch, KV realloc)
    // bail out to a normal encode + re-record (stale bindings are caught by
    // replay validation).
    virtual bool onReplayUpdate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override {
        if (nullptr == dst) {
            return true;
        }
        auto exe = new AttentionBufExecution(bn, mKVCache, mOutputC4, mAttnScale, mKVQuantParameter);
        if (mKVCache && bn->getMetaPtr() == mMeta && mMeta != nullptr) {
            exe->mKVCacheManager = mKVCacheManager;
        }
        *dst = exe;
        MNN_METAL_PROFILE_REGISTER_CLONE(bn, op, *dst);
        return true;
    }

private:
    void _init();
    void compilerShader(const std::vector<Tensor*>& inputs);
    void handleKVAllocMemory();
    // Per-token encode-path decisions (split-kv / fused decode-qk-softmax /
    // simd flags / causal flags) + KV memory bookkeeping. Shared by onEncode
    // and onReplayUpdate so replayed tokens recompute identical state.
    void _computePathFlags(const std::vector<Tensor*>& inputs);
    void _writeCopyParam(const Tensor* key, const Tensor* value);
    void _writeQKVParam(const std::vector<Tensor*>& inputs, int seqLenPiece);
    void _writeSoftmaxParam(int seqLenPiece, int seq_idx);
    // Structural fingerprint of the current encode path; replay is only valid
    // while it matches the value captured at the end of the last onEncode.
    uint32_t _pathSignature() const;
    uint32_t mLastEncodeSig = 0;
    id<MTLBuffer> mLastKScaleBuffer = nil;
    id<MTLBuffer> mLastVScaleBuffer = nil;
    // K/V cache tensor identities at last encode. KV expansion DESTROYS the
    // old cache tensors (mPastKey.reset), leaving dangling tensor pointers in
    // the recorded bindings — onReplayUpdate must detect the swap by pointer
    // identity (never dereferenced) before any recorded tensor is validated.
    const Tensor* mLastKTensor = nullptr;
    const Tensor* mLastVTensor = nullptr;
    bool mKVCache = true;
    std::shared_ptr<MetalKVCacheManager> mKVCacheManager = nullptr;
    float mAttnScale = 0.0f;
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
    id<MTLBuffer> mParamQKV;
    id<MTLBuffer> mParamSoftmax;
    id<MTLBuffer> mParamCopy;

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
    // Single-pass fused decode SDPA (roadmap #20 restart): decode_splitkv runs
    // a single workgroup (one threadgroup per q head), no reduce dispatch, final
    // output written by the kernel. The sole kv>=threshold fused decode path
    // (split-KV removed 2026-07-30). Default auto-on (MNN_METAL_DECODE_SDPA);
    // =0 falls back to fused decode_qk_softmax (kv<=cap) / three-stage
    // decode_qk (kv>cap). NSG device-tiered via MNN_METAL_DECODE_SDPA_NSG
    // (0 = auto: M5->8, M4-class->32).
    bool mSdpaSinglePass = false;
    int mSdpaNsg = 8;
    id<MTLComputePipelineState> mKernel_sdpa = nil;
    // Fused prefill attention on the Metal tensor API (matmul2d + input
    // cooperative tensors, single-simdgroup scope): S and O stay in registers
    // across the whole KV sweep, so the O(n^2) score matrix never reaches
    // global memory. MNN_METAL_PREFILL_FA_TENSORAPI (default on for causal models).
    bool mFaNaxPrefill = false;
    bool mFaNaxUnavailable = false;
    id<MTLComputePipelineState> mKernel_faNax = nil;
    // Causal triangular dispatch for prefill_qk (simdgroup-matrix path):
    // launch only the trapezoid of tiles at or below the causal diagonal;
    // the CAUSAL_BOUND softmax reduces/writes only each row's causally-valid
    // prefix (+24 zero pad) and prefill_qkv truncates its AV loop, so the
    // upper-triangle region of mTempQK/mTempSoftMax is never read or written.
    // Interior (fully-valid) tiles also skip per-element mask logic. Gated on
    // mCausalLayout (standard causal mask, data-driven).
    bool mQkCausalTri = false;
    // CAUSAL_BOUND: bounded softmax + prefill_qkv AV early-exit. Independent of
    // the QK-side CAUSAL_TRI trapezoid dispatch — this can activate on the
    // tensor-API path (M5+) too, whereas CAUSAL_TRI is currently only wired for
    // the simdgroup-matrix path (16x16 tile coord inversion). Also gated on
    // mCausalLayout.
    bool mCausalBound = false;
    // Data-driven causal layout: true iff the mask is standard lower-triangular
    // causal (scalar sentinel / absent + kv-cache), so DEFAULT_MASK causal
    // arithmetic is valid and causal-tri / causal-bound / FA-v1 / faNax may
    // engage. A real-tensor mask (SWA / prefix-LM / cross-attn) forces this
    // false so every element is honored via ADD_MASK/SET_MASK. Detected in
    // _computePathFlags from inputs[3]'s shape (mHasTensorMask), no env needed.
    bool mCausalLayout = false;
    // Fused prefill flash-attention. Currently opt-in via env var
    // MNN_ENABLE_FLASH_ATTN_PREFILL=1 and gated to head_dim in {64,128},
    // non-quant KV, causal-only. Kernel body TBD in follow-up commit; for now
    // this flag routes through the existing prefill_qk/softmax/prefill_qkv
    // pipeline (i.e. no behavior change) so the wiring and eligibility check
    // can land independently of the fused shader.
    bool mFlashAttnPrefill = false;

private:
    // A per-element tensor mask is present (dims>=2): must be bound at buffer 7
    // and read per position. A scalar mask (dims<2) is NOT a tensor mask -- it is
    // llm.cpp's "no per-element mask needed" sentinel; causal-ness for that case
    // comes from kv-cache (DEFAULT_MASK), tracked by mCausalLayout.
    bool mHasTensorMask = false;
    bool mIsAddMask = false;
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
