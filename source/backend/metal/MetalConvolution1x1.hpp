//
//  MetalConvolution1x1.hpp
//  MNN
//
//  Created by MNN on 2019/02/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalConvolution1x1_hpp
#define MetalConvolution1x1_hpp

#import "MetalConvolutionCommon.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalConvolution1x1 : public MetalConvolutionCommon {
public:
    static bool isValid(const Convolution2D* conv, const Tensor* input);
    MetalConvolution1x1(Backend* backend, const MNN::Op* op);
    virtual ~MetalConvolution1x1() = default;
    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
    virtual void onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                          id<MTLComputeCommandEncoder> encoder) override;

    // Gate/Up fusion: called by the owning MetalFusedProj for the two members of
    // an exported FusedGateUp group. 'this' is the gate (leader), 'peer' the up
    // (follower); peerOutput is the up projection's output tensor.
    bool setupGateUpFusion(MetalConvolution1x1* peer, const Tensor* peerOutput);
    bool isGateUpLeader() const { return mIsGateUpLeader; }
    bool isGateUpFollower() const { return mIsGateUpFollower; }
    // QKV fusion: called by the owning MetalFusedProj for the three (or four,
    // e.g. Qwen3.5 linear-attention qkv/z/b/a) decode GEMV projections of one
    // exported group. 'this' (first in member order) becomes the leader and
    // dispatches all of them in a single grid.z=3/4 kernel; the followers'
    // onEncode become no-ops.
    bool setupQKVFusion(MetalConvolution1x1* peerK, const Tensor* peerKOutput,
                        MetalConvolution1x1* peerV, const Tensor* peerVOutput,
                        MetalConvolution1x1* peerW = nullptr, const Tensor* peerWOutput = nullptr);
    bool isQKVLeader() const { return mIsQKVLeader; }
    bool isQKVFollower() const { return mIsQKVFollower; }
    // Check if this Conv1x1 uses the 2sg decode GEMV pipeline (eligible for fusion)
    bool is2sgDecodePipeline() const { return mIs2sgDecode; }

    // Accessors for peer's buffers (used by leader during fused encode)
    std::shared_ptr<MNN::Tensor> getWeight() const { return mWeight; }
    std::shared_ptr<MNN::Tensor> getBias() const { return mBias; }
    std::shared_ptr<MNN::Tensor> getDequantScale() const { return mDequantScaleBias; }

    bool setupLNFusion(const Tensor* hiddenInput, const Tensor* residualInput, const Tensor* residualOutput,
                       std::shared_ptr<Tensor> gamma, float eps);

private:
    MetalConvolution1x1(Backend* backend, const MNN::Op* op, std::shared_ptr<MNN::Tensor> weight,
                        std::shared_ptr<MNN::Tensor> bias, std::shared_ptr<MNN::Tensor> dequantScale,
                        int dequantBits, float scaleCoef);
    id<MTLComputePipelineState> mPipeline;
    std::pair<MTLSize, MTLSize> mThreads;
    id<MTLComputePipelineState> mDequantPipeline;
    std::pair<MTLSize, MTLSize> mDequantThreads;
    bool mPreDequantWeight = false;
    std::shared_ptr<Tensor> mTempWeight;
    // Fused weight+scale buffer for decode GEMV optimization
    std::shared_ptr<Tensor> mFusedWeightScale;
    bool mUseFusedDecode = false;
    // Gate/Up fusion state
    bool mIs2sgDecode = false;                              // true if using conv1x1_gemv_g4m1_2sg_wquant_sg pipeline
    bool mIsGateUpLeader = false;                           // true if this is the gate (leader) in a fused pair
    bool mIsGateUpFollower = false;                         // true if this is the up (follower) in a fused pair
    MetalConvolution1x1* mGateUpPeer = nullptr;             // leader points to follower (up)
    const Tensor* mGateUpPeerOutput = nullptr;              // follower's output tensor
    id<MTLComputePipelineState> mGateUpFusedPipeline = nil; // fused pipeline with GATE_UP_FUSED
    id<MTLBuffer> mGateUpSegBuffer = nil;                   // {up_scale_coef} (gate uses cst.scale_coef)

    // QKV fusion state (see setupQKVFusion)
    bool mIsQKVLeader = false;
    bool mIsQKVFollower = false;
    MetalConvolution1x1* mQKVPeerK = nullptr;
    MetalConvolution1x1* mQKVPeerV = nullptr;
    MetalConvolution1x1* mQKVPeerW = nullptr;   // optional 4th projection
    const Tensor* mQKVPeerKOutput = nullptr;
    const Tensor* mQKVPeerVOutput = nullptr;
    const Tensor* mQKVPeerWOutput = nullptr;
    id<MTLComputePipelineState> mQKVFusedPipeline = nil;    // fused pipeline with QKV_FUSED
    id<MTLBuffer> mQKVSegBuffer = nil;  // {k_coef, v_coef, k_oslice, v_oslice[, w_coef, w_oslice]}
    bool mQKVCompactGrid = false;  // one packed grid.x range for all projections
    // Quant block count along IC (per output_slice); fused projections must match.
    int mBlockSize = 1;
    // C4 slices per Q4 quant block for the generalized 16-byte decode path.
    // Supported values are 8/16/32/64 (quant blocks 32/64/128/256). Zero means
    // the layout is not eligible. Recorded so fusion setup can compile the same
    // block shape as the standalone decode pipeline.
    int mQ4W16BlockSlices = 0;

    // Fused Q4/Q8 GEMM: kernel unpacks quantized weights in-kernel
    // (FUSED_Q4_REAL_UNPACK), skipping the dequant pre-pass and mTempWeight.
    // Kill-switch: MNN_METAL_W4W8_OUTER_DEQUANT_GEMM_TENSORAPI=1.
    bool mFusedQ4 = false;
    // K-split x4 for TG-starved speculative-verify shapes; fp32 partials land in
    // mKsplitPartial and are summed by mKsplitReducePipeline.
    bool mUseFusedKsplit = false;
    bool mKsplitM8 = false;   // ksplit uses the M8 tile
    id<MTLComputePipelineState> mKsplitReducePipeline = nil;
    std::shared_ptr<Tensor> mKsplitPartial;
    std::pair<MTLSize, MTLSize> mKsplitReduceThreads;
    // M=64 tile variant of the fused Q4 GEMM (conv1x1_fused_q4_gemm_stage_m64).
    // Halves grid.x for prefill (M_TILE=64 vs baseline M_TILE=32) — cuts
    // weight-read redundancy across TGs in half. Auto: fused + Q4 + area >= 128.
    bool mFusedQ4M64 = false;

    void bindLNBuffers(id<MTLComputeCommandEncoder> encoder);

    // LayerNorm fusion state
    bool mHasLNFusion = false;
    id<MTLComputePipelineState> mLNFusedPipeline = nil; // pipeline with LN_FUSED macro
    const Tensor* mLNHiddenInput = nullptr;             // LN inputs[1] → buffer 0
    const Tensor* mLNResidualInput = nullptr;           // LN inputs[0] → buffer 20
    const Tensor* mLNResidualOutput = nullptr;          // LN outputs[0] → buffer 22
    std::shared_ptr<Tensor> mLNGamma = nullptr;         // LN gamma → buffer 21
    id<MTLBuffer> mLNEpsBuffer = nil;                   // eps → buffer 23

#if MNN_METAL_OP_PROFILE
    // Kernel-variant tag used to distinguish which shader path this Conv1x1 dispatched
    // in the per-op profile output (e.g. "gemm_32x64_split_k_wq", "gemv_g4m1_2sg_wq").
    std::string mProfileTag;
#endif
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */

#endif /* MetalConvolution1x1_hpp */
