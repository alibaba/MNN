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
    static bool isValid(const Convolution2D *conv, const Tensor *input);
    MetalConvolution1x1(Backend *backend, const MNN::Op *op);
    virtual ~MetalConvolution1x1() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
    virtual void onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder)  override;

    // Gate/Up fusion: called by MetalMulSiluVec to pair gate and up projections
    // 'this' becomes the leader (gate), 'peer' becomes the follower (up)
    // peerOutput is the up projection's output tensor (needed for buffer binding)
    bool setupGateUpFusion(MetalConvolution1x1* peer, const Tensor* peerOutput);
    bool isGateUpLeader() const { return mIsGateUpLeader; }
    bool isGateUpFollower() const { return mIsGateUpFollower; }
    // Check if this Conv1x1 uses the 2sg decode GEMV pipeline (eligible for fusion)
    bool is2sgDecodePipeline() const { return mIs2sgDecode; }

    // QKV fusion: called by MetalBackend::matchQKVFusions to triple Q/K/V projections
    // 'this' becomes the leader (Q), 'peerK' and 'peerV' become followers
    bool setupQKVFusion(MetalConvolution1x1* peerK, const Tensor* peerKOutput,
                        MetalConvolution1x1* peerV, const Tensor* peerVOutput);
    bool isQKVLeader() const { return mIsQKVLeader; }
    bool isQKVFollower() const { return mIsQKVFollower; }
    // Accessors for peer's buffers (used by leader during fused encode)
    std::shared_ptr<MNN::Tensor> getWeight() const { return mWeight; }
    std::shared_ptr<MNN::Tensor> getBias() const { return mBias; }
    std::shared_ptr<MNN::Tensor> getDequantScale() const { return mDequantScaleBias; }
    id<MTLBuffer> getConstBuffer() const { return mConstBuffer; }

    bool setupLNFusion(const Tensor* hiddenInput, const Tensor* residualInput,
                       const Tensor* residualOutput, std::shared_ptr<Tensor> gamma, float eps);

private:
    MetalConvolution1x1(Backend *backend, const MNN::Op *op, std::shared_ptr<MNN::Tensor> weight, std::shared_ptr<MNN::Tensor> bias, std::shared_ptr<MNN::Tensor> dequantScale, int dequantBits, float scaleCoef);
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
    bool mIs2sgDecode = false;           // true if using conv1x1_gemv_g4m1_2sg_wquant_sg pipeline
    bool mIsGateUpLeader = false;        // true if this is the gate (leader) in a fused pair
    bool mIsGateUpFollower = false;      // true if this is the up (follower) in a fused pair
    MetalConvolution1x1* mGateUpPeer = nullptr;  // leader points to follower (up)
    const Tensor* mGateUpPeerOutput = nullptr;    // follower's output tensor
    id<MTLComputePipelineState> mGateUpFusedPipeline = nil;  // fused pipeline with GATE_UP_FUSED
    id<MTLBuffer> mGateUpSegBuffer = nil;         // {up_scale_coef} (gate uses cst.scale_coef)

    // QKV fusion state
    bool mIsQKVLeader = false;           // true if this is Q (leader) in a QKV triple
    bool mIsQKVFollower = false;         // true if this is K or V (follower)
    MetalConvolution1x1* mQKVPeerK = nullptr;    // leader points to K
    MetalConvolution1x1* mQKVPeerV = nullptr;    // leader points to V
    const Tensor* mQKVPeerKOutput = nullptr;      // K's output tensor
    const Tensor* mQKVPeerVOutput = nullptr;      // V's output tensor
    id<MTLComputePipelineState> mQKVFusedPipeline = nil;  // fused pipeline with QKV_FUSED
    id<MTLBuffer> mQKVSegBuffer = nil;            // {q_groups, k_groups, k_oc_slice, v_oc_slice}
    MTLSize mQKVOriginalGrid = {0, 0, 0};         // original grid before QKV fusion modified mThreads

    void bindLNBuffers(id<MTLComputeCommandEncoder> encoder);

    // LayerNorm fusion state
    bool mHasLNFusion = false;
    id<MTLComputePipelineState> mLNFusedPipeline = nil;  // pipeline with LN_FUSED macro
    const Tensor* mLNHiddenInput = nullptr;       // LN inputs[1] → buffer 0
    const Tensor* mLNResidualInput = nullptr;      // LN inputs[0] → buffer 20
    const Tensor* mLNResidualOutput = nullptr;     // LN outputs[0] → buffer 22
    std::shared_ptr<Tensor> mLNGamma = nullptr;     // LN gamma → buffer 21
    id<MTLBuffer> mLNEpsBuffer = nil;              // eps → buffer 23

#if MNN_METAL_OP_PROFILE
    // Kernel-variant tag used to distinguish which shader path this Conv1x1 dispatched
    // in the per-op profile output (e.g. "gemm_32x64_split_k_wq", "gemv_g4m1_2sg_wq").
    std::string mProfileTag;
#endif
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */

#endif /* MetalConvolution1x1_hpp */