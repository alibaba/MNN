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

#if MNN_METAL_ENABLED
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

namespace MNN {

struct MetalStateCache {
    std::shared_ptr<Tensor> mConvState;      // Conv1D padding state: [B, D, kernel_size - 1]
    std::shared_ptr<Tensor> mRecurrentState; // Gated Delta Rule recurrent state S: [B, H, d_k, d_v]
};

class MetalLinearAttention : public MetalExecution {
public:
    MetalLinearAttention(Backend *backend, const MNN::Op* op);
    virtual ~MetalLinearAttention() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual void onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) override;
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

    // Temporary buffer (DYNAMIC)
    std::shared_ptr<Tensor> mConvOut;         // [B, D, L]
    std::shared_ptr<Tensor> mQ;               // [B, L, H, d_k]
    std::shared_ptr<Tensor> mK;               // [B, L, H, d_k]
    std::shared_ptr<Tensor> mV;               // [B, L, H, d_v]
    // Param buffer for shader
    id<MTLBuffer> mParamBuffer;

    // Pipeline states
    id<MTLComputePipelineState> mConvSiluPipeline;
    id<MTLComputePipelineState> mConvStateUpdatePipeline;
    id<MTLComputePipelineState> mQKVPrepPipeline;
    id<MTLComputePipelineState> mGatedDeltaRulePipeline;
    id<MTLComputePipelineState> mGatedDeltaRuleSGPipeline;    // simdgroup: prefill (reads Q/K/V)
    id<MTLComputePipelineState> mGatedDeltaRuleFusedSGPipeline; // simdgroup: decode (reads conv_out)
    bool mUseSimdGroupOpt = false;
};

} // namespace MNN
#endif /* MNN_SUPPORT_TRANSFORMER_FUSE */
#endif /* MNN_METAL_ENABLED */
#endif /* MetalLinearAttention_hpp */
