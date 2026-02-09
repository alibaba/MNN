//
//  LinearAttentionBufExecution.hpp
//  MNN
//
//  Created by MNN on 2026/02/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#ifndef LinearAttentionBufExecution_hpp
#define LinearAttentionBufExecution_hpp

#include "backend/opencl/execution/image/CommonExecution.hpp"
#include "core/OpCommonUtils.hpp"

namespace MNN {
namespace OpenCL {

struct OpenCLStateCache {
    std::shared_ptr<Tensor> mConvState;      // Conv1D padding state: [B, D, kernel_size - 1]
    std::shared_ptr<Tensor> mRecurrentState; // Gated Delta Rule recurrent state S: [B, H, d_k, d_v]
};

class LinearAttentionBufExecution : public CommonExecution {
public:
    LinearAttentionBufExecution(const MNN::Op *op, Backend *backend);
    virtual ~LinearAttentionBufExecution() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;

private:
    std::string mAttentionType;
    int mNumKHeads;
    int mNumVHeads;
    int mHeadKDim;
    int mHeadVDim;
    bool mUseQKL2Norm;

    OpenCLBackend *mOpenCLBackend;

    // Persistent state buffers shared between prefill and decode via onClone
    std::shared_ptr<OpenCLStateCache> mStateCache;
    // Temporary conv output: [B * D * L]
    std::shared_ptr<Tensor> mConvOut;

    // Kernels
    std::shared_ptr<KernelWrap> mKernelConvSilu;
    std::shared_ptr<KernelWrap> mKernelConvStateUpdate;
    std::shared_ptr<KernelWrap> mKernelGatedDeltaRule;

    // Work sizes
    std::vector<uint32_t> mGWSConvSilu;
    std::vector<uint32_t> mLWSConvSilu;
    std::vector<uint32_t> mGWSConvStateUpdate;
    std::vector<uint32_t> mLWSConvStateUpdate;
    std::vector<uint32_t> mGWSGatedDeltaRule;
    std::vector<uint32_t> mLWSGatedDeltaRule;
};

} // namespace OpenCL
} // namespace MNN
#endif /* LinearAttentionBufExecution_hpp */
#endif /* MNN_SUPPORT_TRANSFORMER_FUSE */
