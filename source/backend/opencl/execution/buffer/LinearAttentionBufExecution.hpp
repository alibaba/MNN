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
    std::shared_ptr<Tensor> mRecurrentStateTune; // Gated Delta Rule recurrent state S: [B, H, d_k, d_v]
};

class LinearAttentionBufExecution : public CommonExecution {
public:
    LinearAttentionBufExecution(const MNN::Op *op, Backend *backend);
    virtual ~LinearAttentionBufExecution() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;

    // Chunked prefill: fully independent branch (called from onResize/onExecute when seqLen > 1)
    ErrorCode onResizeChunkedPrefill(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onExecuteChunkedPrefill(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);

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
    std::shared_ptr<KernelWrap> mKernell2Norm;
    std::shared_ptr<KernelWrap> mKernelGatedDeltaRule;

    // Work sizes
    std::vector<uint32_t> mGWSConvSilu;
    std::vector<uint32_t> mLWSConvSilu;
    std::vector<uint32_t> mGWSConvStateUpdate;
    std::vector<uint32_t> mLWSConvStateUpdate;
    std::vector<uint32_t> mGWSl2Norm;
    std::vector<uint32_t> mLWSl2Norm;
    std::vector<uint32_t> mGWSGatedDeltaRule;
    std::vector<uint32_t> mLWSGatedDeltaRule;

    // ─── Chunked prefill ───
    bool mUseChunkedPrefill = false;
    int mChunkSize = 16;
    int mNumChunks = 0;

    // Chunked prefill common kernels (independent copies)
    std::shared_ptr<Tensor> mConvOutPrefill;
    std::shared_ptr<KernelWrap> mKernelConvSiluPrefill;
    std::shared_ptr<KernelWrap> mKernelConvStateUpdatePrefill;
    std::shared_ptr<KernelWrap> mKernell2NormPrefill;
    std::vector<uint32_t> mGWSConvSiluPrefill, mLWSConvSiluPrefill;
    std::vector<uint32_t> mGWSConvStateUpdatePrefill, mLWSConvStateUpdatePrefill;
    std::vector<uint32_t> mGWSl2NormPrefill, mLWSl2NormPrefill;

    // Chunked prefill kernels
    std::shared_ptr<KernelWrap> mKernelChunkGCumsum;
    std::shared_ptr<KernelWrap> mKernelChunkNeumannAttn0;
    std::shared_ptr<KernelWrap> mKernelChunkNeumannAttn1;
    std::shared_ptr<KernelWrap> mKernelChunkCorrectV;
    std::shared_ptr<KernelWrap> mKernelChunkQKAttn;
    std::shared_ptr<KernelWrap> mKernelChunkVnew;
    std::shared_ptr<KernelWrap> mKernelChunkOutput;
    std::shared_ptr<KernelWrap> mKernelChunkOutputUpdate;

    // Chunked prefill work sizes
    std::vector<uint32_t> mGWSChunkGCumsum, mLWSChunkGCumsum;
    std::vector<uint32_t> mGWSChunkNeumannAttn0, mLWSChunkNeumannAttn0;
    std::vector<uint32_t> mGWSChunkNeumannAttn1, mLWSChunkNeumannAttn1;
    std::vector<uint32_t> mGWSChunkCorrectV, mLWSChunkCorrectV;
    std::vector<uint32_t> mGWSChunkQKAttn, mLWSChunkQKAttn;
    std::vector<uint32_t> mGWSChunkVnew, mLWSChunkVnew;
    std::vector<uint32_t> mGWSChunkOutput, mLWSChunkOutput;
    std::vector<uint32_t> mGWSChunkOutputUpdate, mLWSChunkOutputUpdate;

    // Chunked prefill intermediate buffers (float32)
    std::shared_ptr<Tensor> mGCumsumBuf;
    std::shared_ptr<Tensor> mAttnMatrixBuf;   // shared for Neumann attn and QK attn
    std::shared_ptr<Tensor> mVCorrectedBuf;
    std::shared_ptr<Tensor> mKCumdecayBuf;
    std::shared_ptr<Tensor> mVNewBuf;         // single chunk only
};

} // namespace OpenCL
} // namespace MNN
#endif /* LinearAttentionBufExecution_hpp */
#endif /* MNN_SUPPORT_TRANSFORMER_FUSE */
