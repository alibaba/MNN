#ifndef MNN_CUDA_LINEAR_ATTENTION_EXECUTION_HPP
#define MNN_CUDA_LINEAR_ATTENTION_EXECUTION_HPP

#include "core/Execution.hpp"
#include "backend/cuda/core/CUDABackend.hpp"

namespace MNN {
namespace CUDA {

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

struct CUDAStateCache {
    std::shared_ptr<Tensor> mConvState;      // [B, D, convStateSize] on GPU, float32
    std::shared_ptr<Tensor> mRecurrentState; // [B, H, d_k, d_v] on GPU, float32
};

class CUDALinearAttention : public Execution {
public:
    CUDALinearAttention(Backend* backend, const MNN::Op* op);
    virtual ~CUDALinearAttention();
    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;

private:
    CUDABackend* mCudaBackend;
    std::string mAttentionType;
    int mNumKHeads;
    int mNumVHeads;
    int mHeadKDim;
    int mHeadVDim;
    bool mUseQKL2Norm;
    int mPrecision;

    // Persistent state shared between prefill/decode via onClone
    std::shared_ptr<CUDAStateCache> mStateCache;

    // Temporary GPU buffers (DYNAMIC)
    std::shared_ptr<Tensor> mConvOut;           // [B, D, L] conv output after SiLU
    std::shared_ptr<Tensor> mConvOutTransposed; // [B, L, D] transposed for coalesced prefill access
};

#endif // MNN_SUPPORT_TRANSFORMER_FUSE

} // namespace CUDA
} // namespace MNN

#endif // MNN_CUDA_LINEAR_ATTENTION_EXECUTION_HPP
