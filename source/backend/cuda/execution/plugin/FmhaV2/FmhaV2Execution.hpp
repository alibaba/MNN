//
//  FmhaV2Execution.hpp
//  MNN
//
//  Created by MNN on 2023/09/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#ifndef FmhaV2Execution_hpp
#define FmhaV2Execution_hpp

#include "fmha_flash_attention/include/fmha_flash_attention.h"
#include "backend/cuda/core/CUDABackend.hpp"

namespace MNN {
class FusedMultiHeadFlashAttentionKernel;

namespace CUDA {
class FmhaV2Execution : public Execution {
public:
    FmhaV2Execution(const MNN::Op* op, Backend *backend);
    virtual ~FmhaV2Execution() = default;

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    int32_t runFMHFAKernel(void const* devQKV, void* cuSeqlens, void* devOutput, size_t total, int32_t sm,
        FusedMultiHeadFlashAttentionKernel const* kernels, int32_t b = 2, int32_t h = 8, int32_t d = 64, int32_t s = 4096,
        cudaStream_t stream = 0);
    static bool isValid(const MNN::Op* op, Backend *backend, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);

private:
    void* mSeqLenDevPtr;
    int32_t mNumHeads;
    int32_t mSeqLen;
    int32_t mBatchSize;
    FusedMultiHeadFlashAttentionKernel const* mKernels{};
    int32_t mSM{};
};

} // namespace CUDA
} // namespace MNN
#endif /* FmhaV2Execution_hpp */
#endif
