//
//  FmhcaExecution.hpp
//  MNN
//
//  Created by MNN on 2023/09/13.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#ifndef FmhcaExecution_hpp
#define FmhcaExecution_hpp

#include "fmha_cross_attention/include/fmha_cross_attention.h"
#include "backend/cuda/core/CUDABackend.hpp"

namespace MNN {
class FusedMultiHeadCrossAttentionKernel;

namespace CUDA {
class FmhcaExecution : public Execution {
public:
    FmhcaExecution(const MNN::Op* op, Backend *backend);
    virtual ~FmhcaExecution() = default;

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    int32_t runFMHCAKernel(void const* devQ, void const* devKV, void* cuSeqlensQ, void* cuSeqlensKV, void* devOutput,
        int32_t sm, FusedMultiHeadCrossAttentionKernel const* kernels, int32_t b = 2, int32_t h = 8, int32_t d = 64,
        int32_t seqQ = 4096, int32_t seqKV = 77, cudaStream_t stream = 0);
    static bool isValid(const MNN::Op* op, Backend *backend, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);

private:
    void* mSeqLenQDevPtr;
    void* mSeqLenKVDevPtr;
    int32_t mNumHeads;
    int32_t mSeqLenQ;
    int32_t mSeqLenKV;
    int32_t mBatchSize;
    FusedMultiHeadCrossAttentionKernel const* mKernels{};
    int32_t mSM{};
};

} // namespace CUDA
} // namespace MNN
#endif /* FmhcaExecution_hpp */
#endif