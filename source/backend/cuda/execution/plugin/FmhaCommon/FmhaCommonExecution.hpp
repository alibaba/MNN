//
//  FmhaV2Execution.hpp
//  MNN
//
//  Created by MNN on 2024/02/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifndef FmhaCommonExecution_hpp
#define FmhaCommonExecution_hpp

#include "backend/cuda/core/CUDABackend.hpp"
#include "fused_multi_head_attention/kernel_forward.h"

namespace MNN {
namespace CUDA {

class FmhaCommonExecution : public Execution {
public:
    FmhaCommonExecution(const MNN::Op* op, Backend *backend);
    virtual ~FmhaCommonExecution() = default;

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    template <int kQueriesPerBlock, int kKeysPerBlock, int kMaxK>
    int run_attention(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
private:
    int32_t mNumHeads;
    int32_t mSeqLen;
    int32_t mSeqLenKV;
    int32_t mBatchSize;
    int32_t mHeadSize;
    int32_t mHeadSizeV;
    void* mQ_Buffer;
    void* mK_Buffer;
    void* mV_Buffer;
    void* mAcc_Buffer;
    int32_t mSM{};
    int mType = 0;
};

} // namespace CUDA
} // namespace MNN
#endif /* FmhaCommonExecution_hpp */