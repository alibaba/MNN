// Copyright © Alibaba Group Holding Limited. All rights reserved.
#ifndef MNN_COREMLATTENTION_HPP
#define MNN_COREMLATTENTION_HPP
#include "CoreMLCommonExecution.hpp"
#include "core/KVMeta.hpp"
namespace MNN {
class CoreMLAttention : public CoreMLCommonExecution {
public:
    CoreMLAttention(Backend*, const Op*, const std::vector<Tensor*>&, const std::vector<Tensor*>&);
    ErrorCode onResize(const std::vector<Tensor*>&, const std::vector<Tensor*>&) override;
    ErrorCode onExecute(const std::vector<Tensor*>&, const std::vector<Tensor*>&) override;

private:
    bool mKVCache = false;
    int mMaxSeqLen = 4096;
    int mSeqLen = 0, mKeyLen = 0;
    std::shared_ptr<Tensor> mUpdate, mKeep, mMask;
};
} // namespace MNN
#endif
