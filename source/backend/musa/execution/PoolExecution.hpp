//
//  PoolExecution.hpp
//  MNN
//
//  Created by MNN on 2026/02/25.
//  Copyright © 2026, Alibaba Group Holding Limited
//

#ifndef PoolExecution_hpp
#define PoolExecution_hpp

#include "core/Execution.hpp"
#include "backend/musa/core/MusaBackend.hpp"

namespace MNN {
namespace MUSA {

class PoolExecution : public Execution {
public:
    PoolExecution(PoolType type, const std::vector<int>& kernels, const std::vector<int>& strides, 
                  const std::vector<int>& pads, Backend *backend);
    virtual ~PoolExecution() = default;

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    MusaRuntime *mRuntime;
    PoolType mType;
    std::vector<int> mKernels;
    std::vector<int> mStrides;
    std::vector<int> mPads;
    int mBatch;
    int mChannels;
    int mInputHeight;
    int mInputWidth;
    int mOutputHeight;
    int mOutputWidth;
};

} // namespace MUSA
} // namespace MNN
#endif /* PoolExecution_hpp */
