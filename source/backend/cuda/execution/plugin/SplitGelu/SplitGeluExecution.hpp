//
//  SplitGeluExecution.hpp
//  MNN
//
//  Created by MNN on 2023/09/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#ifndef SplitGeluExecution_hpp
#define SplitGeluExecution_hpp

#include "splitGeLUKernel.h"
#include "backend/cuda/core/CUDABackend.hpp"

namespace MNN {

namespace CUDA {
class SplitGeluExecution : public Execution {
public:
    SplitGeluExecution(Backend *backend);
    virtual ~SplitGeluExecution() = default;

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    float mFDiv{};
    float mFAdd{};
    float mFMul{};
};

} // namespace CUDA
} // namespace MNN
#endif /* SplitGeluExecution_hpp */
#endif