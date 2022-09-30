//
//  ArgMinExecution.hpp
//  MNN
//
//  Created by MNN on 2022/06/29.
//  Copyright © 2018 - 2022, Alibaba Group Holding Limited
//

#ifndef ArgMinExecution_hpp
#define ArgMinExecution_hpp
#include <vector>
#include "backend/cuda/core/CUDABackend.hpp"
#include "core/Execution.hpp"
namespace MNN {
namespace CUDA {
class ArgMinExecution : public Execution {
public:
    ArgMinExecution(const Op* op, Backend *backend);
    virtual ~ArgMinExecution();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    const Op* mOp;
    int mAxis;
    int mInside;
    int mOutside;
    int mDim;
};
} // namespace CUDA
} // namespace MNN

#endif
