//
//  ArgMaxExecution.hpp
//  MNN
//
//  Created by MNN on 2020/07/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ArgMaxExecution_hpp
#define ArgMaxExecution_hpp
#include <vector>
#include "backend/cuda/core/CUDABackend.hpp"
#include "core/Execution.hpp"
namespace MNN {
namespace CUDA {
class ArgMaxExecution : public Execution {
public:
    ArgMaxExecution(const Op* op, Backend *backend);
    virtual ~ArgMaxExecution();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    const Op* mOp;
    int mAxis;
    int mInside;
    int mOutside;
    int mDim;
    bool mSplitKernel = false;
    int mSecondArgLen;
    void *mTempDataBuffer;
    void *mTempIndexBuffer;
};
} // namespace CUDA
} // namespace MNN

#endif
