//
//  PoolGradExecution.hpp
//  MNN
//
//  Created by MNN on 2019/10/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef PoolGradExecution_hpp
#define PoolGradExecution_hpp

#include <vector>
#include "backend/opencl/execution/CommonExecution.hpp"
namespace MNN {
namespace OpenCL {

class PoolGradExecution : public CommonExecution {
public:
    PoolGradExecution(const MNN::Op *op, Backend *backend);
    virtual ~PoolGradExecution();

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::vector<int> mKernels;
    std::vector<int> mStrides;
    PoolType mType;
};
}
}

#endif /* PoolGradExecution_hpp */
