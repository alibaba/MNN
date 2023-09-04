//
//  SoftmaxExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef SoftmaxExecution_hpp
#define SoftmaxExecution_hpp

#include <vector>
#include "ReductionTemplate.cuh"
#include "MNNCUDAFunction.cuh"
#include "backend/cuda/core/CUDABackend.hpp"
#include <float.h>

namespace MNN {
namespace CUDA {

class SoftmaxExecution : public Execution {
public:
    SoftmaxExecution(int axis, Backend *backend);
    virtual ~SoftmaxExecution();

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    int mAxis;
    Tensor mStorage;
    bool mNeedUnpackC4;
    ReduceParam mCpuParam;
    MemChunk mParam;
};

} // namespace CUDA
} // namespace MNN
#endif /* SoftmaxExecution_hpp */