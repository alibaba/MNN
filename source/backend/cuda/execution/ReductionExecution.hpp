//
//  ReductionExecution.hpp
//  MNN
//
//  Created by MNN on 2020/08/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ReductionExecution_hpp
#define ReductionExecution_hpp
#include <vector>
#include "backend/cuda/core/CUDABackend.hpp"
#include "core/Execution.hpp"
#include "ReductionTemplate.cuh"
namespace MNN {
namespace CUDA {
class ReductionExecution : public Execution {
public:
    ReductionExecution(ReductionType opType, int axis, Backend *backend);
    virtual ~ReductionExecution();
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    ReductionType mType;
    int mAxis;
    ReduceParam mCpuParam;
};
} // namespace CUDA
} // namespace MNN

#endif