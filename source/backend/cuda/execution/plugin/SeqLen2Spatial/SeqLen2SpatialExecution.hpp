//
//  SeqLen2SpatialExecution.hpp
//  MNN
//
//  Created by MNN on 2023/09/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#ifndef SeqLen2SpatialExecution_hpp
#define SeqLen2SpatialExecution_hpp

#include "seqLen2SpatialKernel.h"
#include "backend/cuda/core/CUDABackend.hpp"

namespace MNN {

namespace CUDA {
class SeqLen2SpatialExecution : public Execution {
public:
    SeqLen2SpatialExecution(Backend *backend);
    virtual ~SeqLen2SpatialExecution() = default;

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:

};

} // namespace CUDA
} // namespace MNN
#endif /* SeqLen2SpatialExecution_hpp */
#endif