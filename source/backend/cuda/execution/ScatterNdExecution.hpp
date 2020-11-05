//
//  ScatterNdExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ScatterNdExecution_hpp
#define ScatterNdExecution_hpp

#include "core/Execution.hpp"

#include <vector>
#include "backend/cuda/core/CUDABackend.hpp"

namespace MNN {
namespace CUDA {

class ScatterNdExecution : public Execution {
public:
    ScatterNdExecution(Backend *backend);
    virtual ~ScatterNdExecution();

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    CUDARuntime *mRuntime;
    int mIndicesLastDim;
    int mIndexes;
    int mAccNumber;
    int mOutElementSize;
    void *mDimsToCount;
    std::shared_ptr<Tensor> dimsTensor;
};

} // namespace CUDA
} // namespace MNN
#endif /* ScatterNdExecution_hpp */
