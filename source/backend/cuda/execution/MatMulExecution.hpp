//
//  MatMulExecution.hpp
//  MNN
//
//  Created by MNN on 2020/07/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MatMulExecution_hpp
#define MatMulExecution_hpp
#include <vector>
#include "backend/cuda/core/CUDABackend.hpp"
#include "core/Execution.hpp"
namespace MNN {
namespace CUDA {
class MatMulExecution : public Execution {
public:
    MatMulExecution(bool transposeA, bool transposeB, Backend *backend);
    virtual ~MatMulExecution();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<Tensor> mTempOutput;
    bool mTransposeA;
    bool mTransposeB;
};
} // namespace CUDA
} // namespace MNN

#endif
