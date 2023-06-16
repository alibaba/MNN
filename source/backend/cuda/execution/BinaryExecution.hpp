//
//  BinaryExecution.hpp
//  MNN
//
//  Created by MNN on 2020/07/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef BinaryExecution_hpp
#define BinaryExecution_hpp
#include <vector>
#include "backend/cuda/core/CUDABackend.hpp"
#include "core/Execution.hpp"
#ifdef ENABLE_CUDA_QUANT
#include "int8/BinaryInt8Execution.hpp"
#endif

namespace MNN {
namespace CUDA {
class BinaryExecution : public Execution {
public:
    BinaryExecution(int opType, Backend *backend, int activationType = 0);
    virtual ~BinaryExecution();
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    int mType;
    int mActivationType;
};
} // namespace CUDA
} // namespace MNN

#endif
