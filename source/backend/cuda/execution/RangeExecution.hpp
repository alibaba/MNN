//
//  RangeExecution.hpp
//  MNN
//
//  Created by MNN on 2022/04/21.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef RangeExecution_hpp
#define RangeExecution_hpp

#include "core/Execution.hpp"

#include <vector>
#include "backend/cuda/core/CUDABackend.hpp"

namespace MNN {
namespace CUDA {
class RangeExecution : public Execution {
public:
    RangeExecution(Backend *backend);
    virtual ~RangeExecution() = default;

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
};

} // namespace CUDA
} // namespace MNN
#endif /* SelectExecution_hpp */
