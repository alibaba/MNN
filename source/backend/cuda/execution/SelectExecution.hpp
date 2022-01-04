//
//  SelectExecution.hpp
//  MNN
//
//  Created by MNN on 2021/12/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef SelectExecution_hpp
#define SelectExecution_hpp

#include "core/Execution.hpp"

#include <vector>
#include "backend/cuda/core/CUDABackend.hpp"

namespace MNN {
namespace CUDA {

class SelectExecution : public Execution {
public:
    SelectExecution(Backend *backend);
    virtual ~SelectExecution() = default;

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
};

} // namespace CUDA
} // namespace MNN
#endif /* SelectExecution_hpp */
