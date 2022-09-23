//
//  NPUCommonExecution.hpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef NPUCommonExecution_hpp
#define NPUCommonExecution_hpp
#include "NPUBackend.hpp"
#include "core/Execution.hpp"

using namespace std;
namespace MNN {

class NPUCommonExecution : public Execution {
public:
    NPUCommonExecution(Backend *backend, const Op *op);
    virtual ~NPUCommonExecution() = default;

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    NPUBackend* mNpuBackend;
    const Op* mOp;
};

} // namespace MNN
#endif /* NPUCommonExecution_hpp */
