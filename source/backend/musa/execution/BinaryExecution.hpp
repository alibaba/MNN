//
//  BinaryExecution.hpp
//  MNN
//
//  Created by MNN on 2026/02/25.
//  Copyright © 2026, Alibaba Group Holding Limited
//

#ifndef BinaryExecution_hpp
#define BinaryExecution_hpp

#include "core/Execution.hpp"
#include "backend/musa/core/MusaBackend.hpp"

namespace MNN {
namespace MUSA {

class BinaryExecution : public Execution {
public:
    BinaryExecution(BinaryOpOperation opType, Backend *backend);
    virtual ~BinaryExecution() = default;

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    MusaRuntime *mRuntime;
    BinaryOpOperation mOpType;
    int mCount;
};

} // namespace MUSA
} // namespace MNN
#endif /* BinaryExecution_hpp */
