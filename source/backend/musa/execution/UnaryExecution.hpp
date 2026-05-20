//
//  UnaryExecution.hpp
//  MNN
//
//  Created by MNN on 2026/02/25.
//  Copyright © 2026, Alibaba Group Holding Limited
//

#ifndef UnaryExecution_hpp
#define UnaryExecution_hpp

#include "core/Execution.hpp"

#include <vector>
#include "backend/musa/core/MusaBackend.hpp"

namespace MNN {
namespace MUSA {

class UnaryExecution : public Execution {
public:
    UnaryExecution(UnaryOpOperation opType, Backend *backend);
    virtual ~UnaryExecution() = default;

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    MusaRuntime *mRuntime;
    UnaryOpOperation mOpType;
    int mCount;
};

} // namespace MUSA
} // namespace MNN
#endif /* UnaryExecution_hpp */
