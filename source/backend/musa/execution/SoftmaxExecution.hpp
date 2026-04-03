//
//  SoftmaxExecution.hpp
//  MNN
//
//  Created by MNN on 2026/02/25.
//  Copyright © 2026, Alibaba Group Holding Limited
//

#ifndef SoftmaxExecution_hpp
#define SoftmaxExecution_hpp

#include "core/Execution.hpp"
#include "backend/musa/core/MusaBackend.hpp"

namespace MNN {
namespace MUSA {

class SoftmaxExecution : public Execution {
public:
    SoftmaxExecution(int axis, Backend *backend);
    virtual ~SoftmaxExecution() = default;

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    MusaRuntime *mRuntime;
    int mAxis;
    int mOuterCount;
    int mInnerCount;
    int mDepth;
};

} // namespace MUSA
} // namespace MNN
#endif /* SoftmaxExecution_hpp */
