//
//  Solution.hpp
//  MNN
//
//  Created by MNN on 2019/07/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ErrorCode.hpp"
#include "Expr.hpp"

namespace MNN {
namespace Express {
class Solution {
public:
    struct Requirement {
        std::vector<bool> contentNeedContent;
        std::vector<bool> shapeNeedContent;
        std::vector<bool> supportError;
    };

    Solution(int inputSize, int outputSize) : mInputSize(inputSize), mOutputSize(outputSize) {
        // Do nothing
    }
    virtual ~Solution() = default;
    virtual Requirement onGetRequirement() const;

    virtual ErrorCode onComputeInfo(const std::vector<const Variable::Info*>& inputs,
                                    const std::vector<Variable::Info*>& outputs) = 0;
    virtual ErrorCode onAlloc(const std::vector<const Variable::Info*>& inputs,
                              const std::vector<Variable::Info*>& outputs)       = 0;
    virtual ErrorCode onComputeContent(const std::vector<const Variable::Info*>& inputs,
    const std::vector<Variable::Info*>& outputs)                                         = 0;

    // Map output's content to host
    virtual void* onMapContent(int index)  = 0;
    virtual void onUnMapContent(int index) = 0;

protected:
    const int mInputSize;
    const int mOutputSize;
};
class Executor {
public:
    Executor()                                                              = default;
    virtual ~Executor()                                                     = default;
    virtual Solution* onCreate(const Op* op, int inputSize, int outputSize) = 0;
};
} // namespace Express
} // namespace MNN
