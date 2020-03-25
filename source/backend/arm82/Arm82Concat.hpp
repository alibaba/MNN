//
//  Arm82Concat.hpp
//  MNN
//
//  Created by MNN on 2020/01/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Arm82Concat_hpp
#define Arm82Concat_hpp

#include "core/Execution.hpp"
#include "backend/arm82/Arm82Backend.hpp"

namespace MNN {
class Arm82Concat : public Execution {
public:
    Arm82Concat(Backend *b, int axis) : Execution(b), mAxis(axis) {
        // Do nothing
    }
    virtual ~Arm82Concat() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    int mAxis = 1;
    std::shared_ptr<Tensor> mTempOutput;
    bool mUseSlowMethod = false;
};

} // namespace MNN

#endif /* Arm82Concat_hpp */
