//
//  Arm82Pooling.hpp
//  MNN
//
//  Created by MNN on 2020/01/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Arm82Pooling_hpp
#define Arm82Pooling_hpp

#include "MNN_generated.h"
#include "backend/arm82/Arm82Backend.hpp"
#include "core/Execution.hpp"

namespace MNN {
class Arm82Pooling : public Execution {
public:
    Arm82Pooling(Backend *b, const Pool *parameter);
    virtual ~Arm82Pooling() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    int mThreadNumber;
    const Pool *mParameter;
    std::function<void(int tId, const FLOAT16 *src, FLOAT16 *dst)> mThreadFunction;
};

} // namespace MNN

#endif
