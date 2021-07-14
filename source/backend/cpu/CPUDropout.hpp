//
//  CPURelu.hpp
//  MNN
//
//  Created by MNN on 2018/07/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPURelu_hpp
#define CPURelu_hpp

#include "core/AutoStorage.h"
#include "core/Execution.hpp"

namespace MNN {
class CPUDropout : public Execution {
public:
    CPUDropout(Backend *b) : Execution(b) {
        // nothing to do
    }
    virtual ~CPUDropout() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    float mDropRatio;
};


} // namespace MNN

#endif /* CPURelu_hpp */
