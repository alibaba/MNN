//
//  CPUSigmoid.hpp
//  MNN
//
//  Created by MNN on 2018/08/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUSigmoid_hpp
#define CPUSigmoid_hpp

#include "Execution.hpp"

namespace MNN {
class CPUSigmoid : public Execution {
public:
    CPUSigmoid(Backend *b) : Execution(b) {
        // nothing to do
    }
    virtual ~CPUSigmoid() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};
} // namespace MNN

#endif /* CPUSigmoid_hpp */
