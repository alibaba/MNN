//
//  CPUShape.hpp
//  MNN
//
//  Created by MNN on 2018/08/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUShape_hpp
#define CPUShape_hpp

#include "Execution.hpp"

namespace MNN {
class CPUShape : public Execution {
public:
    CPUShape(Backend *b) : Execution(b) {
        // nothing to do
    }
    virtual ~CPUShape() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};
} // namespace MNN

#endif /* CPUShape_hpp */
