//
//  CPUWhere.hpp
//  MNN
//
//  Created by MNN on 2018/08/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUWhere_hpp
#define CPUWhere_hpp

#include "core/Execution.hpp"

namespace MNN {
class CPUWhere : public Execution {
public:
    CPUWhere(Backend *b) : Execution(b) {
        // nothing to do
    }
    virtual ~CPUWhere() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};
} // namespace MNN

#endif /* CPUWhere_hpp */
