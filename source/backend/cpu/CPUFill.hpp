//
//  CPUFill.hpp
//  MNN
//
//  Created by MNN on 2018/08/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUFill_hpp
#define CPUFill_hpp

#include "Execution.hpp"

namespace MNN {
class CPUFill : public Execution {
public:
    CPUFill(Backend *backend);
    virtual ~CPUFill() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};

} // namespace MNN

#endif /* CPUFill_hpp */
