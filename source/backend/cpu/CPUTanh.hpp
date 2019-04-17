//
//  CPUTanh.hpp
//  MNN
//
//  Created by MNN on 2018/08/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUTanh_hpp
#define CPUTanh_hpp

#include "Execution.hpp"

namespace MNN {
class CPUTanh : public Execution {
public:
    CPUTanh(Backend *b) : Execution(b) {
        // nothing to do
    }
    virtual ~CPUTanh() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};
} // namespace MNN

#endif // CPUTanh_hpp
