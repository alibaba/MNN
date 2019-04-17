//
//  CPUSize.hpp
//  MNN
//
//  Created by MNN on 2018/08/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUSize_hpp
#define CPUSize_hpp

#include "Execution.hpp"

namespace MNN {
template <typename T>
class CPUSize : public Execution {
public:
    CPUSize(Backend *backend, const Op *op);
    virtual ~CPUSize() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};

} // namespace MNN
#endif /* CPUSize_hpp */
