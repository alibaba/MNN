//
//  CPURange.hpp
//  MNN
//
//  Created by MNN on 2018/08/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPURange_hpp
#define CPURange_hpp

#include "core/Execution.hpp"

namespace MNN {
template <typename T>
class CPURange : public Execution {
public:
    CPURange(Backend *backend);
    virtual ~CPURange() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};

} // namespace MNN

#endif /* CPURange.hpp */
