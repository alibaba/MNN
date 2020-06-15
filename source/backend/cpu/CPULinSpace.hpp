//
//  CPULinSpace.hpp
//  MNN
//
//  Created by MNN on 2019/12/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPULinSpace_hpp
#define CPULinSpace_hpp

#include "core/Execution.hpp"

namespace MNN {
class CPULinSpace : public Execution {
public:
    CPULinSpace(Backend *b) : Execution(b) {
        // nothing to do
    }
    virtual ~CPULinSpace() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};

} // namespace MNN

#endif /* CPULinSpace_hpp */
