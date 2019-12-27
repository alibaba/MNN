//
//  CPUOneHot.hpp
//  MNN
//
//  Created by MNN on 2019/11/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUOneHot_hpp
#define CPUOneHot_hpp

#include "core/Execution.hpp"

namespace MNN {

class CPUOneHot : public Execution{
public:
    CPUOneHot(Backend* b, int axis):Execution(b), mAxis(axis){}
    virtual ~CPUOneHot() = default;

    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    int mAxis;
};

} // namespace MNN

#endif /* CPUOneHot_hpp */
