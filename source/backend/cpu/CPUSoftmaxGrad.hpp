//
//  CPUSoftmaxGrad.hpp
//  MNN
//
//  Created by MNN on 2019/04/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUSoftmaxGrad_hpp
#define CPUSoftmaxGrad_hpp

#include "CPUBackend.hpp"

namespace MNN {
class CPUSoftmaxGrad : public Execution {
public:
    CPUSoftmaxGrad(int axis, Backend *bn) : Execution(bn), mAxis(axis) {
    }
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    int mAxis = 1;
};
} // namespace MNN

#endif /* CPUSoftmaxGrad_hpp */
