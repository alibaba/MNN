//
//  CPUReluGrad.hpp
//  MNN
//
//  Created by MNN on 2019/04/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUReluGrad_hpp
#define CPUReluGrad_hpp

#include "CPUBackend.hpp"
namespace MNN {
class CPUReluGrad : public Execution {
public:
    CPUReluGrad(float slope, Backend *bn) : Execution(bn), mSlope(slope) {
    }
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    float mSlope = 0.0f;
};
} // namespace MNN

#endif /* CPUReluGrad_hpp */
