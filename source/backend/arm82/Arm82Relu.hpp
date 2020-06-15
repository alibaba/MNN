//
//  Arm82Relu.hpp
//  MNN
//
//  Created by MNN on 2020/2/13.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Arm82Relu_hpp
#define Arm82Relu_hpp

#include "core/Execution.hpp"

namespace MNN {

class Arm82Relu : public Execution {
public:
    Arm82Relu(Backend *backend, const Op *op);
    virtual ~Arm82Relu() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};

} // namespace MNN

#endif /* Arm82Relu_hpp */
