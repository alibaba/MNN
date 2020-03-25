//
//  Arm82Eltwise.hpp
//  MNN
//
//  Created by MNN on 2020/2/13.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Arm82Eltwise_hpp
#define Arm82Eltwise_hpp

#include "core/Execution.hpp"
#include "MNN_generated.h"

namespace MNN {

class Arm82Eltwise : public Execution {
public:
    Arm82Eltwise(Backend *backend, EltwiseType type);
    virtual ~Arm82Eltwise() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    
private:
    EltwiseType mType;
};

} // namespace MNN

#endif /* Arm82Eltwise_hpp */
