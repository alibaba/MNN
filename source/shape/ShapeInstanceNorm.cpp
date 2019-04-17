//
//  ShapeInstanceNorm.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNNDefine.h"
#include "SizeComputer.hpp"
#include "TensorUtils.hpp"

namespace MNN {
class InstanceNormComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(3 == inputs.size());
        MNN_ASSERT(1 == outputs.size());

        TensorUtils::copyShape(inputs[0], outputs[0]);

        return true;
    }
};

REGISTER_SHAPE(InstanceNormComputer, OpType_BatchNorm);

} // namespace MNN
