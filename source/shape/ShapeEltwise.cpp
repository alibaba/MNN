//
//  ShapeEltwise.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "core/Macro.h"
#include "core/SizeComputer.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {
// Size Computer
class EltWiseComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(2 <= inputs.size());
        MNN_ASSERT(1 == outputs.size());
        TensorUtils::copyShape(inputs[0], outputs[0], true);
        outputs[0]->buffer().type = inputs[0]->getType();
        return true;
    }
    virtual float onComputeFlops(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                                 const std::vector<Tensor*>& outputs) const override {
        auto size = (float)outputs[0]->elementSize() / 1024.0f / 1024.0f;
        return size * (inputs.size() - 1);
    }
};

REGISTER_SHAPE(EltWiseComputer, OpType_Eltwise);
REGISTER_SHAPE(EltWiseComputer, OpType_SpatialProduct);
} // namespace MNN
