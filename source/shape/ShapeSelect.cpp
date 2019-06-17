//
//  ShapeSelect.cpp
//  MNN
//
//  Created by MNN on 2019/5/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Macro.h"
#include "SizeComputer.hpp"
#include "TensorUtils.hpp"
namespace MNN {
class SelectSizeComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(3 == inputs.size());
        MNN_ASSERT(1 == outputs.size());
        const auto& ib = inputs[0]->buffer();
        auto& ob       = outputs[0]->buffer();
        memcpy(ob.dim, ib.dim, sizeof(halide_dimension_t) * ib.dimensions);
        ob.dimensions = ib.dimensions;
        ob.type       = inputs[1]->buffer().type;
        TensorUtils::getDescribe(outputs[0])->dimensionFormat =  TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        return true;
    }
};

REGISTER_SHAPE(SelectSizeComputer, OpType_Select);
} // namespace MNN
