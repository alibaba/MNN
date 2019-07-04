//
//  ShapeWhere.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Macro.h"
#include "SizeComputer.hpp"

namespace MNN {

class WhereSizeComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(1 == inputs.size());
        MNN_ASSERT(1 == outputs.size());
        auto& ib = inputs[0]->buffer();
        auto& ob = outputs[0]->buffer();
        MNN_ASSERT(ib.type.code == halide_type_int);
        ob.dimensions = 2;
        // Assume all elements are true
        ob.dim[0].extent = inputs[0]->elementSize();
        ob.dim[1].extent = ib.dimensions;
        outputs[0]->buffer().type = halide_type_of<int32_t>();
        return true;
    }
};

REGISTER_SHAPE(WhereSizeComputer, OpType_Where);
} // namespace MNN
