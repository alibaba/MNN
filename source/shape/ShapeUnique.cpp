//
//  ShapeSetDiff1D.cpp
//  MNN
//
//  Created by MNN on 2021/06/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <unordered_set>
#include "shape/SizeComputer.hpp"
#include "core/Macro.h"

namespace MNN {

class ShapeUnique : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(1 == inputs.size());
        if (inputs[0]->getType().code != halide_type_int) {
            return false;
        }
        auto& ib = inputs[0]->buffer();
        auto& ob = outputs[0]->buffer();
        ob.dimensions = 1;
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        ob.type = ib.type;
        if (inputs[0]->host<int32_t>() == nullptr) {
            return false;
        }
        std::unordered_set<int> values;
        auto eleSize = inputs[0]->elementSize();
        for (int i = 0; i < eleSize; i++) {
            values.insert(inputs[0]->host<int32_t>()[i]);
        }
        ob.dim[0].extent = values.size();
        if (outputs.size() > 1) {
            TensorUtils::copyShape(outputs[0], outputs[1], true);
            outputs[1]->buffer().type = halide_type_of<int>();
        }
        return true;
    }
};

REGISTER_SHAPE_INPUTS(ShapeUnique, OpType_Unique, std::vector<int>({0}));
} // namespace MNN
