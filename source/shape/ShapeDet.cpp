//
//  ShapeDet.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"

namespace MNN {
class DetComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        if (inputs.size() != 1) {
            MNN_ERROR("Det only accept 1 input\n");
            return false;
        }
        auto shape = inputs[0]->shape();
        int dim = shape.size();
        if (dim < 2 || shape[dim - 1] != shape[dim - 2]) {
            MNN_ERROR("input must be [*, M, M]\n");
            return false;
        }
        auto& ib = inputs[0]->buffer();
        auto& ob = outputs[0]->buffer();

        ob.dimensions = dim - 2;
        if (dim > 2) {
            ::memcpy(ob.dim, ib.dim, ob.dimensions * sizeof(halide_dimension_t));
        }
        ob.type = ib.type;
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;

        return true;
    }
};

REGISTER_SHAPE(DetComputer, OpType_Det);

} // namespace MNN
