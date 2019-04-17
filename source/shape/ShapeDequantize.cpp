//
//  ShapeDequantize.cpp
//  MNN
//
//  Created by MNN on 2019/01/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Macro.h"
#include "SizeComputer.hpp"

namespace MNN {
class ShapeDequantize : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op *op, const std::vector<Tensor *> &inputs,
                               const std::vector<Tensor *> &outputs) const override {
        MNN_ASSERT(3 == inputs.size() || 1 == inputs.size());
        MNN_ASSERT(1 == outputs.size());

        // copy dims
        auto &input  = inputs[0]->buffer();
        auto &output = outputs[0]->buffer();
        memcpy(output.dim, input.dim, sizeof(halide_dimension_t) * input.dimensions);
        output.dimensions = input.dimensions;

        return true;
    }
};
REGISTER_SHAPE(ShapeDequantize, OpType_Dequantize);
} // namespace MNN
