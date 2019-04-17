//
//  ShapeArgMax.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Macro.h"
#include "SizeComputer.hpp"

namespace MNN {

// –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
// Size Computer
// –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
class ArgMaxComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op *op, const std::vector<Tensor *> &inputs,
                               const std::vector<Tensor *> &outputs) const override {
        MNN_ASSERT(1 == inputs.size());
        MNN_ASSERT(1 == outputs.size());

        // copy dims
        auto &input       = inputs[0]->buffer();
        auto &output      = outputs[0]->buffer();
        output.dimensions = input.dimensions;
        memcpy(output.dim, input.dim, sizeof(halide_dimension_t) * input.dimensions);

        // key extent
        auto argMax   = op->main_as_ArgMax();
        int keyExtent = argMax->topK();
        if (argMax->outMaxVal())
            keyExtent *= 2;

        if (input.dim[3].extent > 1) {
            output.dim[3].extent = keyExtent;
        } else if (input.dim[2].extent > 1) { // iw = ow = 1
            output.dim[2].extent = keyExtent;
        } else { // iw = ow = 1, ih = oh = 1;
            output.dim[1].extent = keyExtent;
        }

        return true;
    }
};

REGISTER_SHAPE(ArgMaxComputer, OpType_ArgMax);

} // namespace MNN
