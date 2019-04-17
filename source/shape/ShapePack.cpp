//
//  ShapePack.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Macro.h"
#include "SizeComputer.hpp"

namespace MNN {

class PackComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op *op, const std::vector<Tensor *> &inputs,
                               const std::vector<Tensor *> &outputs) const override {
        auto pack      = op->main_as_PackParam();
        const int axis = pack->axis();

        if (inputs[0]->buffer().dimensions == 0) {
            MNN_ASSERT(axis == 0);
        }

        auto &input  = inputs[0]->buffer();
        auto &output = outputs[0]->buffer();

        output.dimensions = input.dimensions + 1;
        output.type       = input.type;

        for (int i = 0, j = 0; i < output.dimensions; i++) {
            if (i == axis) {
                output.dim[i].extent = (int)inputs.size();
            } else {
                output.dim[i].extent = input.dim[j].extent;
                j++;
            }
        }

        return true;
    }
};

REGISTER_SHAPE(PackComputer, OpType_Pack);
} // namespace MNN
