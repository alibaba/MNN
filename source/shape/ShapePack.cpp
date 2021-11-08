//
//  ShapePack.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"

namespace MNN {

class PackComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op *op, const std::vector<Tensor *> &inputs,
                               const std::vector<Tensor *> &outputs) const override {
        auto &input  = inputs[0]->buffer();
        auto &output = outputs[0]->buffer();
        output.dimensions = input.dimensions + 1;
        output.type       = input.type;

        auto pack      = op->main_as_PackParam();
        int axis = pack->axis();
        if (axis < 0) {
            axis += outputs[0]->dimensions();
        }

        if (inputs[0]->buffer().dimensions == 0) {
            MNN_ASSERT(axis == 0);
        }

        for (int i = 0, j = 0; i < output.dimensions; i++) {
            if (i == axis) {
                output.dim[i].extent = (int)inputs.size();
            } else {
                output.dim[i].extent = input.dim[j].extent;
                j++;
            }
        }
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;

        return true;
    }
};

REGISTER_SHAPE(PackComputer, OpType_Pack);
} // namespace MNN
