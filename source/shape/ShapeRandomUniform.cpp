//
//  ShapeRandomUniform.cpp
//  MNN
//
//  Created by MNN on 2020/8/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "core/Macro.h"
#include "shape/SizeComputer.hpp"

namespace MNN {

class RandomUniformComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op *op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(1 == outputs.size());
        auto param = op->main_as_RandomUniform();
        outputs[0]->setType(param->type());
        auto &output = outputs[0]->buffer();
        auto shapePtr = inputs[0]->host<int>();
        output.dimensions = inputs[0]->elementSize();
        for (int i = 0; i < outputs[0]->dimensions(); i++) {
            output.dim[i].extent = shapePtr[i];
        }
        TensorUtils::setLinearLayout(outputs[0]);
        return true;
    }
};

REGISTER_SHAPE_INPUTS(RandomUniformComputer, OpType_RandomUniform, {0});
REGISTER_SHAPE_INPUTS(RandomUniformComputer, OpType_RandomNormal, {0});

} // namespace MNN
