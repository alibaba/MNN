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
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(1 == outputs.size());
        auto input = inputs[0];
        auto output    = outputs[0];
        auto parameter = op->main_as_RandomUniform();
        output->setType(parameter->type());
        output->buffer().dimensions = input->elementSize();
        for (int i = 0; i < input->elementSize(); i++) {
            output->buffer().dim[i].extent = input->host<int>()[i];
        }
        return true;
    }
};

REGISTER_SHAPE_INPUTS(RandomUniformComputer, OpType_RandomUniform, {0});

} // namespace MNN
