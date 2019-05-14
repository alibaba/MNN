//
//  ShapeInnerProduct.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Macro.h"
#include "SizeComputer.hpp"

namespace MNN {
class InnerProductComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(1 == inputs.size());
        MNN_ASSERT(1 == outputs.size());

        auto output    = outputs[0];
        auto input     = inputs[0];
        auto parameter = op->main_as_InnerProduct();

        MNN_ASSERT(2 == input->buffer().dimensions);
        output->buffer().dimensions    = input->buffer().dimensions;
        output->buffer().dim[0].extent = input->buffer().dim[0].extent;
        output->buffer().dim[0].flags  = 0;
        output->buffer().dim[1].extent = parameter->outputCount();
        output->buffer().dim[1].flags  = 0;

        return true;
    }
};

REGISTER_SHAPE(InnerProductComputer, OpType_InnerProduct);
} // namespace MNN
