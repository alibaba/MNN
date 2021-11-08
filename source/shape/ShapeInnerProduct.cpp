//
//  ShapeInnerProduct.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"

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
        output->buffer().dim[1].extent = parameter->outputCount();
        output->buffer().type = halide_type_of<float>();
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;

        return true;
    }
};

REGISTER_SHAPE(InnerProductComputer, OpType_InnerProduct);
} // namespace MNN
