//
//  ShapeBroadcastTo.cpp
//  MNN
//
//  Created by MNN on 2019/12/2.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "core/Macro.h"
#include "core/SizeComputer.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {

class ShapeBroadcastTo : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(inputs.size() == 2);
        MNN_ASSERT(outputs.size() == 1);

        auto input  = inputs[0];
        auto shape  = inputs[1];
        auto output = outputs[0];

        const int dimension = input->dimensions();
        MNN_CHECK(shape->elementSize() == dimension, "input dimension does not match given shape!");

        output->buffer().dimensions = dimension;
        const int* shapeData        = shape->host<int>();
        for (int i = 0; i < dimension; ++i) {
            const int dim = input->length(i);
            if (shapeData[i] != dim) {
                MNN_CHECK(dim == 1, "for each dimension pair they are either equal or one of them is one.");
            }
            output->setLength(i, shapeData[i]);
        }
        output->buffer().type                             = input->buffer().type;
        TensorUtils::getDescribe(output)->dimensionFormat = TensorUtils::getDescribe(input)->dimensionFormat;

        return true;
    }
};

REGISTER_SHAPE_INPUTS(ShapeBroadcastTo, OpType_BroadcastTo, {1});

} // namespace MNN
