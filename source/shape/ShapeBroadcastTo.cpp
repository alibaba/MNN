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
        output->buffer().dimensions = shape->elementSize();
        const int dimension = output->dimensions();
        const int* shapeData        = shape->host<int>();
        for (int i = 0; i < dimension; ++i) {
            output->setLength(i, shapeData[i]);
        }
        output->buffer().type                             = input->buffer().type;
        TensorUtils::getDescribe(output)->dimensionFormat = TensorUtils::getDescribe(input)->dimensionFormat;
        if (output->dimensions() != input->dimensions()) {
            if (output->elementSize() != input->elementSize()) {
                MNN_ERROR("Don't support dimension not the same and size not the same for BroadcastTo\n");
                return false;
            }
        }
        return true;
    }
};

REGISTER_SHAPE_INPUTS(ShapeBroadcastTo, OpType_BroadcastTo, {1});

} // namespace MNN
