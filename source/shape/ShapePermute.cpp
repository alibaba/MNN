//
//  ShapePermute.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Macro.h"
#include "SizeComputer.hpp"

namespace MNN {
class PermuteComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(1 == inputs.size());
        MNN_ASSERT(1 == outputs.size());

        auto input  = inputs[0];
        auto output = outputs[0];
        auto shape  = op->main_as_Permute()->dims();
        MNN_ASSERT(shape->size() == input->buffer().dimensions);
        output->buffer().dimensions = shape->size();

        for (int i = 0; i < shape->size(); ++i) {
            output->buffer().dim[i].extent = input->buffer().dim[shape->data()[i]].extent;
        }

        return true;
    }
};

REGISTER_SHAPE(PermuteComputer, OpType_Permute);
} // namespace MNN
