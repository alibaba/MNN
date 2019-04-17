//
//  ShapeQuantizedConcat.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Macro.h"
#include "SizeComputer.hpp"

namespace MNN {

class QuantizedConcatComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        auto params    = op->main_as_QuantizedConcat();
        int axis       = params->axis();
        int num_inputs = (int)inputs.size();

        int input0_dim_size = inputs[0]->buffer().dimensions;

        if (axis < 0) {
            axis += input0_dim_size;
        }

        MNN_ASSERT(axis >= 0);
        MNN_ASSERT(axis < input0_dim_size && input0_dim_size <= 4);

        int sum_axis = inputs[0]->buffer().dim[axis].extent;

        for (int i = 1; i < num_inputs; ++i) {
            Tensor* t = inputs[i];
            MNN_ASSERT(t->buffer().dimensions == input0_dim_size);
            for (int d = 0; d < input0_dim_size; ++d) {
                if (d == axis) {
                    sum_axis += t->buffer().dim[axis].extent;
                } else {
                    MNN_ASSERT(t->buffer().dim[d].extent == inputs[0]->buffer().dim[d].extent);
                }
            }
        }

        int output_size[input0_dim_size];
        for (int d = 0; d < input0_dim_size; ++d) {
            output_size[d] = (d == axis) ? sum_axis : inputs[0]->buffer().dim[d].extent;
        }

        outputs[0]->buffer().dimensions = input0_dim_size;

        for (int i = 0; i < input0_dim_size; i++) {
            outputs[0]->buffer().dim[i].extent = output_size[i];
        }

        outputs[0]->setType(DataType_DT_UINT8);
        return true;
    }
};

REGISTER_SHAPE(QuantizedConcatComputer, OpType_QuantizedConcat);
} // namespace MNN
