//
//  ShapeLSTM.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Macro.h"
#include "SizeComputer.hpp"
#include "TensorUtils.hpp"

namespace MNN {

// –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
// Size Computer
// –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
class LSTMComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op *op, const std::vector<Tensor *> &inputs,
                               const std::vector<Tensor *> &outputs) const override {
        MNN_ASSERT(2 >= inputs.size());
        MNN_ASSERT(1 == outputs.size());

        // copy dims
        auto &input  = inputs[0]->buffer();
        auto &output = outputs[0]->buffer();
        memcpy(output.dim, input.dim, sizeof(halide_dimension_t) * input.dimensions);

        auto LSTM            = op->main_as_LSTM();
        output.dim[3].extent = LSTM->outputCount();
        output.dim[2].extent = 1;

        return true;
    }
};

REGISTER_SHAPE(LSTMComputer, OpType_LSTM);
} // namespace MNN
