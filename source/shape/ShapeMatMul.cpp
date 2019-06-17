//
//  ShapeMatMul.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Macro.h"
#include "SizeComputer.hpp"
#include "TensorUtils.hpp"

namespace MNN {

class MatMulSizeComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(2 == inputs.size());
        MNN_ASSERT(1 == outputs.size());
        MNN_ASSERT(2 == inputs[0]->buffer().dimensions);
        MNN_ASSERT(2 == inputs[1]->buffer().dimensions);
        MNN_ASSERT(op->main_type() == OpParameter_MatMul);
        auto matMul = op->main_as_MatMul();

        auto output = outputs[0];
        TensorUtils::copyShape(inputs[0], output, true);
        auto w0 = inputs[0]->length(1);
        auto h0 = inputs[0]->length(0);

        if (matMul->transposeA()) {
            auto t = w0;
            w0     = h0;
            h0     = t;
        }

        auto w1 = inputs[1]->length(1);
        auto h1 = inputs[1]->length(0);
        if (matMul->transposeB()) {
            auto t = w1;
            w1     = h1;
            h1     = t;
        }

        if (w0 != h1) {
            return false;
        }
        output->buffer().type = inputs[0]->buffer().type;
        output->setLength(0, h0);
        output->setLength(1, w1);
        TensorUtils::setLinearLayout(output);

        return true;
    }
};

REGISTER_SHAPE(MatMulSizeComputer, OpType_MatMul);
} // namespace MNN
