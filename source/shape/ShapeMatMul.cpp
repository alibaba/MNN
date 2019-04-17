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

        auto output = outputs[0];
        TensorUtils::copyShape(inputs[0], output);

        output->buffer().dim[1].extent = inputs[1]->buffer().dim[inputs[1]->buffer().dimensions - 1].extent;

        return true;
    }
};

REGISTER_SHAPE(MatMulSizeComputer, OpType_MatMul);
} // namespace MNN
