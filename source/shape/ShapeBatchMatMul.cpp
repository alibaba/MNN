//
//  ShapeBatchMatMul.cpp
//  MNN
//
//  Created by MNN on 2019/03/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "core/Macro.h"
#include "core/SizeComputer.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {

class BatchMatMulComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(2 == inputs.size());
        MNN_ASSERT(1 == outputs.size());

        auto param = op->main_as_BatchMatMulParam();
        auto input0 = inputs[0];
        auto input1 = inputs[1];
        MNN_ASSERT(input0->dimensions() == input1->dimensions());

        const int dimensions = input0->dimensions();
        MNN_ASSERT(dimensions >= 2);
        for (int i = 0; i < dimensions - 2; ++i) {
            MNN_ASSERT(input0->length(i) == input1->length(i));
        }

        auto output = outputs[0];
        output->buffer().type = input0->buffer().type;
        TensorUtils::copyShape(input0, output, true);
        auto k0 = input0->length(dimensions - 1);
        auto k1 = input1->length(dimensions - 2);
        if (param->adjX()) {
            k0 = input0->length(dimensions - 2);
            output->setLength(dimensions - 2, input0->length(dimensions - 1));
        } else {
            output->setLength(dimensions - 2, input0->length(dimensions - 2));
        }
        if (param->adjY()) {
            k1 = input1->length(dimensions - 1);
            output->setLength(dimensions - 1, input1->length(dimensions - 2));
        } else {
            output->setLength(dimensions - 1, input1->length(dimensions - 1));
        }
        if (k0 != k1) {
            return false;
        }

        return true;
    }
};

REGISTER_SHAPE(BatchMatMulComputer, OpType_BatchMatMul);

} // namespace MNN
