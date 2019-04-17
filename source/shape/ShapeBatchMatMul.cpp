//
//  ShapeBatchMatMul.cpp
//  MNN
//
//  Created by MNN on 2019/03/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Macro.h"
#include "SizeComputer.hpp"
#include "TensorUtils.hpp"

namespace MNN {

class BatchMatMulComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(2 == inputs.size());
        MNN_ASSERT(1 == outputs.size());

        auto param = op->main_as_BatchMatMulParam();
        // [TODO] now only support: adj_x and adj_y are false!
        MNN_ASSERT(!param->adjX());
        MNN_ASSERT(!param->adjY());

        auto input0 = inputs[0];
        auto input1 = inputs[1];
        MNN_ASSERT(input0->dimensions() == input1->dimensions());

        const int dimensions = input0->dimensions();
        MNN_ASSERT(dimensions >= 2);
        for (int i = 0; i < dimensions - 2; ++i) {
            MNN_ASSERT(input0->length(i) == input1->length(i));
        }

        const int input0LastDimSize       = input0->length(dimensions - 1);
        const int input1LastDimSize       = input1->length(dimensions - 1);
        const int input1LastSecondDimSize = input1->length(dimensions - 2);
        MNN_ASSERT(input0LastDimSize == input1LastSecondDimSize);

        auto output = outputs[0];
        TensorUtils::copyShape(input0, output, true);
        output->setLength(dimensions - 1, input1LastDimSize);

        return true;
    }
};

REGISTER_SHAPE(BatchMatMulComputer, OpType_BatchMatMul);

} // namespace MNN
