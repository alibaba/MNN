//
//  ShapeUnravelIndex.cpp
//  MNN
//
//  Created by MNN on 2019/11/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"

namespace MNN {

class UnravelIndexSize : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op *op, const std::vector<Tensor *> &inputs,
                               const std::vector<Tensor *> &outputs) const override {
        MNN_ASSERT(2 == inputs.size());

        auto indices = inputs[0];
        auto dims    = inputs[1];
        auto output  = outputs[0];
        MNN_CHECK(dims->dimensions() == 1, "dims should be one dimension tensor!");

        const int inputDimension = indices->dimensions();
        output->setType(DataType_DT_INT32);
        if (inputDimension == 0) {
            output->buffer().dimensions = 1;
            output->setLength(0, dims->length(0));
        } else {
            output->buffer().dimensions = 2;
            output->setLength(0, dims->length(0));
            output->setLength(1, indices->elementSize());
        }

        return true;
    }
};

REGISTER_SHAPE(UnravelIndexSize, OpType_UnravelIndex);

} // namespace MNN
