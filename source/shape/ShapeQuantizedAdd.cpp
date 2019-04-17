//
//  ShapeQuantizedAdd.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Macro.h"
#include "SizeComputer.hpp"

namespace MNN {

class QuantizedAddComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op *op, const std::vector<Tensor *> &inputs,
                               const std::vector<Tensor *> &outputs) const override {
        MNN_ASSERT(2 == inputs[0]->buffer().dimensions || 4 == inputs[0]->buffer().dimensions);
        // copy dims
        auto &input  = inputs[0]->buffer();
        auto &output = outputs[0]->buffer();
        ::memcpy(output.dim, input.dim, sizeof(halide_dimension_t) * input.dimensions);
        output.dimensions = input.dimensions;
        outputs[0]->setType(DataType_DT_UINT8);
        return true;
    }
};

REGISTER_SHAPE(QuantizedAddComputer, OpType_QuantizedAdd);
} // namespace MNN
