//
//  ShapeFill.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {

class FillComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        auto input0 = inputs[0], output0 = outputs[0];
        const int* ptr = input0->host<int32_t>();
        MNN_ASSERT(inputs.size() == 2);
        MNN_ASSERT(input0->buffer().dimensions == 1);
        output0->buffer().dimensions = input0->buffer().dim[0].extent;
        output0->buffer().type = inputs[1]->buffer().type;
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[1])->dimensionFormat;
        for (int i = 0; i < input0->buffer().dim[0].extent; i++) {
            output0->buffer().dim[i].extent = input0->host<int32_t>()[i];
        }
        return true;
    }
};

REGISTER_SHAPE_INPUTS(FillComputer, OpType_Fill, {0});
} // namespace MNN
