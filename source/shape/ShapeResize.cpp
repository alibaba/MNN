//
//  ShapeResize.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "core/Macro.h"
#include "core/SizeComputer.hpp"

namespace MNN {
// Size Computer
class ResizeComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op *op, const std::vector<Tensor *> &inputs,
                               const std::vector<Tensor *> &outputs) const override {
        MNN_ASSERT(1 == inputs.size());
        MNN_ASSERT(1 == outputs.size());

        // copy dims
        auto resize  = op->main_as_Resize();
        auto &input  = inputs[0]->buffer();
        auto &output = outputs[0]->buffer();
        TensorUtils::copyShape(inputs[0], outputs[0], true);

        // set dims
        output.dim[3].extent = input.dim[3].extent * resize->xScale();
        output.dim[2].extent = input.dim[2].extent * resize->yScale();
        output.type = inputs[0]->getType();

        return true;
    }
    virtual float onComputeFlops(const MNN::Op *op, const std::vector<Tensor *> &inputs,
                                 const std::vector<Tensor *> &outputs) const override {
        return (float)outputs[0]->elementSize() / 1024.0f / 1024.0f * 4;
    }
};

REGISTER_SHAPE(ResizeComputer, OpType_Resize);
} // namespace MNN
