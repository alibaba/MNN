//
//  ShapeHistogram.cpp
//  MNN
//
//  Created by MNN on 2022/07/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"
#include "math.h"

namespace MNN {

class ShapeHistogram : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(inputs.size() == 1 && outputs.size() == 1);
        
        auto output = outputs[0];
        output->buffer().dimensions = 1;
        output->setLength(0, op->main_as_ArgMax()->outMaxVal());
        output->buffer().type = halide_type_of<float>();
        TensorUtils::getDescribe(output)->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        return true;
    }
};

REGISTER_SHAPE(ShapeHistogram, OpType_Histogram);
} // namespace MNN
