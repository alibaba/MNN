//
//  ShapeShape.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {

class ShapeSizeComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(1 <= inputs.size());
        MNN_ASSERT(1 == outputs.size());
        auto& ib = inputs[0]->buffer();
        auto& ob = outputs[0]->buffer();

        ob.dimensions = 1;
        outputs[0]->setType(DataType_DT_INT32);
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = op->defaultDimentionFormat();
        auto inputFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        if (inputFormat == MNN_DATA_FORMAT_NC4HW4 && op->defaultDimentionFormat() == MNN_DATA_FORMAT_NHWC) {
            // For compability
            ob.dim[0].extent = 4;
        } else {
            ob.dim[0].extent = ib.dimensions;
        }
        return true;
    }
};

REGISTER_SHAPE(ShapeSizeComputer, OpType_Shape);
} // namespace MNN
