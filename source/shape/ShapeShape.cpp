//
//  ShapeShape.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Macro.h"
#include "SizeComputer.hpp"
#include "TensorUtils.hpp"

namespace MNN {

class ShapeSizeComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(1 == inputs.size());
        MNN_ASSERT(1 == outputs.size());
        auto& ib = inputs[0]->buffer();
        auto& ob = outputs[0]->buffer();
        for (int i = 0; i < ib.dimensions; i++) {
            if (ib.dim[i].extent <= 0) {
                return false;
            }
        }
        ob.dimensions = 1;
        outputs[0]->setType(DataType_DT_INT32);
        if (TensorUtils::getDescribe(inputs[0])->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
            ob.dim[0].extent = 4;
        } else {
            ob.dim[0].extent = ib.dimensions;
        }
        return true;
    }
};

REGISTER_SHAPE(ShapeSizeComputer, OpType_Shape);
} // namespace MNN
