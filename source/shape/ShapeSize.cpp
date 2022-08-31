//
//  ShapeSize.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {

class SizeOpComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        // Scalar
        outputs[0]->buffer().dimensions = 0;

        outputs[0]->setType(DataType_DT_INT32);
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = op->defaultDimentionFormat();

        return true;
    }
};

REGISTER_SHAPE(SizeOpComputer, OpType_Size);
REGISTER_SHAPE(SizeOpComputer, OpType_Rank);
} // namespace MNN
