//
//  ShapeSize.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Macro.h"
#include "SizeComputer.hpp"
#include "TensorUtils.hpp"

namespace MNN {

class SizeOpComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        // Scalar
        outputs[0]->buffer().dimensions = 0;

        outputs[0]->setType(DataType_DT_INT32);
        return true;
    }
};

REGISTER_SHAPE(SizeOpComputer, OpType_Size);
} // namespace MNN
