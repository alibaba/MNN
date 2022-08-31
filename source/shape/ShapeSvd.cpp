//
//  ShapeSvd.cpp
//  MNN
//
//  Created by MNN on 2022/07/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"
#include "math.h"

namespace MNN {

class SvdComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(inputs.size() == 1 && outputs.size() == 3);
        auto shape = inputs[0]->shape();
        MNN_ASSERT(shape.size() == 2);
        int row = shape[0];
        int col  = shape[1];
        // int single_num = std::min(row, col);
        int single_num = col;
        // w is [ single_num ]
        outputs[0]->buffer().dimensions = 1;
        outputs[0]->setLength(0, single_num);
        // u is [row, single_num ]
        outputs[1]->buffer().dimensions = 2;
        outputs[1]->setLength(0, row);
        outputs[1]->setLength(1, single_num);
        // vt is [single_num, col ]
        outputs[2]->buffer().dimensions = 2;
        outputs[2]->setLength(0, single_num);
        outputs[2]->setLength(1, col);
        for (int i = 0; i < 3; i++) {
            outputs[i]->buffer().type = inputs[0]->getType();
            TensorUtils::getDescribe(outputs[i])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        }
        return true;
    }
};

REGISTER_SHAPE(SvdComputer, OpType_Svd);
} // namespace MNN
