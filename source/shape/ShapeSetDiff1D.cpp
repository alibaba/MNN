//
//  ShapeSetDiff1D.cpp
//  MNN
//
//  Created by MNN on 2021/06/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"

namespace MNN {

class ShapeSetDiff1D : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(2 == inputs.size());
        MNN_ASSERT(1 == outputs.size());
        if (inputs[0]->getType().code != halide_type_int || inputs[1]->getType().code != halide_type_int) {
            return false;
        }
        auto& ib = inputs[0]->buffer();
        auto& ob = outputs[0]->buffer();
        ob.dimensions = 1;
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        ob.type = ib.type;
        const int32_t* inputData = inputs[0]->host<int32_t>();
        const int32_t* removeData = inputs[1]->host<int32_t>();
        if (!inputData || !removeData) {
            return false;
        }
        int outputNum = 0;
        for (int i = 0; i < inputs[0]->elementSize(); i++) {
            bool remove = false;
            for (int j = 0; j < inputs[1]->elementSize(); j++) {
                if (inputData[i] == removeData[j]) {
                    remove = true;
                    break;
                }
            }
            if (!remove) {
                outputNum++;
            }
        }
        ob.dim[0].extent = outputNum;
        return true;
    }
};

REGISTER_SHAPE_INPUTS(ShapeSetDiff1D, OpType_SetDiff1D, std::vector<int>({0, 1}));
} // namespace MNN
