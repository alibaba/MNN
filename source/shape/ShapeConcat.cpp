//
//  ShapeConcat.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Macro.h"
#include "SizeComputer.hpp"
#include "TensorUtils.hpp"

namespace MNN {
class ConcatSizeComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(1 == outputs.size());
        MNN_ASSERT(inputs.size() >= 2);
        auto& ob      = outputs[0]->buffer();
        int basicAxis = 0;
        if (op->type() == OpType_Concat) {
            basicAxis = op->main_as_Axis()->axis();
        } else if (op->type() == OpType_QuantizedConcat) {
            basicAxis = op->main_as_QuantizedConcat()->axis();
        }

        int axis = basicAxis;
        // Concat-inputs may have scalar which should be delete
        for (const auto& input : inputs) {
            if (0 == input->buffer().dimensions) {
                continue;
            } else {
                auto inputDimensions = input->buffer().dimensions;
                ::memcpy(ob.dim, input->buffer().dim, sizeof(halide_dimension_t) * inputDimensions);
                ob.dimensions = inputDimensions;
                ob.type       = input->buffer().type;
                if (axis < 0) {
                    axis = inputDimensions + axis;
                }
                break;
            }
        }

        int sum = 0;
        for (auto t : inputs) {
            sum += t->buffer().dim[axis].extent;
            for (int i = 0; i < t->dimensions(); ++i) {
                if (axis == i) {
                    continue;
                }
                if (t->length(i) != outputs[0]->length(i)) {
                    MNN_PRINT("Error for concat size of op %s, %d input not match output\n", op->name()->c_str(), i);
                    return false;
                }
            }
        }
        ob.dim[axis].extent                                   = sum;
        ob.type                                               = inputs[0]->buffer().type;
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;

        return true;
    }
};

REGISTER_SHAPE(ConcatSizeComputer, OpType_Concat);
REGISTER_SHAPE(ConcatSizeComputer, OpType_QuantizedConcat);
} // namespace MNN
