//
//  ShapeConcat.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"

namespace MNN {
class ConcatSizeComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(1 == outputs.size());
        if (inputs.empty()) {
            MNN_ERROR("Concat op has no input\n");
            return false;
        }
        auto& ob      = outputs[0]->buffer();
        int basicAxis = 0;
        if (op->type() == OpType_Concat) {
            if (op->main_as_Axis() != nullptr) {
                basicAxis = op->main_as_Axis()->axis();
            } else {
                MNN_ERROR("Concat op axis is nullptr, set to 0 as default\n");
            }
        } else if (op->type() == OpType_QuantizedConcat) {
            basicAxis = op->main_as_QuantizedConcat()->axis();
        }
        int axis = basicAxis;
        // Concat-inputs may have scalar which should be delete
        for (const auto& input : inputs) {
            auto inputDimensions = input->buffer().dimensions;
            if (inputDimensions <= 0 || inputDimensions > MNN_MAX_TENSOR_DIM) {
                MNN_ERROR("Concat op input has invalid rank %d\n", inputDimensions);
                return false;
            }

            //  Tensor might be zeros size, but some dims may not be zero. should concat as usual.

            ::memcpy(ob.dim, input->buffer().dim, sizeof(halide_dimension_t) * inputDimensions);
            ob.dimensions = inputDimensions;
            ob.type       = input->buffer().type;
            if (axis < 0) {
                axis = inputDimensions + axis;
            }
            if (axis < 0 || axis >= inputDimensions) {
                MNN_ERROR("Concat op axis %d out of range for %d dims\n", axis, inputDimensions);
                return false;
            }
            break;
        }


        int sum = 0;
        for (auto t : inputs) {
            if (axis >= t->dimensions()) {
                MNN_ERROR("Concat op axis %d out of range for %d dims\n", axis, t->dimensions());
                return false;
            }
            sum += t->buffer().dim[axis].extent;
            ob.type = t->buffer().type;
            for (int i = 0; i < t->dimensions(); ++i) {
                if (axis == i) {
                    continue;
                }
                if (t->length(i) != outputs[0]->length(i)) {
                    auto name = op->name() ? op->name()->c_str() : "";
                    MNN_PRINT("Error for concat size of op [ %s ], the %d input not match output\n", name, i);
                    return false;
                }
            }
        }
        ob.dim[axis].extent                                   = sum;
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        return true;
    }
};

REGISTER_SHAPE(ConcatSizeComputer, OpType_Concat);
REGISTER_SHAPE(ConcatSizeComputer, OpType_QuantizedConcat);
} // namespace MNN
