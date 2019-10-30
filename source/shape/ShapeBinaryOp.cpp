//
//  ShapeBinaryOp.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Macro.h"
#include "SizeComputer.hpp"
namespace MNN {
class BinaryOpComputer : public SizeComputer {
public:
    static bool outputBool(int operation) {
        if (operation == BinaryOpOperation_GREATER_EQUAL) {
            return true;
        }
        if (operation == BinaryOpOperation_GREATER) {
            return true;
        }
        if (operation == BinaryOpOperation_LESS) {
            return true;
        }
        if (operation == BinaryOpOperation_LESS_EQUAL) {
            return true;
        }
        if (operation == BinaryOpOperation_EQUAL) {
            return true;
        }
        return false;
    }
    virtual bool onComputeSize(const Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(2 == inputs.size());
        MNN_ASSERT(1 == outputs.size());
        // set output type & format
        auto input0 = inputs[0], input1 = inputs[1], output = outputs[0];
        auto &buffer = output->buffer();
        const auto opType = op->main_as_BinaryOp()->opType();
        if (outputBool(opType)) {
            buffer.type = halide_type_of<int32_t>();
        } else {
            buffer.type = input0->getType();
        }
        TensorUtils::getDescribe(output)->dimensionFormat = TensorUtils::getDescribe(input0)->dimensionFormat;
        if (input0->dimensions() < input1->dimensions()) {
            auto temp = input0;
            input0 = input1;
            input1 = temp;
        }

        // if scalar input -> just copy the other
        if (input1->dimensions() == 0) {
            TensorUtils::copyShape(input0, output);
            return true;
        }

        // else if inputs shape equals -> just copy any one
        bool sameShape = input0->elementSize() == input1->elementSize();
        if (sameShape) {
            TensorUtils::copyShape(input0, output);
            return true;
        }
        
        // else if broadcast NOT supported -> failed
        const int maxDimensions = input0->dimensions();
        const int diffDimension = input0->dimensions() - input1->dimensions();
        
        // else broadcast
        for (int i = maxDimensions-1; i >=0 ; --i) {
            auto input0Length = input0->length(i);
            auto input1Length = 1;
            if (i >= diffDimension) {
                input1Length = input1->length(i-diffDimension);
            }
            if (input0Length != input1Length && input1Length != 1 && input0Length != 1) {
                MNN_PRINT("%d, %d\n", input1Length, input0Length);
                return false;
            }
            buffer.dim[i].extent = std::max(input0Length, input1Length);
        }
        buffer.dimensions = maxDimensions;
        return true;
    }
};

REGISTER_SHAPE(BinaryOpComputer, OpType_BinaryOp);
} // namespace MNN
