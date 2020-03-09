//
//  ShapeBinaryOp.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "core/Macro.h"
#include "core/SizeComputer.hpp"
#include <vector>
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
        if (input0->getType().code != input1->getType().code) {
            MNN_PRINT("Error for binary op: input0's type != input1's type\n");
            return false;
        }
        if (input0->dimensions() < input1->dimensions()) {
            auto temp = input0;
            input0 = input1;
            input1 = temp;
        }
        TensorUtils::getDescribe(output)->dimensionFormat = TensorUtils::getDescribe(input0)->dimensionFormat;

        // if one scalar input -> just copy the other
        if (input1->dimensions() == 0) {
            TensorUtils::copyShape(input0, output);
            return true;
        }

        // else if inputs shape equals -> just copy any one
        bool sameShape = true;
        if (input0->dimensions() == input1->dimensions()) {
            for (int i = 0; i < input0->buffer().dimensions; i++) {
                if (input0->buffer().dim[i].extent != input1->buffer().dim[i].extent) {
                    sameShape = false;
                    break;
                }
            }
        }
        else {
            sameShape = false;
        }
        if (sameShape) {
            TensorUtils::copyShape(input0, output);
            return true;
        }
        
        // else if broadcast NOT supported -> failed
        const int maxDimensions = input0->dimensions();
        const int diffDimension = input0->dimensions() - input1->dimensions();
        
        std::vector<int> outputDims(maxDimensions);
        for (int i = 0; i < maxDimensions; i++) {
            outputDims[i] = input0->buffer().dim[i].extent;
        }
        for (int i = diffDimension; i < maxDimensions; i++) {
            const int input1Index = i - diffDimension;
            int dim1 = input1->buffer().dim[input1Index].extent;
            if (dim1 != outputDims[i] && (dim1 != 1 && outputDims[i] != 1)) {
                MNN_PRINT("Don't support broadcast for binaryOp, i0=%d, i1=%d\n", outputDims[i], dim1);
                return false;
            }
            if (dim1 == outputDims[i]) {
                continue;
            }
            if (dim1 != outputDims[i] && (dim1 == 1 || outputDims[i] == 1)) {
                outputDims[i] = outputDims[i] * dim1;
            } else {
                MNN_PRINT("Error, the logic flow should never get here");
                return false;
            }
        }

        buffer.dimensions = maxDimensions;
        for (int i = 0; i < maxDimensions; i++) {
            buffer.dim[i].extent = outputDims[i];
        }

        return true;
    }
};

REGISTER_SHAPE(BinaryOpComputer, OpType_BinaryOp);
} // namespace MNN
