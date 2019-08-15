//
//  ShapeBinaryOp.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <set>
#include "Macro.h"
#include "SizeComputer.hpp"
#include "TensorUtils.hpp"
//#define FORCE_SAME_SHAPE
namespace MNN {
class BinaryOpComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(2 == inputs.size());
        MNN_ASSERT(1 == outputs.size());
        
        // set output type & format
        static std::set<int> supportedTypes{BinaryOpOperation_GREATER, BinaryOpOperation_GREATER_EQUAL,
            BinaryOpOperation_LESS};
        auto input0 = inputs[0], input1 = inputs[1], output = outputs[0];
        auto &buffer = output->buffer();
        const auto opType = op->main_as_BinaryOp()->opType();
        if (supportedTypes.find(opType) != supportedTypes.end()) {
            buffer.type = halide_type_of<int32_t>();
        } else {
            buffer.type = input0->getType();
        }
        TensorUtils::getDescribe(output)->dimensionFormat = TensorUtils::getDescribe(input0)->dimensionFormat;

        // if scalar input -> just copy the other
        if (input0->dimensions() == 0) {
            TensorUtils::copyShape(input1, output);
            return true;
        }
        if (input1->dimensions() == 0) {
            TensorUtils::copyShape(input0, output);
            return true;
        }
        
        // else if inputs shape equals -> just copy any one
#ifdef FORCE_SAME_SHAPE
        bool sameShape = true;
        for (int i = 0; i < input0->dimensions(); ++i) {
            if (input0->length(i) != input1->length(i)) {
                sameShape = false;
                break;
            }
        }
#else
        bool sameShape = input0->elementSize() == input1->elementSize();
#endif
        if (sameShape) {
            TensorUtils::copyShape(input0, output);
            return true;
        }
        
        // else if broadcast NOT supported -> failed
        const int maxDimensions = std::max(input0->dimensions(), input1->dimensions());
        std::vector<int> dims0(maxDimensions, 1), dims1(maxDimensions, 1);
        for (int i = input0->dimensions() - 1, j = maxDimensions - 1; i >= 0; i--, j--) {
            dims0[j] = input0->length(i);
        }
        for (int i = input1->dimensions() - 1, j = maxDimensions - 1; i >= 0; i--, j--) {
            dims1[j] = input1->length(i);
        }
        for (int i = 0; i < maxDimensions; i++) {
            if (dims0[i] != dims1[i] && dims0[i] != 1 && dims1[i] != 1) {
                return false;
            }
        }
        
        // else broadcast
        for (int i = 0; i < maxDimensions; i++) {
            buffer.dim[i].extent = std::max(dims0[i], dims1[i]);
            buffer.dim[i].flags  = 0;
        }
        buffer.dimensions = maxDimensions;
        return true;
    }
};

REGISTER_SHAPE(BinaryOpComputer, OpType_BinaryOp);
} // namespace MNN
