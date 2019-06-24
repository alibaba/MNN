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
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(2 == inputs.size());
        MNN_ASSERT(1 == outputs.size());
        static std::set<int> supportedTypes{MNN::BinaryOpOperation_GREATER, MNN::BinaryOpOperation_GREATER_EQUAL,
            MNN::BinaryOpOperation_LESS};
        
        auto &input0 = inputs[0]->buffer(), &input1 = inputs[1]->buffer(), &output = outputs[0]->buffer();
        const auto opType = op->main_as_BinaryOp()->opType();
        if (supportedTypes.find(opType) != supportedTypes.end()) {
            output.type = halide_type_of<int32_t>();
        } else {
            output.type = input0.type;
        }

        if (input0.dimensions == 0) {
            ::memcpy(output.dim, input1.dim, input1.dimensions * sizeof(halide_dimension_t));
            output.dimensions = input1.dimensions;
        } else if (input1.dimensions == 0) {
            ::memcpy(output.dim, input0.dim, input0.dimensions * sizeof(halide_dimension_t));
            output.dimensions = input0.dimensions;
        } else { // no scalar input
#ifdef FORCE_SAME_SHAPE
            bool sameShape = true;
            for (int i = 0; i < inputs[0]->dimensions(); ++i) {
                if (inputs[0]->length(i) != inputs[1]->length(i)) {
                    sameShape = false;
                    break;
                }
            }
#else
            bool sameShape = inputs[0]->elementSize() == inputs[1]->elementSize();
#endif
            if (sameShape) {
                ::memcpy(output.dim, input0.dim, input0.dimensions * sizeof(halide_dimension_t));
                output.dimensions = input0.dimensions;
            } else { // not the same shape, use broadcast
                const int maxDimensions = std::max(input0.dimensions, input1.dimensions);

                std::vector<int> dims0(maxDimensions, 1), dims1(maxDimensions, 1);
                for (int i = input0.dimensions - 1, j = maxDimensions - 1; i >= 0; i--, j--) {
                    dims0[j] = input0.dim[i].extent;
                }
                for (int i = input1.dimensions - 1, j = maxDimensions - 1; i >= 0; i--, j--) {
                    dims1[j] = input1.dim[i].extent;
                }
                bool supportBroadcast = true;
                for (int i = 0; i < maxDimensions; i++) {
                    if ((dims0[i] != dims1[i]) && !(dims0[i] == 1 || dims1[i] == 1)) {
                        supportBroadcast = false;
                        break;
                    }
                }
                if (supportBroadcast) {
                    for (int i = 0; i < maxDimensions; i++) {
                        output.dim[i].extent = std::max(dims0[i], dims1[i]);
                        output.dim[i].flags  = 0;
                    }
                    output.dimensions = maxDimensions;
                } else {
                    return false;
                }
            }
        }
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;

        return true;
    }
};

REGISTER_SHAPE(BinaryOpComputer, OpType_BinaryOp);
} // namespace MNN
