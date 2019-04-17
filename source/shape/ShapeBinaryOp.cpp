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
//#define FORCE_SAME_SHAPE
namespace MNN {
class BinaryOpComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(2 == inputs.size());
        MNN_ASSERT(1 == outputs.size());

        const auto opType = op->main_as_BinaryOp()->opType();
        static std::set<int> int32Types{MNN::BinaryOpOperation_GREATER, MNN::BinaryOpOperation_GREATER_EQUAL,
                                        MNN::BinaryOpOperation_LESS};
        if (int32Types.find(opType) != int32Types.end()) {
            outputs[0]->setType(MNN::DataType_DT_INT32);
        } else {
            outputs[0]->buffer().type = inputs[0]->buffer().type;
        }

        if (inputs[0]->buffer().dimensions == 0) {
            ::memcpy(outputs[0]->buffer().dim, inputs[1]->buffer().dim,
                     inputs[1]->buffer().dimensions * sizeof(halide_dimension_t));
            outputs[0]->buffer().dimensions = inputs[1]->buffer().dimensions;
        } else if (inputs[1]->buffer().dimensions == 0) {
            ::memcpy(outputs[0]->buffer().dim, inputs[0]->buffer().dim,
                     inputs[0]->buffer().dimensions * sizeof(halide_dimension_t));
            outputs[0]->buffer().dimensions = inputs[0]->buffer().dimensions;
        } else { // no scalar input
#ifdef FORCE_SAME_SHAPE
            bool same_shape = true;
            for (int i = 0; i < inputs[0]->dimensions(); ++i) {
                if (inputs[0]->length(i) != inputs[1]->length(i)) {
                    same_shape = false;
                    break;
                }
            }
#else
            bool same_shape = inputs[0]->elementSize() == inputs[1]->elementSize();
#endif
            if (same_shape) {
                ::memcpy(outputs[0]->buffer().dim, inputs[0]->buffer().dim,
                         inputs[0]->buffer().dimensions * sizeof(halide_dimension_t));
                outputs[0]->buffer().dimensions = inputs[0]->buffer().dimensions;
            } else { // not the same shape, use broadcast
                const int max_dimensions = std::max(inputs[0]->buffer().dimensions, inputs[1]->buffer().dimensions);

                std::vector<int> dims0(max_dimensions, 1);
                std::vector<int> dims1(max_dimensions, 1);
                for (int i = inputs[0]->buffer().dimensions - 1, j = max_dimensions - 1; i >= 0; i--, j--) {
                    dims0[j] = inputs[0]->buffer().dim[i].extent;
                }
                for (int i = inputs[1]->buffer().dimensions - 1, j = max_dimensions - 1; i >= 0; i--, j--) {
                    dims1[j] = inputs[1]->buffer().dim[i].extent;
                }
                bool supportBroadcast = true;
                for (int i = 0; i < max_dimensions; i++) {
                    if ((dims0[i] != dims1[i]) && !(dims0[i] == 1 || dims1[i] == 1)) {
                        supportBroadcast = false;
                        break;
                    }
                }
                if (supportBroadcast) {
                    for (int i = 0; i < max_dimensions; i++) {
                        outputs[0]->buffer().dim[i].extent = std::max(dims0[i], dims1[i]);
                        outputs[0]->buffer().dim[i].flags  = 0;
                    }
                    outputs[0]->buffer().dimensions = max_dimensions;
                } else {
                    return false;
                }
            }
        }

        return true;
    }
};

REGISTER_SHAPE(BinaryOpComputer, OpType_BinaryOp);
} // namespace MNN
