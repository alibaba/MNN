//
//  ShapeTranspose.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
namespace MNN {

class TransposeComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        const Tensor* input = inputs[0];
        Tensor* perm        = inputs[1];
        const int dims = input->buffer().dimensions;
        if (perm->getType().code != halide_type_int || 32 != perm->getType().bits || dims != perm->buffer().dim[0].extent) {
            return false;
        }
        auto permutation = perm->host<int32_t>();
        outputs[0]->buffer().dimensions = dims;
        outputs[0]->buffer().type = input->getType();
        for (int i = 0; i < dims; ++i) {
            const int32_t d                    = permutation[i];
            if (d < 0 || d >= dims) {
                return false;
            }
            outputs[0]->buffer().dim[i].extent = input->buffer().dim[d].extent;
        }
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;

        return true;
    }
};

REGISTER_SHAPE_INPUTS(TransposeComputer, OpType_Transpose, {1});
} // namespace MNN
