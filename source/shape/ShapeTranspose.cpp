//
//  ShapeTranspose.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "core/Macro.h"
#include "core/SizeComputer.hpp"
#include "core/TensorUtils.hpp"
namespace MNN {

class TransposeComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        const Tensor* input = inputs[0];
        Tensor* perm        = inputs[1];
        const int dims = input->buffer().dimensions;
        MNN_ASSERT(dims == perm->buffer().dim[0].extent);

        std::vector<int32_t> permutation;
        if (perm->getType().code == halide_type_int && 32 == perm->getType().bits) {
            for (int i = 0; i < perm->buffer().dim[0].extent; i++) {
                permutation.push_back(perm->host<int32_t>()[i]);
            }
        } else {
            MNN_ASSERT(false);
        }

        outputs[0]->buffer().dimensions = dims;
        outputs[0]->buffer().type = input->getType();
        for (int i = 0; i < dims; ++i) {
            const int32_t d                    = permutation[i];
            outputs[0]->buffer().dim[i].extent = input->buffer().dim[d].extent;
        }
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;

        return true;
    }
};

REGISTER_SHAPE_INPUTS(TransposeComputer, OpType_Transpose, {1});
} // namespace MNN
