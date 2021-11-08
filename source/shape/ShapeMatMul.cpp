//
//  ShapeMatMul.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {

class MatMulSizeComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(1 == outputs.size());
        bool transposeA = false;
        bool transposeB = false;
        if (op->type() == OpType_MatMul) {
            transposeA = op->main_as_MatMul()->transposeA();
            transposeB = op->main_as_MatMul()->transposeB();
        } else {
            // BatchMatMul
            transposeA = op->main_as_BatchMatMulParam()->adjX();
            transposeB = op->main_as_BatchMatMulParam()->adjY();
        }
        auto i0Dim = inputs[0]->dimensions();
        auto i1Dim = inputs[1]->dimensions();
        if (i0Dim < 2 || i1Dim < 2) {
            return false;
        }

        auto output = outputs[0];
        auto w0 = inputs[0]->length(i0Dim - 1);
        auto h0 = inputs[0]->length(i0Dim - 2);
        output->buffer().type = inputs[0]->buffer().type;

        if (transposeA) {
            auto t = w0;
            w0     = h0;
            h0     = t;
        }

        auto w1 = inputs[1]->length(i1Dim - 1);
        auto h1 = inputs[1]->length(i1Dim - 2);
        if (transposeB) {
            auto t = w1;
            w1     = h1;
            h1     = t;
        }

        if (w0 != h1) {
            return false;
        }
        // Compute BroastCast Dims
        auto input0 = inputs[0];
        auto input1 = inputs[1];
        auto o0Dim = i0Dim;
        if (i1Dim > i0Dim) {
            o0Dim = i1Dim;
            input0 = inputs[1];
            input1 = inputs[0];
        }
        auto dimOffset = o0Dim - 2;
        output->buffer().dimensions = o0Dim;
        const int maxDimensions = dimOffset;
        const int diffDimension = input0->dimensions() - input1->dimensions();
        
        for (int i = 0; i < maxDimensions; i++) {
            output->setLength(i, input0->length(i));
        }
        for (int i = diffDimension; i < maxDimensions; i++) {
            const int input1Index = i - diffDimension;
            int dim1 = input1->buffer().dim[input1Index].extent;
            if (dim1 != output->length(i) && (dim1 != 1 && output->length(i) != 1)) {
                MNN_PRINT("Don't support broadcast for MatMulOp, i0=%d, i1=%d\n", output->length(i), dim1);
                return false;
            }
            if (dim1 == output->length(i)) {
                continue;
            }
            if (dim1 != output->length(i) && (dim1 == 1 || output->length(i) == 1)) {
                output->setLength(i, output->length(i) * dim1);
            } else {
                MNN_PRINT("Error, the logic flow should never get here");
                return false;
            }
        }
        // Last Two dim
        output->setLength(o0Dim - 2, h0);
        output->setLength(o0Dim - 1, w1);
        
        TensorUtils::getDescribe(output)->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        return true;
    }
    virtual float onComputeFlops(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                                 const std::vector<Tensor*>& outputs) const override {
        Tensor* C       = outputs[0];
        auto w0         = inputs[0]->length(1);
        auto h0         = inputs[0]->length(0);
        auto e = C->length(0);
        auto h = C->length(1);
        auto l = w0;
        const auto mat = op->main_as_MatMul();
        if (mat->transposeA()) {
            l = h0;
        }
        auto flops = (float)e * l * h / FLOPS_M;
        for (int i=0; i<C->dimensions() - 2; ++i) {
            flops *= C->length(i);
        }
        return flops;
    }
};

REGISTER_SHAPE(MatMulSizeComputer, OpType_MatMul);
REGISTER_SHAPE(MatMulSizeComputer, OpType_BatchMatMul);
} // namespace MNN
