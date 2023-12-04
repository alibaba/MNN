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
#include "core/OpCommonUtils.hpp"

namespace MNN {

class MatMulSizeComputer : public SizeComputer {
    static void _getTranspose(const MNN::Op* op, bool& transposeA, bool& transposeB) {
        transposeA = false;
        transposeB = false;
        if (op->type() == OpType_MatMul) {
            transposeA = op->main_as_MatMul()->transposeA();
            transposeB = op->main_as_MatMul()->transposeB();
        } else {
            // BatchMatMul
            transposeA = op->main_as_BatchMatMulParam()->adjX();
            transposeB = op->main_as_BatchMatMulParam()->adjY();
        }
    }
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(1 == outputs.size());
        auto output = outputs[0];
        output->buffer().type = inputs[0]->buffer().type;
        bool transposeA;
        bool transposeB;
        _getTranspose(op, transposeA, transposeB);
        int e, l, h;
        bool valid = OpCommonUtils::computeMatMulSize(transposeA, transposeB, inputs[0], inputs[1], e, l, h);
        if (!valid) {
            return false;
        }
        // Compute BroastCast Dims
        auto i0Dim = inputs[0]->dimensions();
        auto i1Dim = inputs[1]->dimensions();

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
        output->setLength(o0Dim - 2, e);
        output->setLength(o0Dim - 1, h);
        bool eValid = inputs[0]->dimensions() > 1;
        bool hValid = inputs[1]->dimensions() > 1;
        int squeezeDim = 0;
        if (!eValid) {
            squeezeDim++;
            output->setLength(o0Dim - 2, h);
        }
        if (!hValid) {
            squeezeDim++;
            output->setLength(o0Dim - 1, e);
        }
        if (squeezeDim > 0) {
            output->buffer().dimensions = o0Dim - squeezeDim;
        }

        TensorUtils::getDescribe(output)->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        return true;
    }
    virtual float onComputeFlops(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                                 const std::vector<Tensor*>& outputs) const override {
        bool transposeA;
        bool transposeB;
        _getTranspose(op, transposeA, transposeB);
        int e=0, l=0, h=0;
        OpCommonUtils::computeMatMulSize(transposeA, transposeB, inputs[0], inputs[1], e, l, h);
        Tensor* C       = outputs[0];
        auto flops = (float)e * l * h / FLOPS_M;
        bool eValid = inputs[0]->dimensions() > 1;
        bool hValid = inputs[1]->dimensions() > 1;
        int squeezeDim = 0;
        if (!eValid) {
            squeezeDim++;
        }
        if (!hValid) {
            squeezeDim++;
        }
        for (int i=0; i<C->dimensions() - 2 + squeezeDim; ++i) {
            flops *= C->length(i);
        }
        return flops;
    }
};

REGISTER_SHAPE(MatMulSizeComputer, OpType_MatMul);
REGISTER_SHAPE(MatMulSizeComputer, OpType_BatchMatMul);
} // namespace MNN
