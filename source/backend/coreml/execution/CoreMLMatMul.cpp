//
//  CoreMLMatMul.cpp
//  MNN
//
//  Created by MNN on 2021/03/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CoreMLMatMul.hpp"
namespace MNN {

CoreMLMatMul::CoreMLMatMul(MNN::Backend* b, const MNN::Op* op, const std::vector<Tensor*>& inputs,
                           const std::vector<MNN::Tensor*>& outputs)
    : CoreMLCommonExecution(b, op) {}

ErrorCode CoreMLMatMul::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    bool transposeX = false, transposeY = false;
    if (mOp->main_type() == OpParameter_MatMul) {
        transposeX = mOp->main_as_MatMul()->transposeA();
        transposeY = mOp->main_as_MatMul()->transposeB();
    } else if (mOp->main_type() == OpParameter_BatchMatMulParam) {
        transposeX = mOp->main_as_BatchMatMulParam()->adjX();
        transposeY = mOp->main_as_BatchMatMulParam()->adjY();
    }
    auto builder = operation("matmul");
    builder.input("x", inputs[0])
        .input("y", inputs[1])
        .scalar("transpose_x", transposeX)
        .scalar("transpose_y", transposeY);
    if (inputs.size() > 2) {
        auto product = builder.output(outputs[0]->shape(), "product");
        auto add = operation("add");
        add.input("x", product).input("y", inputs[2]);
        add.output(outputs);
    } else {
        builder.output(outputs);
    }
    return NO_ERROR;
}

REGISTER_COREML_OP_CREATOR(CoreMLMatMul, OpType_BatchMatMul);
REGISTER_COREML_OP_CREATOR(CoreMLMatMul, OpType_MatMul);

} // namespace MNN
