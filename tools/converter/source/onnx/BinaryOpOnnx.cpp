//
//  BinaryOpOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(BinaryOpOnnx);

MNN::OpType BinaryOpOnnx::opType() {
    return MNN::OpType_BinaryOp;
}

MNN::OpParameter BinaryOpOnnx::type() {
    return MNN::OpParameter_BinaryOp;
}

void BinaryOpOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                       std::vector<const onnx::TensorProto*> initializers) {
    auto param = new MNN::BinaryOpT;

    auto type = onnxNode->op_type();
    if (type == "Add" || type == "Sum") {
        param->opType = MNN::BinaryOpOperation_ADD;
    } else {
        DLOG(ERROR) << "TODO";
    }
    param->T          = MNN::DataType_DT_FLOAT;
    dstOp->main.value = param;
}

REGISTER_CONVERTER(BinaryOpOnnx, Sum);
REGISTER_CONVERTER(BinaryOpOnnx, Add);
