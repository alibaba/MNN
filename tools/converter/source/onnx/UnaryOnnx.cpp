//
//  UnaryOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/10/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(UnaryOnnx);

MNN::OpType UnaryOnnx::opType() {
    return MNN::OpType_UnaryOp;
}

MNN::OpParameter UnaryOnnx::type() {
    return MNN::OpParameter_UnaryOp;
}

void UnaryOnnx::run(MNN::OpT *dstOp, const onnx::NodeProto *onnxNode,
                    std::vector<const onnx::TensorProto *> initializers) {
    std::unique_ptr<MNN::UnaryOpT> unaryOpParam(new MNN::UnaryOpT);
    unaryOpParam->T = MNN::DataType_DT_FLOAT;

    const auto &originalType = onnxNode->op_type();

#define TO_UNARY_OP(src, dst)       \
    if (originalType == src) {      \
        unaryOpParam->opType = dst; \
    }

    TO_UNARY_OP("Floor", MNN::UnaryOpOperation_FLOOR);

    dstOp->main.value = unaryOpParam.release();
}

REGISTER_CONVERTER(UnaryOnnx, Floor);
