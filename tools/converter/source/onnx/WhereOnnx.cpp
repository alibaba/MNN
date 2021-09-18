//
//  WhereOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2020/07/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(WhereOnnx);

MNN::OpType WhereOnnx::opType() {
    return MNN::OpType_Select;
}

MNN::OpParameter WhereOnnx::type() {
    return MNN::OpParameter_NONE;
}

void WhereOnnx::run(MNN::OpT *dstOp, const onnx::NodeProto *onnxNode,
                   OnnxScope* scope) {
    return;
}

REGISTER_CONVERTER(WhereOnnx, Where);
