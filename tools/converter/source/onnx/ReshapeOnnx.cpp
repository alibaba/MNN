//
//  ReshapeOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/06/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(ReshapeOnnx);

MNN::OpType ReshapeOnnx::opType() {
    return MNN::OpType_Reshape;
}
MNN::OpParameter ReshapeOnnx::type() {
    return MNN::OpParameter_Reshape;
}

void ReshapeOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                      OnnxScope* scope) {
    auto para = new MNN::ReshapeT;
    para->dimType = MNN::MNN_DATA_FORMAT_NCHW;
    dstOp->main.value = para;
}

REGISTER_CONVERTER(ReshapeOnnx, Reshape);
