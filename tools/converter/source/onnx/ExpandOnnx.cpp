//
//  ExpandOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/12/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(ExpandOnnx);

MNN::OpType ExpandOnnx::opType(){
    return MNN::OpType_BroadcastTo;
}

MNN::OpParameter ExpandOnnx::type(){
    return MNN::OpParameter_NONE;
}

void ExpandOnnx::run(MNN::OpT *dstOp, const onnx::NodeProto *onnxNode, OnnxScope* scope){
    DCHECK(2 == onnxNode->input_size()) << "ONNX Expand should have 2 inputs!";
    return;
}

REGISTER_CONVERTER(ExpandOnnx, Expand);
