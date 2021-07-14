//
//  DropoutOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/11/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(DropoutOnnx);

MNN::OpType DropoutOnnx::opType(){
    return MNN::OpType_Dropout;
}

MNN::OpParameter DropoutOnnx::type(){
    return MNN::OpParameter_NONE;
}

void DropoutOnnx::run(MNN::OpT *dstOp, const onnx::NodeProto *onnxNode, std::vector<const onnx::TensorProto *> initializers){
    DCHECK(3 == onnxNode->input_size()) << "ONNX Dropout should have 3 inputs!";
    return;
}

REGISTER_CONVERTER(DropoutOnnx, Dropout);
