//
//  DetOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/11/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(DetOnnx);

MNN::OpType DetOnnx::opType(){
    return MNN::OpType_Det;
}
MNN::OpParameter DetOnnx::type(){
    return MNN::OpParameter_NONE;
}

void DetOnnx::run(MNN::OpT *dstOp, const onnx::NodeProto *onnxNode, OnnxScope* scope){
    
}

REGISTER_CONVERTER(DetOnnx, Det);
