//
//  SplitGeLUOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2023/09/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(SplitGeLUOnnx);
DECLARE_OP_CONVERTER(SeqLen2SpatialOnnx);

MNN::OpType SplitGeLUOnnx::opType(){
    return MNN::OpType_SplitGeLU;
}

MNN::OpParameter SplitGeLUOnnx::type(){
    return MNN::OpParameter_NONE;
}

void SplitGeLUOnnx::run(MNN::OpT *dstOp, const onnx::NodeProto *onnxNode, OnnxScope* scope){
    return;
}


MNN::OpType SeqLen2SpatialOnnx::opType(){
    return MNN::OpType_SeqLen2Spatial;
}

MNN::OpParameter SeqLen2SpatialOnnx::type(){
    return MNN::OpParameter_NONE;
}

void SeqLen2SpatialOnnx::run(MNN::OpT *dstOp, const onnx::NodeProto *onnxNode, OnnxScope* scope){
    return;
}

REGISTER_CONVERTER(SplitGeLUOnnx, SplitGeLU);
REGISTER_CONVERTER(SeqLen2SpatialOnnx, SeqLen2Spatial);
