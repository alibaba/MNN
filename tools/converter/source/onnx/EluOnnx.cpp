//
//  EluOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/11/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(EluOnnx);

MNN::OpType EluOnnx::opType(){
    return MNN::OpType_ELU;
}

MNN::OpParameter EluOnnx::type(){
    return MNN::OpParameter_ELU;
}

void EluOnnx::run(MNN::OpT *dstOp, const onnx::NodeProto *onnxNode, std::vector<const onnx::TensorProto *> initializers){
    
    auto eluParam = new MNN::ELUT;
    
    float alpha = 1.0f;
    for (int i = 0; i < onnxNode->attribute_size(); ++i) {
        const auto &attributeProto = onnxNode->attribute(i);
        const auto &attributeName  = attributeProto.name();
        if (attributeName == "alpha") {
            alpha = attributeProto.f();
        }
    }
    
    eluParam->alpha = alpha;
    
    dstOp->main.value = eluParam;
}

REGISTER_CONVERTER(EluOnnx, Elu);
