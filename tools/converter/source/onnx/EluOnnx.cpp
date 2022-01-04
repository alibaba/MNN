//
//  EluOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/11/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(EluOnnx);
DECLARE_OP_CONVERTER(SEluOnnx);

MNN::OpType EluOnnx::opType(){
    return MNN::OpType_ELU;
}
MNN::OpType SEluOnnx::opType(){
    return MNN::OpType_Selu;
}

MNN::OpParameter EluOnnx::type(){
    return MNN::OpParameter_ELU;
}
MNN::OpParameter SEluOnnx::type(){
    return MNN::OpParameter_Selu;
}

void EluOnnx::run(MNN::OpT *dstOp, const onnx::NodeProto *onnxNode, OnnxScope* scope){
    
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
void SEluOnnx::run(MNN::OpT *dstOp, const onnx::NodeProto *onnxNode, OnnxScope* scope){
    
    auto seluParam = new MNN::SeluT;
    
    float alpha = 1.67326, gamma = 1.0507;
    for (int i = 0; i < onnxNode->attribute_size(); ++i) {
        const auto &attributeProto = onnxNode->attribute(i);
        const auto &attributeName  = attributeProto.name();
        if (attributeName == "alpha") {
            alpha = attributeProto.f();
        } else if (attributeName == "gamma") {
            gamma = attributeProto.f();
        }
    }
    
    seluParam->alpha = alpha;
    seluParam->scale = gamma;
    
    dstOp->main.value = seluParam;
}

REGISTER_CONVERTER(EluOnnx, Elu);
REGISTER_CONVERTER(SEluOnnx, Selu);
