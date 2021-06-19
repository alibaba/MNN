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
    return MNN::OpParameter_Dropout;
}

void DropoutOnnx::run(MNN::OpT *dstOp, const onnx::NodeProto *onnxNode, std::vector<const onnx::TensorProto *> initializers){
    
    auto dropoutParam = new MNN::DropoutT;
    
    float ratio = 0.5f;
    for (int i = 0; i < onnxNode->attribute_size(); ++i) {
        const auto &attributeProto = onnxNode->attribute(i);
        const auto &attributeName  = attributeProto.name();
        if (attributeName == "ratio") {
            ratio = attributeProto.f();
        }
    }
    
    dropoutParam->ratio = ratio;
    
    dstOp->main.value = dropoutParam;
}

REGISTER_CONVERTER(DropoutOnnx, Dropout);
