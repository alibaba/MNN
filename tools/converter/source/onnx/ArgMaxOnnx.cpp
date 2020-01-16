//
//  ArgMax.cpp
//  MNNConverter
//
//  Created by MNN on 2020/01/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "onnxOpConverter.hpp"
#include <MNN/MNNDefine.h>

DECLARE_OP_CONVERTER(ArgMaxOnnx);

MNN::OpType ArgMaxOnnx::opType(){
    return MNN::OpType_ArgMax;
}

MNN::OpParameter ArgMaxOnnx::type(){
    return MNN::OpParameter_ArgMax;
}

void ArgMaxOnnx::run(MNN::OpT *dstOp, const onnx::NodeProto *onnxNode, std::vector<const onnx::TensorProto *> initializers){
    auto axisT              = new MNN::ArgMaxT;
    int axis = 0;
    int keepdims = 1;
    int selectLastIndex = 0; // Boolean value. Default to False.

    for (int i = 0; i < onnxNode->attribute_size(); ++i) {
        const auto& attributeProto = onnxNode->attribute(i);
        const auto& attributeName  = attributeProto.name();

        if (attributeName == "axis") {
            axis = attributeProto.i();
        }
        if (attributeName == "keepdims") {
            keepdims = attributeProto.i();
        }
        if (attributeName == "select_last_index") {
            // Ignored for now. MNN argmax implementation does not support this yet.
            selectLastIndex = attributeProto.i();
        }
    }
    if (keepdims == 1) {
        MNN_ERROR("ONNX ArgMax with keepdims == true is currently not supported.\n");
    }
    axisT->axis = axis;
    axisT->topK = 1;
    axisT->outMaxVal = 0;
    dstOp->main.value = axisT;
}

REGISTER_CONVERTER(ArgMaxOnnx, ArgMax);
