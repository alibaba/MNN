//
//  ArgMaxOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2020/01/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "onnxOpConverter.hpp"
#include <MNN/MNNDefine.h>

DECLARE_OP_CONVERTER(ArgMaxOnnx);
DECLARE_OP_CONVERTER(ArgMinOnnx);

MNN::OpType ArgMaxOnnx::opType(){
    return MNN::OpType_ArgMax;
}

MNN::OpParameter ArgMaxOnnx::type(){
    return MNN::OpParameter_ArgMax;
}

MNN::OpType ArgMinOnnx::opType(){
    return MNN::OpType_ArgMin;
}

MNN::OpParameter ArgMinOnnx::type(){
    return MNN::OpParameter_ArgMax;
}

static void _run(MNN::OpT *dstOp, const onnx::NodeProto *onnxNode, OnnxScope* scope){
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
    axisT->axis = axis;
    axisT->topK = 1;
    axisT->outMaxVal = 0;
    if (keepdims == 1) {
        std::unique_ptr<MNN::OpT> op(new MNN::OpT);
        op->name = dstOp->name + "/not_keepdim";
        op->type = dstOp->type;
        op->main.type = dstOp->main.type;
        op->main.value = axisT;
        op->inputIndexes = dstOp->inputIndexes;
        std::vector<int> midIndexs(1, scope->declareTensor(op->name));
        op->outputIndexes = dstOp->inputIndexes = midIndexs;
        dstOp->type = MNN::OpType_Unsqueeze;
        auto param = new MNN::SqueezeParamT;
        param->squeezeDims.assign({axis});
        dstOp->main.type = MNN::OpParameter_SqueezeParam;
        dstOp->main.value = param;
        scope->oplists().emplace_back(std::move(op));
        return;
    }
    dstOp->main.value = axisT;
}

void ArgMaxOnnx::run(MNN::OpT *dstOp, const onnx::NodeProto *onnxNode, OnnxScope* scope){
    _run(dstOp, onnxNode, scope);
}

void ArgMinOnnx::run(MNN::OpT *dstOp, const onnx::NodeProto *onnxNode, OnnxScope* scope){
    _run(dstOp, onnxNode, scope);
}

REGISTER_CONVERTER(ArgMaxOnnx, ArgMax);
REGISTER_CONVERTER(ArgMinOnnx, ArgMin);
