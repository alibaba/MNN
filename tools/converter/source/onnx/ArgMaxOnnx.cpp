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
            selectLastIndex = attributeProto.i();
        }
    }
    axisT->axis = axis;
    axisT->topK = 1;
    axisT->outMaxVal = 0;
    if (selectLastIndex == 1) {
        // The first extremum in the reversed axis is the last one in the original input.
        auto addOp = [&](MNN::OpType type, const std::string& suffix, const std::vector<int>& inputs) {
            std::unique_ptr<MNN::OpT> op(new MNN::OpT);
            op->name = dstOp->name + suffix;
            op->type = type;
            op->inputIndexes = inputs;
            op->outputIndexes = {scope->declareTensor(op->name)};
            auto result = op.get();
            scope->oplists().emplace_back(std::move(op));
            return result;
        };
        auto scalar = [&](int value, const std::string& suffix) {
            auto op = addOp(MNN::OpType_Const, suffix, {});
            op->main.type = MNN::OpParameter_Blob;
            auto blob = new MNN::BlobT;
            blob->dataType = MNN::DataType_DT_INT32;
            blob->dataFormat = MNN::MNN_DATA_FORMAT_NCHW;
            blob->int32s = {value};
            op->main.value = blob;
            return op->outputIndexes[0];
        };
        auto binary = [&](MNN::BinaryOpOperation operation, int x, int y, const std::string& suffix) {
            auto op = addOp(MNN::OpType_BinaryOp, suffix, {x, y});
            op->main.type = MNN::OpParameter_BinaryOp;
            auto param = new MNN::BinaryOpT;
            param->opType = operation;
            param->T = MNN::DataType_DT_INT32;
            op->main.value = param;
            return op->outputIndexes[0];
        };
        const int input = dstOp->inputIndexes[0];
        int axisIndex = scalar(axis, "/axis");
        if (axis < 0) {
            auto rank = addOp(MNN::OpType_Rank, "/rank", {input});
            axisIndex = binary(MNN::BinaryOpOperation_ADD, rank->outputIndexes[0], axisIndex, "/positive_axis");
        }
        auto reverse = addOp(MNN::OpType_Reverse, "/reverse", {input, axisIndex});
        auto arg = addOp(dstOp->type, "/reversed_index", reverse->outputIndexes);
        arg->main.type = MNN::OpParameter_ArgMax;
        arg->main.value = axisT;
        auto shape = addOp(MNN::OpType_Shape, "/shape", {input});
        shape->defaultDimentionFormat = MNN::MNN_DATA_FORMAT_NCHW;
        auto size = addOp(MNN::OpType_Gather, "/axis_size", {shape->outputIndexes[0], axisIndex});
        const int one = scalar(1, "/one");
        const int last = binary(MNN::BinaryOpOperation_SUB, size->outputIndexes[0], one, "/axis_last_index");
        auto param = new MNN::BinaryOpT;
        param->opType = MNN::BinaryOpOperation_SUB;
        param->T = MNN::DataType_DT_INT32;
        dstOp->type = MNN::OpType_BinaryOp;
        dstOp->main.type = MNN::OpParameter_BinaryOp;
        dstOp->main.value = param;
        dstOp->inputIndexes = {last, arg->outputIndexes[0]};
        if (keepdims == 1) {
            auto remap = addOp(dstOp->type, "/last_index", dstOp->inputIndexes);
            remap->main.type = dstOp->main.type;
            remap->main.value = param;
            dstOp->inputIndexes = remap->outputIndexes;
            dstOp->type = MNN::OpType_Unsqueeze;
            auto squeeze = new MNN::SqueezeParamT;
            squeeze->squeezeDims = {axis};
            dstOp->main.type = MNN::OpParameter_SqueezeParam;
            dstOp->main.value = squeeze;
        }
        return;
    }
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
