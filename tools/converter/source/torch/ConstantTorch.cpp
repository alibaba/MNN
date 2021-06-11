//
//  ConstantTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2021/05/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(ConstantTorch);

MNN::OpType ConstantTorch::opType() {
    return MNN::OpType_Const;
}
MNN::OpParameter ConstantTorch::type() {
    return MNN::OpParameter_Blob;
}
std::vector<int> ConstantTorch::inputTensorIdx() {
    return {};
}

void ConstantTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, torchContext* context) {
    auto param = new MNN::BlobT;
    const auto output = node->output();
    const std::string& type = output->type()->str();
    if (type == "None") {
        return;
    }
    auto attr = node->attributeNames()[0];
    auto kind = node->kindOf(attr);
    switch (kind) {
        case torch::jit::AttributeKind::f:
            param->dataType = MNN::DataType_DT_FLOAT;
            param->dims.push_back(1);
            param->float32s.push_back(node->f(attr));
            break;
        case torch::jit::AttributeKind::i:
            param->dataType = MNN::DataType_DT_INT32;
            param->dims.push_back(1);
            param->int32s.push_back(node->i(attr));
            break;
        case torch::jit::AttributeKind::s:
            param->dataType = MNN::DataType_DT_STRING;
            param->dims.push_back(1);
            param->strings.push_back(node->s(attr));
            break;
        case torch::jit::AttributeKind::t: {
            auto tensor = node->t(attr);
            param->dataType = MNN::DataType_DT_FLOAT;
            param->float32s = std::move(getValue<float>(output, param->dims));
            break;
        }
        default:
            return;
    }
    dstOp->main.value = param;
}

REGISTER_CONVERTER(ConstantTorch, Constant);
