//
//  LayerNormTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2021/07/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(LayerNormTorch);

MNN::OpType LayerNormTorch::opType() {
    return MNN::OpType_LayerNorm;
}
MNN::OpParameter LayerNormTorch::type() {
    return MNN::OpParameter_LayerNorm;
}
std::vector<int> LayerNormTorch::inputTensorIdx() {
    return {0};
}

void LayerNormTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope) {
    auto param = new MNN::LayerNormT;
    const auto& inputs = node->inputs();
    const auto weight = inputs[2];
    const auto bias = inputs[3];
    const auto eps = inputs[4];
    param->epsilon = getValue<float>(eps);
    std::vector<int> shape;
    std::string opType = getRealOpType(node);
    if (opType == "group_norm") {
        param->group = getValue<int64_t>(inputs[1]);
        param->axis = {-1};
        // add scale op after layernorm
        {
            auto scaleName = dstOp->name + "/scale";
            int idx = scope->declareTensor(scaleName);
            std::unique_ptr<MNN::OpT> sclaeOp(new MNN::OpT);
            sclaeOp->name      = scaleName;
            sclaeOp->type      = MNN::OpType_Scale;
            sclaeOp->main.type = MNN::OpParameter_Scale;
            auto scale  = new MNN::ScaleT;
            scale->scaleData = getValue<float>(weight, shape);
            scale->biasData = getValue<float>(bias, shape);
            scale->channels = shape[0];
            sclaeOp->main.value = scale;
            sclaeOp->inputIndexes.push_back(idx);
            sclaeOp->outputIndexes.push_back(dstOp->outputIndexes[0]);
            dstOp->outputIndexes[0] = idx;
            scope->oplists().emplace_back(std::move(sclaeOp));
        }
    } else {
        auto norm_shape = getValue<std::vector<int64_t>>(inputs[1]);
        // TODO: convert norm_shape to axis
        param->axis = {-1};
        param->gamma = getValue<float>(weight, shape);
        param->beta = getValue<float>(bias, shape);
    }
    dstOp->main.value = param;
}

REGISTER_CONVERTER(LayerNormTorch, layer_norm);
REGISTER_CONVERTER(LayerNormTorch, group_norm);
