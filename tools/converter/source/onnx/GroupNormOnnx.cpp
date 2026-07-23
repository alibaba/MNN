//
//  GroupNormOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2023/09/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "onnxOpConverter.hpp"
#include <MNN/MNNDefine.h>

DECLARE_OP_CONVERTER(GroupNormOnnx);

MNN::OpType GroupNormOnnx::opType(){
    return MNN::OpType_GroupNorm;
}

MNN::OpParameter GroupNormOnnx::type(){
    return MNN::OpParameter_GroupNorm;
}

static void _run(MNN::OpT *dstOp, const onnx::NodeProto *onnxNode, OnnxScope* scope){
    auto groupnormT              = new MNN::GroupNormT;
    float epsilon = 1e-10;
    int bSwish = 0;
    std::vector<float> gammaData;
    std::vector<float> betaData;
    for (int i = 0; i < onnxNode->attribute_size(); ++i) {
        const auto& attributeProto = onnxNode->attribute(i);
        const auto& attributeName  = attributeProto.name();

        if (attributeName == "bSwish") {
            // Determine whether do swish activation
            bSwish = attributeProto.i();
        }
        if (attributeName == "epsilon") {
            epsilon = attributeProto.f();
        }
	if (attributeName == "gamma") {
            auto vec = attributeProto.floats();
            gammaData.assign(vec.begin(), vec.end());
        }
        if (attributeName == "beta") {
            auto vec = attributeProto.floats();
            betaData.assign(vec.begin(), vec.end());
        }
    }
    groupnormT->gamma = gammaData;
    groupnormT->beta = betaData;
    groupnormT->bSwish = bSwish;
    groupnormT->epsilon = epsilon;
    dstOp->main.value = groupnormT;
}

void GroupNormOnnx::run(MNN::OpT *dstOp, const onnx::NodeProto *onnxNode, OnnxScope* scope){
    _run(dstOp, onnxNode, scope);
}

REGISTER_CONVERTER(GroupNormOnnx, GroupNorm);
