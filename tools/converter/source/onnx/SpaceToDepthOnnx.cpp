//
//  SpaceToDepthOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/06/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(SpaceToDepthOnnx);

MNN::OpType SpaceToDepthOnnx::opType() {
    return MNN::OpType_SpaceToDepth;
}

MNN::OpParameter SpaceToDepthOnnx::type() {
    return MNN::OpParameter_DepthSpaceParam;
}

void SpaceToDepthOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
OnnxScope* scope) {
    
    auto spaceToDepthParam = new MNN::DepthSpaceParamT;
    
    const auto attrSize = onnxNode->attribute_size();
    for (int i = 0; i < attrSize; ++i) {
        const auto& attributeProto = onnxNode->attribute(i);
        const auto& attributeName  = attributeProto.name();
        if (attributeName == "blocksize") {
            spaceToDepthParam->blockSize = (int)attributeProto.i();
        }
    }
    
    dstOp->main.value = spaceToDepthParam;
}

REGISTER_CONVERTER(SpaceToDepthOnnx, SpaceToDepth);
