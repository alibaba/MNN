//
//  DepthToSpaceOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/06/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "onnxOpConverter.hpp"
#include <MNN/MNNDefine.h>

DECLARE_OP_CONVERTER(DepthToSpaceOnnx);

MNN::OpType DepthToSpaceOnnx::opType() {
    return MNN::OpType_DepthToSpace;
}

MNN::OpParameter DepthToSpaceOnnx::type() {
    return MNN::OpParameter_DepthSpaceParam;
}

void DepthToSpaceOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode, OnnxScope* scope) {
    auto spaceToDepthParam = new MNN::DepthSpaceParamT;
    
    const auto attrSize = onnxNode->attribute_size();
    for (int i = 0; i < attrSize; ++i) {
        const auto& attributeProto = onnxNode->attribute(i);
        const auto& attributeName  = attributeProto.name();
        if (attributeName == "blocksize") {
            spaceToDepthParam->blockSize = (int)attributeProto.i();
        } else if (attributeName == "mode") {
            std::map<const std::string, MNN::DepthToSpaceMode> strToMode = {
                {"DCR", MNN::DepthToSpaceMode_DCR}, {"CRD", MNN::DepthToSpaceMode_CRD}
            };
            const std::string& modeStr = attributeProto.s();
            if (strToMode.find(modeStr) != strToMode.end()) {
                spaceToDepthParam->mode = strToMode[modeStr];
            } else {
                MNN_ERROR("ONNX DepthToSpace mode [%s] is currently not supported.\n", modeStr.c_str());
            }
        }
    }
    
    dstOp->main.value = spaceToDepthParam;
}

REGISTER_CONVERTER(DepthToSpaceOnnx, DepthToSpace);
