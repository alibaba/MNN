//
//  CumSumOnnx.cpp
//  MNN
//
//  Created by MNN on 2021/06/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(CumSumOnnx);

MNN::OpType CumSumOnnx::opType() {
    return MNN::OpType_CumSum;
}
MNN::OpParameter CumSumOnnx::type() {
    return MNN::OpParameter_CumSum;
}

void CumSumOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                     OnnxScope* scope) {
    auto param = new MNN::CumSumT;
    param->exclusive = param->reverse = false;
    for (const auto& attr : onnxNode->attribute()) {
        if (attr.name() == "exclusive") {
            param->exclusive = attr.i();
        } else if (attr.name() == "reverse") {
            param->reverse= attr.i();
        }
    }
    dstOp->main.value = param;
}

REGISTER_CONVERTER(CumSumOnnx, CumSum);
