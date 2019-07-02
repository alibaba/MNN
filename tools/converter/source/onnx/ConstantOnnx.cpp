//
//  ConstantOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/05/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(ConstantOnnx);

MNN::OpType ConstantOnnx::opType() {
    return MNN::OpType_Const;
}
MNN::OpParameter ConstantOnnx::type() {
    return MNN::OpParameter_Blob;
}

void ConstantOnnx::run(MNN::OpT *dstOp, const onnx::NodeProto *onnxNode,
                       std::vector<const onnx::TensorProto *> initializers) {

    const onnx::TensorProto *constantTp;
    for (int i = 0; i < onnxNode->attribute_size(); ++i) {
        const auto &attributeProto = onnxNode->attribute(i);
        const auto &attributeName  = attributeProto.name();
        if (attributeName == "value") {
            constantTp = &attributeProto.t();
        }
    }
    if (!constantTp) {
        DLOG(FATAL) << "Constant No TensorProto Data!!!==> " << dstOp->name;
    }
    auto constantParam = convertTensorToBlob(constantTp);
    dstOp->main.value = constantParam;
    DCHECK(onnxNode->input_size() == 0) << "Constant Should Not Have Input!!! ===> " << dstOp->name;
}

REGISTER_CONVERTER(ConstantOnnx, Constant);
