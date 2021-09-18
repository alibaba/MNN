//
//  CastOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/10/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(CastOnnx);

MNN::OpType CastOnnx::opType() {
    return MNN::OpType_Cast;
}
MNN::OpParameter CastOnnx::type() {
    return MNN::OpParameter_CastParam;
}

void CastOnnx::run(MNN::OpT *dstOp, const onnx::NodeProto *onnxNode,
                   OnnxScope* scope) {
    std::unique_ptr<MNN::CastParamT> castParam(new MNN::CastParamT);

    // not to use srcT parameter!
    castParam->srcT = MNN::DataType_MAX;

    ::onnx::TensorProto_DataType castTo = ::onnx::TensorProto_DataType_UNDEFINED;
    const int attrSize                  = onnxNode->attribute_size();
    for (int i = 0; i < attrSize; ++i) {
        const auto &attributeProto = onnxNode->attribute(i);
        const auto &attributeName  = attributeProto.name();
        if (attributeName == "to") {
            castTo = static_cast<::onnx::TensorProto_DataType>(attributeProto.i());
        }
    }

    castParam->dstT   = onnxOpConverter::convertDataType(castTo);
    dstOp->main.value = castParam.release();
}

REGISTER_CONVERTER(CastOnnx, Cast);
