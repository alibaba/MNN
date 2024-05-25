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

void ConstantOnnx::run(MNN::OpT *dstOp, const onnx::NodeProto *onnxNode, OnnxScope* scope) {
    int type; // 0: TensorProto, 1: float, 2: floats, 3: int, 4: ints
    const onnx::TensorProto *constantTp;
    float value_float;
    std::vector<float> value_floats;
    int value_int;
    std::vector<int> value_ints;
    for (int i = 0; i < onnxNode->attribute_size(); ++i) {
        const auto &attributeProto = onnxNode->attribute(i);
        const auto &attributeName  = attributeProto.name();
        if (attributeName == "value") {
            constantTp = &attributeProto.t();
            type = 0;
        } else if (attributeName == "value_float") {
            value_float = attributeProto.f();
            type = 1;
        } else if (attributeName == "value_floats") {
            auto vec = attributeProto.floats();
            value_floats.assign(vec.begin(), vec.end());
            type = 2;
        } else if (attributeName == "value_int") {
            value_int = attributeProto.i();
            type = 3;
        } else if (attributeName == "value_ints") {
            auto vec = attributeProto.ints();
            value_ints.assign(vec.begin(), vec.end());
            type = 4;
        } else if (attributeName == "value_string" || attributeName == "value_strings") {
            DLOG(FATAL) << "Not support %s attr!!!==> " << dstOp->name;
            return;
        }
    }
    if (type == 0) {
        dstOp->main.value = convertTensorToBlob(constantTp, scope->mModelDir, dstOp);
    } else {
        auto blob = new MNN::BlobT;
        blob->dataFormat = MNN::MNN_DATA_FORMAT_NCHW;
        if (type == 1) {
            blob->dataType = MNN::DataType_DT_FLOAT;
            blob->float32s.push_back(value_float);
            blob->dims.assign({1});
        } else if (type == 2) {
            blob->dataType = MNN::DataType_DT_FLOAT;
            blob->float32s.assign(value_floats.begin(), value_floats.end());
            blob->dims.assign({(int)value_floats.size()});
        } else if (type == 3) {
            blob->dataType = MNN::DataType_DT_INT32;
            blob->int32s.push_back(value_int);
            blob->dims.assign({1});
        } else {
            blob->dataType = MNN::DataType_DT_INT32;
            blob->int32s.assign(value_ints.begin(), value_ints.end());
            blob->dims.assign({(int)value_ints.size()});
        }
        dstOp->main.value = blob;
    }
    DCHECK(onnxNode->input_size() == 0) << "Constant Should Not Have Input!!! ===> " << dstOp->name;
}

REGISTER_CONVERTER(ConstantOnnx, Constant);
