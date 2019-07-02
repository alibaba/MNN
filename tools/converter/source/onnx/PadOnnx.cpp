//
//  PadOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/06/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(PadOnnx);

MNN::OpType PadOnnx::opType() {
    return MNN::OpType_Padding;
}
MNN::OpParameter PadOnnx::type() {
    return MNN::OpParameter_Blob;
}

void PadOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                  std::vector<const onnx::TensorProto*> initializers) {
    auto para        = new MNN::BlobT;
    para->dataFormat = MNN::MNN_DATA_FORMAT_NHWC;
    para->dataType   = MNN::DataType_DT_INT32;
    for (int i = 0; i < onnxNode->attribute_size(); ++i) {
        const auto& attributeProto = onnxNode->attribute(i);
        const auto& attributeName  = attributeProto.name();
        if (attributeName == "pads") {
            para->int32s.resize(attributeProto.ints_size());
            para->dims = {(int)para->int32s.size() / 2, 2};
            for (int i = 0; i < para->int32s.size(); ++i) {
                para->int32s[i] = attributeProto.ints(i);
            }
        }
    }
    dstOp->main.value = para;
}

REGISTER_CONVERTER(PadOnnx, Pad);
