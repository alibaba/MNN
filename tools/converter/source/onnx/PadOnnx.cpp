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
            const int size = attributeProto.ints_size();
            para->int32s.resize(size);
            para->dims = {size};
            for (int i = 0; i < size / 2; ++i) {
                para->int32s[i * 2] = attributeProto.ints(i);
                para->int32s[i * 2 + 1] = attributeProto.ints(i + size / 2);
            }
        }
    }
    dstOp->main.value = para;
}

REGISTER_CONVERTER(PadOnnx, Pad);
