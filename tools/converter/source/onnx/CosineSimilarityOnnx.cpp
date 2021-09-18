//
//  CosineSimilarityOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/07/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(CosineSimilarityOnnx);

MNN::OpType CosineSimilarityOnnx::opType() {
    return MNN::OpType_CosineSimilarity;
}

MNN::OpParameter CosineSimilarityOnnx::type() {
    return MNN::OpParameter_NONE;
}

void CosineSimilarityOnnx::run(MNN::OpT *dstOp, const onnx::NodeProto *onnxNode,
                               OnnxScope* scope) {
    std::string type;
    for (int i=0; i<onnxNode->attribute_size(); ++i) {
        auto att = onnxNode->attribute(i);
        if ("operator" == att.name()) {
            type = att.s();
            break;
        }
    }
    DCHECK(type == "cosine_similarity") << " NOT SUPPPRT";
    return;
}

REGISTER_CONVERTER(CosineSimilarityOnnx, ATen);
