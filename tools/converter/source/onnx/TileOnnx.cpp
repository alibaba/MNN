//
//  TileOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/07/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(TileOnnx);

MNN::OpType TileOnnx::opType() {
    return MNN::OpType_Tile;
}

MNN::OpParameter TileOnnx::type() {
    return MNN::OpParameter_NONE;
}

void TileOnnx::run(MNN::OpT *dstOp, const onnx::NodeProto *onnxNode,
                   OnnxScope* scope) {
    return;
}

REGISTER_CONVERTER(TileOnnx, Tile);
