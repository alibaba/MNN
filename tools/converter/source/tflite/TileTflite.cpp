//
//  TileTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2019/11/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "liteOpConverter.hpp"

DECLARE_OP_COVERTER(TileTflite);
MNN::OpType TileTflite::opType(int quantizedModel) {
    return MNN::OpType_Tile;
}
MNN::OpParameter TileTflite::type(int quantizedModel) {
    return MNN::OpParameter_NONE;
}

void TileTflite::run(MNN::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                         const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                         const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                         const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, int quantizedModel){
  dstOp->main.value = nullptr;
}


using namespace tflite;
REGISTER_CONVERTER(TileTflite, BuiltinOperator_TILE);
