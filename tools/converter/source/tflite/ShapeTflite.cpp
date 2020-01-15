//
//  ShapeTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2020/01/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "liteOpConverter.hpp"

DECLARE_OP_COVERTER(ShapeTflite);
MNN::OpType ShapeTflite::opType(bool quantizedModel) {
    return MNN::OpType_Shape;
}
MNN::OpParameter ShapeTflite::type(bool quantizedModel) {
    return MNN::OpParameter_NONE;
}

void ShapeTflite::run(MNN::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                         const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                         const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                         const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, bool quantizedModel){
  dstOp->main.value = nullptr;
}


using namespace tflite;
REGISTER_CONVERTER(ShapeTflite, BuiltinOperator_SHAPE);
