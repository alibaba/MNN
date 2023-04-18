//
//  TanHTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2020/01/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "liteOpConverter.hpp"

DECLARE_OP_COVERTER(TanHTflite);
MNN::OpType TanHTflite::opType(int quantizedModel) {
    return MNN::OpType_TanH;
}
MNN::OpParameter TanHTflite::type(int quantizedModel) {
    return MNN::OpParameter_NONE;
}

void TanHTflite::run(MNN::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                         const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                         const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                         const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, int quantizedModel){
  dstOp->main.value = nullptr;
}


using namespace tflite;
REGISTER_CONVERTER(TanHTflite, BuiltinOperator_TANH);
