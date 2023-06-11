//
//  SliceTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2021/09/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "liteOpConverter.hpp"

DECLARE_OP_COVERTER(SliceTflite);
MNN::OpType SliceTflite::opType(int quantizedModel) {
    return MNN::OpType_SliceTf;
}
MNN::OpParameter SliceTflite::type(int quantizedModel) {
    return MNN::OpParameter_NONE;
}

void SliceTflite::run(MNN::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                         const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                         const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                         const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, int quantizedModel){
    // Do nothing
}


using namespace tflite;
REGISTER_CONVERTER(SliceTflite, BuiltinOperator_SLICE);
