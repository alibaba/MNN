//
//  RangeTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2021/09/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "liteOpConverter.hpp"

DECLARE_OP_COVERTER(RangeTflite);
MNN::OpType RangeTflite::opType(int quantizedModel) {
    return MNN::OpType_Range;
}
MNN::OpParameter RangeTflite::type(int quantizedModel) {
    return MNN::OpParameter_NONE;
}

void RangeTflite::run(MNN::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                         const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                         const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                         const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, int quantizedModel){
    // Do nothing
    
}


using namespace tflite;
REGISTER_CONVERTER(RangeTflite, BuiltinOperator_RANGE);
