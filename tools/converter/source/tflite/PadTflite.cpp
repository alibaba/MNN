//
//  PadTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2019/11/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "liteOpConverter.hpp"

using namespace tflite;
DECLARE_OP_COVERTER(PadTflite);

MNN::OpType PadTflite::opType(bool quantizedModel) {
    return MNN::OpType_Padding;
}
MNN::OpParameter PadTflite::type(bool quantizedModel) {
    return MNN::OpParameter_NONE;
}
void PadTflite::run(MNN::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                       const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                       const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                       const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, bool quantizedModel) {

}

REGISTER_CONVERTER(PadTflite, BuiltinOperator_PAD);
