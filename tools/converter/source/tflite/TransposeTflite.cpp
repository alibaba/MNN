//
//  TransposeTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2019/12/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "liteOpConverter.hpp"

DECLARE_OP_COVERTER(TransposeTflite);
MNN::OpType TransposeTflite::opType(bool quantizedModel) {
    return MNN::OpType_Transpose;
}
MNN::OpParameter TransposeTflite::type(bool quantizedModel) {
    return MNN::OpParameter_Transpose;
}
void TransposeTflite::run(MNN::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                         const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                         const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                         const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, bool quantizedModel){
  auto param = new MNN::TransposeT;
  auto tfliteSoftmaxOption = tfliteOp->builtin_options.AsTransposeOptions();

  dstOp->main.value = param;
}


using namespace tflite;
REGISTER_CONVERTER(TransposeTflite, BuiltinOperator_TRANSPOSE);
