//
//  TopKV2Tflite.cpp
//  MNNConverter
//
//  Created by MNN on 2020/01/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "liteOpConverter.hpp"

DECLARE_OP_COVERTER(TopKV2Tflite);
MNN::OpType TopKV2Tflite::opType(bool quantizedModel) {
    return MNN::OpType_TopKV2;
}
MNN::OpParameter TopKV2Tflite::type(bool quantizedModel) {
    return MNN::OpParameter_TopKV2;
}

void TopKV2Tflite::run(MNN::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                         const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                         const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                         const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, bool quantizedModel){

  auto topkv2Param = new MNN::TopKV2T;
  topkv2Param->sorted = false;
  topkv2Param->T = MNN::DataType_DT_FLOAT;
  dstOp->main.value = topkv2Param;
}


using namespace tflite;
REGISTER_CONVERTER(TopKV2Tflite, BuiltinOperator_TOPK_V2);
