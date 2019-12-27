//
//  ReluTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2019/11/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "liteOpConverter.hpp"

DECLARE_OP_COVERTER(ReluTflite);
MNN::OpType ReluTflite::opType(bool quantizedModel) {
    return MNN::OpType_ReLU;
}
MNN::OpParameter ReluTflite::type(bool quantizedModel) {
    return MNN::OpParameter_Relu;
}

void ReluTflite::run(MNN::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                         const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                         const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                         const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, bool quantizedModel){
  auto Relu   = new MNN::ReluT;
  Relu->slope = 0.0f;
  dstOp->main.value = Relu;
}

DECLARE_OP_COVERTER(LeakyReluTflite);
MNN::OpType LeakyReluTflite::opType(bool quantizedModel) {
    return MNN::OpType_ReLU;
}
MNN::OpParameter LeakyReluTflite::type(bool quantizedModel) {
    return MNN::OpParameter_Relu;
}

void LeakyReluTflite::run(MNN::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                         const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                         const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                         const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, bool quantizedModel){
  auto Relu = new MNN::ReluT;
  auto opt = tfliteOp->builtin_options.AsLeakyReluOptions();
  Relu->slope = opt->alpha;
  dstOp->main.value = Relu;
}

DECLARE_OP_COVERTER(Relu6Tflite);
MNN::OpType Relu6Tflite::opType(bool quantizedModel) {
    return MNN::OpType_ReLU6;
}
MNN::OpParameter Relu6Tflite::type(bool quantizedModel) {
    return MNN::OpParameter_Relu6;
}

void Relu6Tflite::run(MNN::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                         const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                         const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                         const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, bool quantizedModel){
  auto relu6   = new MNN::Relu6T;
  relu6->slope = 0.0f;
  dstOp->main.value = relu6;
}

using namespace tflite;
REGISTER_CONVERTER(ReluTflite, BuiltinOperator_RELU);
REGISTER_CONVERTER(LeakyReluTflite, BuiltinOperator_LEAKY_RELU);
REGISTER_CONVERTER(Relu6Tflite, BuiltinOperator_RELU6);
