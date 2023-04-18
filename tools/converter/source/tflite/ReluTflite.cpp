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
MNN::OpType ReluTflite::opType(int quantizedModel) {
    return MNN::OpType_ReLU;
}
MNN::OpParameter ReluTflite::type(int quantizedModel) {
    return MNN::OpParameter_Relu;
}

void ReluTflite::run(MNN::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                         const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                         const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                         const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, int quantizedModel){
  auto Relu   = new MNN::ReluT;
  Relu->slope = 0.0f;
  dstOp->main.value = Relu;
}

DECLARE_OP_COVERTER(LeakyReluTflite);
MNN::OpType LeakyReluTflite::opType(int quantizedModel) {
    return MNN::OpType_ReLU;
}
MNN::OpParameter LeakyReluTflite::type(int quantizedModel) {
    return MNN::OpParameter_Relu;
}

void LeakyReluTflite::run(MNN::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                         const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                         const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                         const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, int quantizedModel){
  auto Relu = new MNN::ReluT;
  auto opt = tfliteOp->builtin_options.AsLeakyReluOptions();
  Relu->slope = opt->alpha;
  dstOp->main.value = Relu;
}

DECLARE_OP_COVERTER(Relu6Tflite);
MNN::OpType Relu6Tflite::opType(int quantizedModel) {
    return MNN::OpType_ReLU6;
}
MNN::OpParameter Relu6Tflite::type(int quantizedModel) {
    return MNN::OpParameter_Relu6;
}

void Relu6Tflite::run(MNN::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                         const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                         const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                         const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, int quantizedModel){
  auto relu6   = new MNN::Relu6T;
  dstOp->main.value = relu6;
}
DECLARE_OP_COVERTER(PreluTflite);
MNN::OpType PreluTflite::opType(int quantizedModel) {
    return MNN::OpType_Extra;
}
MNN::OpParameter PreluTflite::type(int quantizedModel) {
    return MNN::OpParameter_Extra;
}

void PreluTflite::run(MNN::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                         const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                         const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                         const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, int quantizedModel){
    dstOp->main.value = new MNN::ExtraT;
    auto dstP = dstOp->main.AsExtra();
    dstP->engine = "Tflite";
    dstP->type = "PRELU";
}

using namespace tflite;
REGISTER_CONVERTER(ReluTflite, BuiltinOperator_RELU);
REGISTER_CONVERTER(LeakyReluTflite, BuiltinOperator_LEAKY_RELU);
REGISTER_CONVERTER(Relu6Tflite, BuiltinOperator_RELU6);
REGISTER_CONVERTER(PreluTflite, BuiltinOperator_PRELU);
