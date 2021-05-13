//
//  UnaryTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2019/11/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "liteOpConverter.hpp"

DECLARE_OP_COVERTER(UnaryTflite);
MNN::OpType UnaryTflite::opType(bool quantizedModel) {
    return MNN::OpType_UnaryOp;
}
MNN::OpParameter UnaryTflite::type(bool quantizedModel) {
    return MNN::OpParameter_UnaryOp;
}

void UnaryTflite::run(MNN::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                         const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                         const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                         const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, bool quantizedModel){
  auto param = new MNN::UnaryOpT;
  switch(tfliteOpSet[tfliteOp->opcode_index]->builtin_code){
    case tflite::BuiltinOperator_FLOOR:{
      param->opType=MNN::UnaryOpOperation_FLOOR;
      break;
    }
    case tflite::BuiltinOperator_SQUARE:{
      param->opType=MNN::UnaryOpOperation_SQUARE;
      break;
    }
    case tflite::BuiltinOperator_RSQRT:{
      param->opType=MNN::UnaryOpOperation_RSQRT;
      break;
    }
    case tflite::BuiltinOperator_EXP:{
      param->opType=MNN::UnaryOpOperation_EXP;
      break;
    }
    case tflite::BuiltinOperator_NEG:{
      param->opType=MNN::UnaryOpOperation_NEG;
      break;
    }
    case tflite::BuiltinOperator_SQRT:{
      param->opType=MNN::UnaryOpOperation_SQRT;
      break;
    }
    case tflite::BuiltinOperator_LOG:{
      param->opType=MNN::UnaryOpOperation_LOG;
      break;
    }
    case tflite::BuiltinOperator_SIN:{
      param->opType=MNN::UnaryOpOperation_SIN;
      break;
    }
    case tflite::BuiltinOperator_HARD_SWISH:{
      param->opType=MNN::UnaryOpOperation_HARDSWISH;
      break;
    }
    default:{
        LOG(ERROR) << "MNN Converter Not "
                      "Supported!!! UnaryOp: "
                   << tfliteOpSet[tfliteOp->opcode_index]->custom_code;
    }
  }
  dstOp->main.value = param;
}


using namespace tflite;
REGISTER_CONVERTER(UnaryTflite, BuiltinOperator_FLOOR);
REGISTER_CONVERTER(UnaryTflite, BuiltinOperator_SQUARE);
REGISTER_CONVERTER(UnaryTflite, BuiltinOperator_RSQRT);
REGISTER_CONVERTER(UnaryTflite, BuiltinOperator_EXP);
REGISTER_CONVERTER(UnaryTflite, BuiltinOperator_NEG);
REGISTER_CONVERTER(UnaryTflite, BuiltinOperator_SQRT);
REGISTER_CONVERTER(UnaryTflite, BuiltinOperator_LOG);
REGISTER_CONVERTER(UnaryTflite, BuiltinOperator_SIN);
REGISTER_CONVERTER(UnaryTflite, BuiltinOperator_HARD_SWISH);
