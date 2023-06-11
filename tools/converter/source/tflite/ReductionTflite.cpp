//
//  ReductionTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2019/12/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "liteOpConverter.hpp"

DECLARE_OP_COVERTER(ReductionTflite);
MNN::OpType ReductionTflite::opType(int quantizedModel) {
    return MNN::OpType_Reduction;
}
MNN::OpParameter ReductionTflite::type(int quantizedModel) {
    return MNN::OpParameter_ReductionParam;
}

void ReductionTflite::run(MNN::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                         const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                         const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                         const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, int quantizedModel){
  auto param = new MNN::ReductionParamT;
  auto opt = tfliteOp->builtin_options.AsReducerOptions();
  param->keepDims=opt->keep_dims;
#ifdef TF_CONVERT_ORIGIN
  const int input1Idx                       = tfliteOp->inputs[1];
  const auto& input1Tensor                  = tfliteTensors[input1Idx];
  if(input1Tensor.is_variable == false){
    auto buffer1Idx=input1Tensor.buffer;
    auto buffer1=tfliteModelBuffer[buffer1Idx];
    auto shape=input1Tensor.shape;
    param->dim.resize(shape.size());
    for(decltype(shape.size()) x=0;x<shape.size();x++){
      param->dim[x]=shape[x];
    }
  }
#endif
  switch(tfliteOpSet[tfliteOp->opcode_index]->builtin_code){
    case tflite::BuiltinOperator_SUM:{
      param->operation=MNN::ReductionType_SUM;
      break;
    }
    case tflite::BuiltinOperator_REDUCE_MAX:{
      param->operation=MNN::ReductionType_MAXIMUM;
      break;
    }
    case tflite::BuiltinOperator_REDUCE_MIN:{
      param->operation=MNN::ReductionType_MINIMUM;
      break;
    }
    case tflite::BuiltinOperator_REDUCE_ANY:{
      param->operation=MNN::ReductionType_ANY;
      break;
    }
    case tflite::BuiltinOperator_REDUCE_PROD:{
      param->operation=MNN::ReductionType_PROD;
      break;
    }
    case tflite::BuiltinOperator_MEAN:{
      param->operation=MNN::ReductionType_MEAN;
      break;
    }
    default:{
        LOG(ERROR) << "MNN Converter Not "
                      "Supported!!! Reduction Op: "
                   << tfliteOpSet[tfliteOp->opcode_index]->custom_code;
    }
  }
  dstOp->main.value = param;
}
using namespace tflite;
REGISTER_CONVERTER(ReductionTflite,BuiltinOperator_SUM);
REGISTER_CONVERTER(ReductionTflite,BuiltinOperator_REDUCE_MAX);
REGISTER_CONVERTER(ReductionTflite,BuiltinOperator_REDUCE_MIN);
REGISTER_CONVERTER(ReductionTflite,BuiltinOperator_REDUCE_ANY);
REGISTER_CONVERTER(ReductionTflite,BuiltinOperator_REDUCE_PROD);
REGISTER_CONVERTER(ReductionTflite,BuiltinOperator_MEAN);
