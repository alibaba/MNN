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

MNN::OpType PadTflite::opType(int quantizedModel) {
    return MNN::OpType_Padding;
}
MNN::OpParameter PadTflite::type(int quantizedModel) {
    return MNN::OpParameter_NONE;
}
void PadTflite::run(MNN::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                       const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                       const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                       const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, int quantizedModel) {
    auto padparm = new MNN::PadParamT;
    switch(tfliteOpSet[tfliteOp->opcode_index]->builtin_code){
      case BuiltinOperator_PADV2:
      case BuiltinOperator_PAD:{
        padparm->mode = MNN::PadValueMode_CONSTANT;
        break;
      }
      case BuiltinOperator_MIRROR_PAD:{
        auto opt=tfliteOp->builtin_options.AsMirrorPadOptions();
        switch(opt->mode){
          case MirrorPadMode_REFLECT:{
            padparm->mode = MNN::PadValueMode_REFLECT;
            break;
          }
          case MirrorPadMode_SYMMETRIC:{
            padparm->mode = MNN::PadValueMode_SYMMETRIC;
            break;
          }
          default:{
            DCHECK(false) << "Unknown Pad Value Mode!";
          }
        }
        break;
      }
      default:{
        DCHECK(false) << "Unknown Pad Operator";
      }
    }
    dstOp->main.value = padparm;

}

REGISTER_CONVERTER(PadTflite, BuiltinOperator_PAD);
REGISTER_CONVERTER(PadTflite, BuiltinOperator_PADV2);
REGISTER_CONVERTER(PadTflite,BuiltinOperator_MIRROR_PAD);
