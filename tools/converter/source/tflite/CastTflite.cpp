//
//  CastTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2020/05/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "liteOpConverter.hpp"
#include "TfliteUtils.hpp"

DECLARE_OP_COVERTER(CastTflite);

MNN::OpType CastTflite::opType(bool quantizedModel){
    return MNN::OpType_Cast;
}

MNN::OpParameter CastTflite::type(bool quantizedModel){
    return MNN::OpParameter_CastParam;
}

void CastTflite::run(MNN::OpT *dstOp, const std::unique_ptr<tflite::OperatorT> &tfliteOp, const std::vector<std::unique_ptr<tflite::TensorT> > &tfliteTensors, const std::vector<std::unique_ptr<tflite::BufferT> > &tfliteModelBuffer, const std::vector<std::unique_ptr<tflite::OperatorCodeT> > &tfliteOpSet, bool quantizedModel){
    
    auto param = new MNN::CastParamT;
    
    auto tfliteParam = tfliteOp->builtin_options.AsCastOptions();
    
    param->srcT = TfliteDataTypeToMNN(tfliteParam->in_data_type);
    param->dstT = TfliteDataTypeToMNN(tfliteParam->out_data_type);
    dstOp->main.value = param;
    
}

using namespace tflite;
REGISTER_CONVERTER(CastTflite, BuiltinOperator_CAST);
