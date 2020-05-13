//
//  CastTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2020/5/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "liteOpConverter.hpp"

DECLARE_OP_COVERTER(CastTflite);

static MNN::DataType _convertType(tflite::TensorType type) {
    if (type == tflite::TensorType_FLOAT32) {
        return MNN::DataType_DT_FLOAT;
    }
    if (type == tflite::TensorType_INT8) {
        return MNN::DataType_DT_INT8;
    }
    if (type == tflite::TensorType_UINT8) {
        return MNN::DataType_DT_UINT8;
    }
    if (type == tflite::TensorType_INT32) {
        return MNN::DataType_DT_INT32;
    }
    return MNN::DataType_DT_INVALID;
}

MNN::OpType CastTflite::opType(bool quantizedModel){
    return MNN::OpType_Cast;
}

MNN::OpParameter CastTflite::type(bool quantizedModel){
    return MNN::OpParameter_CastParam;
}

void CastTflite::run(MNN::OpT *dstOp, const std::unique_ptr<tflite::OperatorT> &tfliteOp, const std::vector<std::unique_ptr<tflite::TensorT> > &tfliteTensors, const std::vector<std::unique_ptr<tflite::BufferT> > &tfliteModelBuffer, const std::vector<std::unique_ptr<tflite::OperatorCodeT> > &tfliteOpSet, bool quantizedModel){
    
    auto param = new MNN::CastParamT;
    
    auto tfliteParam = tfliteOp->builtin_options.AsCastOptions();
    
    param->srcT = _convertType(tfliteParam->in_data_type);
    param->dstT = _convertType(tfliteParam->out_data_type);
    dstOp->main.value = param;
    
}

using namespace tflite;
REGISTER_CONVERTER(CastTflite, BuiltinOperator_CAST);
