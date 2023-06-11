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

MNN::OpType CastTflite::opType(int quantizedModel){
    return MNN::OpType_Cast;
}

MNN::OpParameter CastTflite::type(int quantizedModel){
    return MNN::OpParameter_CastParam;
}

void CastTflite::run(MNN::OpT *dstOp, const std::unique_ptr<tflite::OperatorT> &tfliteOp, const std::vector<std::unique_ptr<tflite::TensorT> > &tfliteTensors, const std::vector<std::unique_ptr<tflite::BufferT> > &tfliteModelBuffer, const std::vector<std::unique_ptr<tflite::OperatorCodeT> > &tfliteOpSet, int quantizedModel){
    
    auto param = new MNN::CastParamT;
    
    auto tfliteParam = tfliteOp->builtin_options.AsCastOptions();
    if (nullptr != tfliteParam) {
        param->srcT = TfliteDataTypeToMNN(tfliteParam->in_data_type);
        param->dstT = TfliteDataTypeToMNN(tfliteParam->out_data_type);
    } else {
        // Find type from tensor
        auto output = tfliteTensors[tfliteOp->outputs[0]].get();
        param->dstT = TfliteDataTypeToMNN(output->type);
        param->srcT = TfliteDataTypeToMNN(tfliteTensors[tfliteOp->inputs[0]]->type);
    }
    dstOp->main.value = param;
}

using namespace tflite;
REGISTER_CONVERTER(CastTflite, BuiltinOperator_CAST);
