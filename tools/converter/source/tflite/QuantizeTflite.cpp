//
//  QuantizeTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2023/04/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "liteOpConverter.hpp"
#include "TfliteUtils.hpp"

DECLARE_OP_COVERTER(QuantizeTflite);
MNN::OpType QuantizeTflite::opType(int quantizedModel) {
    return MNN::OpType_Cast;
}
MNN::OpParameter QuantizeTflite::type(int quantizedModel) {
    return MNN::OpParameter_CastParam;
}

void QuantizeTflite::run(MNN::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                         const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                         const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                         const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, int quantizedModel){
    auto inputIndex = tfliteOp->inputs[0];
    auto outputIndex = tfliteOp->outputs[0];
    const auto& inputTensor = tfliteTensors[inputIndex];
    const auto& outputTensor = tfliteTensors[outputIndex];
#if 0
    auto extraOpParam = new MNN::ExtraT;
    extraOpParam->engine = "Tflite";
    extraOpParam->type = "Quantize";
    extraOpParam->attr.resize(2);
    extraOpParam->attr[0].reset(new MNN::AttributeT);
    extraOpParam->attr[1].reset(new MNN::AttributeT);
    extraOpParam->attr[0]->key = "scale";
    extraOpParam->attr[0]->f = outputTensor->quantization->scale[0];
    extraOpParam->attr[1]->key = "zero_point";
    extraOpParam->attr[1]->i = (int)outputTensor->quantization->zero_point[0];
    dstOp->main.value = extraOpParam;
#else
    // just add cast, quant/dequant is runtime insert
    auto param = new MNN::CastParamT;
    // other type to int8  => cast + quant
    if (outputTensor->type == tflite::TensorType_INT8) {
        param->srcT = TfliteDataTypeToMNN(inputTensor->type);
        param->dstT = MNN::DataType_DT_FLOAT;
    }
    // int8 to other type => dequant + cast
    if (inputTensor->type == tflite::TensorType_INT8) {
        param->srcT = MNN::DataType_DT_FLOAT;
        param->dstT = TfliteDataTypeToMNN(outputTensor->type);
    }
    dstOp->main.value = param;
#endif
}

using namespace tflite;
REGISTER_CONVERTER(QuantizeTflite, BuiltinOperator_QUANTIZE);
