//
//  LogisticTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "liteOpConverter.hpp"

DECLARE_OP_COVERTER(LogisticTflite);

MNN::OpType LogisticTflite::opType(int quantizedModel) {
    if (quantizedModel)
        return MNN::OpType_QuantizedLogistic;
    return MNN::OpType_Sigmoid;
}
MNN::OpParameter LogisticTflite::type(int quantizedModel) {
    if (quantizedModel)
        return MNN::OpParameter_QuantizedLogistic;
    return MNN::OpParameter_NONE;
}

void LogisticTflite::run(MNN::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                         const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                         const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                         const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, int quantizedModel) {
    if (quantizedModel) {
        auto LogisticParam = new MNN::QuantizedLogisticT;

        const int inputIndex                          = tfliteOp->inputs[0];
        const auto& inputTensor                       = tfliteTensors[inputIndex];
        LogisticParam->inputQuantizedParam            = std::unique_ptr<MNN::QuantizedParamT>(new MNN::QuantizedParamT);
        LogisticParam->inputQuantizedParam->zeroPoint = inputTensor->quantization->zero_point[0];
        LogisticParam->inputQuantizedParam->scale     = inputTensor->quantization->scale[0];

        const int outputIndex               = tfliteOp->outputs[0];
        const auto& outputTensor            = tfliteTensors[outputIndex];
        LogisticParam->outputQuantizedParam = std::unique_ptr<MNN::QuantizedParamT>(new MNN::QuantizedParamT);
        LogisticParam->outputQuantizedParam->zeroPoint = outputTensor->quantization->zero_point[0];
        LogisticParam->outputQuantizedParam->scale     = outputTensor->quantization->scale[0];

        dstOp->main.value = LogisticParam;
    } else {
        dstOp->main.value = nullptr;
    }
}

using namespace tflite;
REGISTER_CONVERTER(LogisticTflite, BuiltinOperator_LOGISTIC);
