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

MNN::OpType LogisticTflite::opType(bool quantizedModel) {
    DCHECK(quantizedModel) << "LogisticTflite TODO(float)";
    if (quantizedModel)
        return MNN::OpType_QuantizedLogistic;
    return MNN::OpType_QuantizedLogistic;
}
MNN::OpParameter LogisticTflite::type(bool quantizedModel) {
    DCHECK(quantizedModel) << "LogisticTflite TODO(float)";
    if (quantizedModel)
        return MNN::OpParameter_QuantizedLogistic;
    return MNN::OpParameter_QuantizedLogistic;
}

void LogisticTflite::run(MNN::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                         const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                         const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                         const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, bool quantizedModel) {
    DCHECK(quantizedModel) << "LogisticTflite TODO(float)";
    auto LogisticParam = new MNN::QuantizedLogisticT;

    const int inputIndex                          = tfliteOp->inputs[0];
    const auto& inputTensor                       = tfliteTensors[inputIndex];
    LogisticParam->inputQuantizedParam            = std::unique_ptr<MNN::QuantizedParamT>(new MNN::QuantizedParamT);
    LogisticParam->inputQuantizedParam->zeroPoint = inputTensor->quantization->zeroPoint[0];
    LogisticParam->inputQuantizedParam->scale     = inputTensor->quantization->scale[0];

    const int outputIndex                          = tfliteOp->outputs[0];
    const auto& outputTensor                       = tfliteTensors[outputIndex];
    LogisticParam->outputQuantizedParam            = std::unique_ptr<MNN::QuantizedParamT>(new MNN::QuantizedParamT);
    LogisticParam->outputQuantizedParam->zeroPoint = outputTensor->quantization->zeroPoint[0];
    LogisticParam->outputQuantizedParam->scale     = outputTensor->quantization->scale[0];

    // set input output index
    dstOp->inputIndexes.resize(tfliteOp->inputs.size());
    dstOp->outputIndexes.resize(tfliteOp->outputs.size());
    for (int i = 0; i < tfliteOp->inputs.size(); i++) {
        dstOp->inputIndexes[i] = tfliteOp->inputs[i];
    }
    for (int i = 0; i < tfliteOp->outputs.size(); i++) {
        dstOp->outputIndexes[i] = tfliteOp->outputs[i];
    }

    dstOp->main.value = LogisticParam;
}

using namespace tflite;
REGISTER_CONVERTER(LogisticTflite, BuiltinOperator_LOGISTIC);
