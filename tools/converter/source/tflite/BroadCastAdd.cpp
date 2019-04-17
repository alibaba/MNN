//
//  BroadCastAdd.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "liteOpConverter.hpp"

DECLARE_OP_COVERTER(AddTflite);

MNN::OpType AddTflite::opType(bool quantizedModel) {
    if (quantizedModel)
        return MNN::OpType_QuantizedAdd;
    return MNN::OpType_BinaryOp;
}

MNN::OpParameter AddTflite::type(bool quantizedModel) {
    if (quantizedModel)
        return MNN::OpParameter_QuantizedAdd;
    return MNN::OpParameter_BinaryOp;
}

void AddTflite::run(MNN::OpT* dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                    const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                    const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                    const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, bool quantizedModel) {
    const auto& addOption = tfliteOp->builtin_options.AsAddOptions();
    if (quantizedModel) {
        auto AddParam = new MNN::QuantizedAddT;

        DCHECK(tfliteOp->inputs.size() == 2) << "tflite Reshape input ERROR";

        // input1
        const int input1Index                     = tfliteOp->inputs[0];
        const auto& input1Tensor                  = tfliteTensors[input1Index];
        AddParam->input1QuantizedParam            = std::unique_ptr<MNN::QuantizedParamT>(new MNN::QuantizedParamT);
        AddParam->input1QuantizedParam->zeroPoint = input1Tensor->quantization->zeroPoint[0];
        AddParam->input1QuantizedParam->scale     = input1Tensor->quantization->scale[0];

        // input1
        const int input2Index                     = tfliteOp->inputs[1];
        const auto& input2Tensor                  = tfliteTensors[input2Index];
        AddParam->input2QuantizedParam            = std::unique_ptr<MNN::QuantizedParamT>(new MNN::QuantizedParamT);
        AddParam->input2QuantizedParam->zeroPoint = input2Tensor->quantization->zeroPoint[0];
        AddParam->input2QuantizedParam->scale     = input2Tensor->quantization->scale[0];

        // output
        const int outputIndex                     = tfliteOp->outputs[0];
        const auto& outputTensor                  = tfliteTensors[outputIndex];
        AddParam->outputQuantizedParam            = std::unique_ptr<MNN::QuantizedParamT>(new MNN::QuantizedParamT);
        AddParam->outputQuantizedParam->zeroPoint = outputTensor->quantization->zeroPoint[0];
        AddParam->outputQuantizedParam->scale     = outputTensor->quantization->scale[0];

        AddParam->activationType = static_cast<MNN::FusedActivation>(addOption->fused_activation_function);

        dstOp->main.value = AddParam;
    } else {
        DCHECK(addOption->fused_activation_function == tflite::ActivationFunctionType_NONE)
            << "BinaryOP Should not has fused_activation_function";
        auto binaryOpParam = new MNN::BinaryOpT;
        // TODO
        binaryOpParam->opType = MNN::BinaryOpOperation_ADD; // defalut
        dstOp->main.value     = binaryOpParam;
    }

    // set input output index
    dstOp->inputIndexes.resize(tfliteOp->inputs.size());
    dstOp->outputIndexes.resize(tfliteOp->outputs.size());
    for (int i = 0; i < tfliteOp->inputs.size(); i++) {
        dstOp->inputIndexes[i] = tfliteOp->inputs[i];
    }
    for (int i = 0; i < tfliteOp->outputs.size(); i++) {
        dstOp->outputIndexes[i] = tfliteOp->outputs[i];
    }
}

using namespace tflite;
REGISTER_CONVERTER(AddTflite, BuiltinOperator_ADD);
