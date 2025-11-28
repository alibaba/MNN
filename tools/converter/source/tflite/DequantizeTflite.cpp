//
//  DequantizeTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2020/05/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "liteOpConverter.hpp"
#include "TfliteUtils.hpp"


DECLARE_OP_COVERTER(DequantizeTflite);

MNN::OpType DequantizeTflite::opType(int quantizedModel){
    return MNN::OpType_Dequantize;
}

MNN::OpParameter DequantizeTflite::type(int quantizedModel){
    return MNN::OpParameter_Dequantize;
}

void DequantizeTflite::run(MNN::OpT *dstOp, const std::unique_ptr<tflite::OperatorT> &tfliteOp, const std::vector<std::unique_ptr<tflite::TensorT> > &tfliteTensors, const std::vector<std::unique_ptr<tflite::BufferT> > &tfliteModelBuffer, const std::vector<std::unique_ptr<tflite::OperatorCodeT> > &tfliteOpSet, int quantizedModel){
    DCHECK(1 == tfliteOp->inputs.size()) << "Dequantize should have one input now";
    auto inputIndex = tfliteOp->inputs[0];
    const auto& inputTensor = tfliteTensors[inputIndex];
    const auto& outputTensor = tfliteTensors[tfliteOp->outputs[0]];
    if (inputTensor->type != tflite::TensorType_UINT8) {
        if (outputTensor->type != tflite::TensorType_UINT8) {
            dstOp->type = MNN::OpType_Identity;
            dstOp->main.type = MNN::OpParameter_NONE;
            return;
        }
    }

    auto dequantizeParam = new MNN::DequantizeT;
    
    dequantizeParam->modelFormat = MNN::ModeFormat_TFLITE;
    
    dequantizeParam->type = TfliteDequantDataTypeToMNN(inputTensor->type);

    auto quantizedParam = new MNN::QuantizedParamT;
    
    quantizedParam->zeroPoint = static_cast<int32_t>(inputTensor->quantization->zero_point[0]);
    quantizedParam->scale = inputTensor->quantization->scale[0];
    dequantizeParam->inputQuantizedParam = std::unique_ptr<MNN::QuantizedParamT>(quantizedParam);
    
    dstOp->main.value = dequantizeParam;
}

using namespace tflite;
REGISTER_CONVERTER(DequantizeTflite, BuiltinOperator_DEQUANTIZE);
