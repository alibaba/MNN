//
//  liteConverter.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <iostream>

#include "logkit.h"

#include "liteConverter.hpp"
#include "liteOpConverter.hpp"

int tflite2MNNNet(const std::string inputModel, const std::string bizCode, std::unique_ptr<MNN::NetT>& MNNNetT) {
    const std::string model_name = inputModel;
    auto model                   = std::shared_ptr<TfliteModel>(new TfliteModel(model_name));
    model->readModel();
    auto& tfliteModel = model->get();

    const auto& tfliteOpSet = tfliteModel->operator_codes;
    // const auto operatorCodesSize = tfliteOpSet.size();
    const auto subGraphsSize      = tfliteModel->subgraphs.size();
    const auto& tfliteModelBuffer = tfliteModel->buffers;

    // check whether this tflite model is quantization model
    // use the weight's data type of Conv2D|DepthwiseConv2D to decide quantizedModel mode
    bool quantizedModel = true;
    for (int i = 0; i < subGraphsSize; ++i) {
        const auto& ops     = tfliteModel->subgraphs[i]->operators;
        const auto& tensors = tfliteModel->subgraphs[i]->tensors;
        const int opNums    = ops.size();
        for (int j = 0; j < opNums; ++j) {
            const int opcodeIndex = ops[j]->opcode_index;
            const auto opCode     = tfliteOpSet[opcodeIndex]->builtin_code;
            if (opCode == tflite::BuiltinOperator_CONV_2D || opCode == tflite::BuiltinOperator_DEPTHWISE_CONV_2D) {
                const int weightIndex    = ops[j]->inputs[1];
                const auto& weightTensor = tensors[weightIndex];
                quantizedModel           = weightTensor->type == tflite::TensorType_UINT8;
                if (!quantizedModel)
                    break;
            }
        }
    }

    for (int i = 0; i < subGraphsSize; ++i) {
        const auto& ops     = tfliteModel->subgraphs[i]->operators;
        const auto& tensors = tfliteModel->subgraphs[i]->tensors;

        // set input
        for (const auto index : tfliteModel->subgraphs[i]->inputs) {
            MNN::OpT* inputOp       = new MNN::OpT;
            const auto& inputTensor = tensors[index];
            inputOp->name           = inputTensor->name;
            inputOp->type           = MNN::OpType_Input;
            inputOp->main.type      = MNN::OpParameter_Input;

            auto inputParam     = new MNN::InputT;
            inputParam->dformat = MNN::MNN_DATA_FORMAT_NHWC;
            DCHECK(inputTensor->shape.size() == 4) << "Input Shape should be 4D";
            inputParam->dims = inputTensor->shape;
            if (quantizedModel) {
                inputParam->dtype = MNN::DataType_DT_UINT8;
            } else {
                inputParam->dtype = MNN::DataType_DT_FLOAT;
            }

            inputOp->main.value = inputParam;
            inputOp->outputIndexes.push_back(index);
            MNNNetT->oplists.emplace_back(inputOp);
        }

        // set output names
        for (int k = 0; k < tfliteModel->subgraphs[i]->outputs.size(); ++k) {
            MNNNetT->outputName.push_back(tensors[tfliteModel->subgraphs[i]->outputs[k]]->name);
        }
        // tensor names
        for (const auto& tensor : tensors) {
            MNNNetT->tensorName.push_back(tensor->name);
        }

        const int opNums = ops.size();
        for (int j = 0; j < opNums; ++j) {
            const int opcodeIndex = ops[j]->opcode_index;
            const auto opCode     = tfliteOpSet[opcodeIndex]->builtin_code;

            MNN::OpT* op = new MNN::OpT;
            auto creator = liteOpConverterSuit::get()->search(opCode);
            DCHECK(creator) << "NOT_SUPPORTED_OP: [ " << tflite::EnumNameBuiltinOperator(opCode) << " ]";

            // tflite op to MNN op
            op->name      = tensors[ops[j]->outputs[0]]->name;
            op->type      = creator->opType(quantizedModel);
            op->main.type = creator->type(quantizedModel);
            creator->run(op, ops[j], tensors, tfliteModelBuffer, tfliteOpSet, quantizedModel);
            MNNNetT->oplists.emplace_back(op);
        }
    }

    MNNNetT->sourceType = MNN::NetSource_TFLITE;
    MNNNetT->bizCode    = bizCode;

    return 0;
}

TfliteModel::TfliteModel(const std::string fileName) : _modelName(fileName) {
}

TfliteModel::~TfliteModel() {
}

void TfliteModel::readModel() {
    std::ifstream inputFile(_modelName, std::ios::binary);
    inputFile.seekg(0, std::ios::end);
    const auto size = inputFile.tellg();
    inputFile.seekg(0, std::ios::beg);

    char* buffer = new char[size];
    inputFile.read(buffer, size);
    inputFile.close();

    // verify model
    flatbuffers::Verifier verify((uint8_t*)buffer, size);
    if (!tflite::VerifyModelBuffer(verify)) {
        LOG(FATAL) << "TFlite model version ERROR!";
    }

    _tfliteModel = tflite::UnPackModel(buffer);
    delete[] buffer;
}

std::unique_ptr<tflite::ModelT>& TfliteModel::get() {
    return _tfliteModel;
}
