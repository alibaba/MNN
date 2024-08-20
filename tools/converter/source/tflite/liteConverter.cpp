//
//  liteConverter.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <iostream>
#include <functional>
#include "logkit.h"

#include "liteConverter.hpp"
#include "liteOpConverter.hpp"

static MNN::DataType _dataTypeMap(tflite::TensorType type) {
    switch (type) {
        case tflite::TensorType_FLOAT32:
            return MNN::DataType_DT_FLOAT;
            break;
        case tflite::TensorType_INT32:
            return MNN::DataType_DT_INT32;
            break;
        case tflite::TensorType_UINT8:
            return MNN::DataType_DT_UINT8;
            break;
        default:
            return MNN::DataType_DT_FLOAT;
            break;
    }
}

static void _converteConstantDataToMNNConstantNode(
    int tensorIndex, const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
    const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffers, std::unique_ptr<MNN::NetT>& MNNNetT) {
    // check whether buffer data size is greater than zero,
    // if size > 0, then this tensor is Constant, convete this tensor to be MNN Constant node
    const auto& tensor         = tfliteTensors[tensorIndex];
    const uint32_t bufferIndex = tensor->buffer;
    const auto tensorBuffer    = tfliteModelBuffers[bufferIndex]->data;
    const auto bufferSize      = tensorBuffer.size();
    if (bufferSize == 0)
        return;

    // this is Constant data
    std::unique_ptr<MNN::OpT> mnnConstantOp(new MNN::OpT);
    mnnConstantOp->name      = tensor->name;
    mnnConstantOp->type      = MNN::OpType_Const;
    mnnConstantOp->main.type = MNN::OpParameter_Blob;
    mnnConstantOp->outputIndexes.push_back(tensorIndex);

    std::unique_ptr<MNN::BlobT> mnnBlob(new MNN::BlobT);
    // TODO, map tflite data type to mnn data type
    mnnBlob->dataType   = _dataTypeMap(tensor->type);
    mnnBlob->dataFormat = MNN::MNN_DATA_FORMAT_NHWC;
    mnnBlob->dims       = tensor->shape;

    if (mnnBlob->dataType == MNN::DataType_DT_FLOAT) {
        mnnBlob->float32s.resize(bufferSize / 4);
        memcpy(mnnBlob->float32s.data(), tensorBuffer.data(), bufferSize);
    } else if (mnnBlob->dataType == MNN::DataType_DT_INT32) {
        mnnBlob->int32s.resize(bufferSize / 4);
        memcpy(mnnBlob->int32s.data(), tensorBuffer.data(), bufferSize);
    } else {
        DCHECK(false) << "TODO support other data type!";
    }
    mnnConstantOp->main.value = mnnBlob.release();

    MNNNetT->tensorName.emplace_back(mnnConstantOp->name);
    MNNNetT->oplists.emplace_back(std::move(mnnConstantOp));
}
template<typename SRC, typename DST>
void convert(const SRC* s, DST* d, size_t sizeInBytes) {
    auto size = sizeInBytes / sizeof(SRC);
    for (size_t i=0; i<size; ++i) {
        d[i] = s[i];
    }
}


static std::function<void(const void*, void*, size_t size)>  _getConvertFunction(tflite::TensorType type) {
    switch (type) {
        case tflite::TensorType_FLOAT64:
            return [](const void* s, void* d, size_t size) {
                convert((double*)s, (float*)d, size);
            };
        case tflite::TensorType_UINT64:
            return [](const void* s, void* d, size_t size) {
                convert((uint64_t*)s, (int32_t*)d, size);
            };
        case tflite::TensorType_INT16:
            return [](const void* s, void* d, size_t size) {
                convert((int16_t*)s, (int32_t*)d, size);
            };
        case tflite::TensorType_INT64:
            return [](const void* s, void* d, size_t size) {
                convert((int64_t*)s, (int32_t*)d, size);
            };
        default:
            break;
    }
    return nullptr;
}
static MNN::DataType _convertType(tflite::TensorType type) {
    if (type == tflite::TensorType_FLOAT32) {
        return MNN::DataType_DT_FLOAT;
    }
    if (type == tflite::TensorType_FLOAT64) {
        return MNN::DataType_DT_FLOAT;
    }
    if (type == tflite::TensorType_INT8) {
        return MNN::DataType_DT_INT8;
    }
    if (type == tflite::TensorType_INT16) {
        return MNN::DataType_DT_INT32;
    }
    if (type == tflite::TensorType_INT32) {
        return MNN::DataType_DT_INT32;
    }
    if (type == tflite::TensorType_INT64) {
        return MNN::DataType_DT_INT32;
    }
    if (type == tflite::TensorType_UINT8) {
        return MNN::DataType_DT_UINT8;
    }
    if (type == tflite::TensorType_UINT64) {
        return MNN::DataType_DT_INT32;
    }
    if (type == tflite::TensorType_FLOAT16) {
        return MNN::DataType_DT_HALF;
    }
    return MNN::DataType_DT_INVALID;
}
static bool needExtractInput(uint32_t opCode) {
#define NONEED(x) if (x == opCode) return false;
    NONEED(tflite::BuiltinOperator_CONV_2D);
    NONEED(tflite::BuiltinOperator_DEPTHWISE_CONV_2D);
    NONEED(tflite::BuiltinOperator_SPLIT);
    NONEED(tflite::BuiltinOperator_CONCATENATION);
    NONEED(tflite::BuiltinOperator_CONV_2D);
    NONEED(tflite::BuiltinOperator_RESIZE_BILINEAR);
    NONEED(tflite::BuiltinOperator_RESIZE_NEAREST_NEIGHBOR);
    NONEED(tflite::BuiltinOperator_SOFTMAX);


    return true;
}

int tflite2MNNNet(const std::string inputModel, const std::string bizCode,
                  std::unique_ptr<MNN::NetT>& MNNNetT) {
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
    int quantizedModel = 0;
    for (int i = 0; i < subGraphsSize; ++i) {
        const auto& ops     = tfliteModel->subgraphs[i]->operators;
        const auto& tensors = tfliteModel->subgraphs[i]->tensors;
        const int opNums    = static_cast<int>(ops.size());
        for (int j = 0; j < opNums; ++j) {
            const int opcodeIndex = ops[j]->opcode_index;
            const auto opCode     = tfliteOpSet[opcodeIndex]->builtin_code;
            if (opCode == tflite::BuiltinOperator_CONV_2D || opCode == tflite::BuiltinOperator_DEPTHWISE_CONV_2D ||
                opCode == tflite::BuiltinOperator_TRANSPOSE_CONV) {
                const int weightIndex    = ops[j]->inputs[1];
                const auto& weightTensor = tensors[weightIndex];
                if (weightTensor->type == tflite::TensorType_UINT8) {
                    quantizedModel = 1;
                } else if (weightTensor->type == tflite::TensorType_INT8) {
                    quantizedModel = 2;
                }
            }
        }
    }
    auto& buffers = tfliteModel->buffers;

    for (int i = 0; i < subGraphsSize; ++i) {
        const auto& ops     = tfliteModel->subgraphs[i]->operators;
        const auto& tensors = tfliteModel->subgraphs[i]->tensors;

        // set const
        std::vector<bool> extractedTensors(tfliteModel->subgraphs[i]->tensors.size(), false);

        // set input
        for (const auto index : tfliteModel->subgraphs[i]->inputs) {
            MNN::OpT* inputOp       = new MNN::OpT;
            const auto& inputTensor = tensors[index];
            inputOp->name           = inputTensor->name;
            inputOp->type           = MNN::OpType_Input;
            inputOp->main.type      = MNN::OpParameter_Input;

            auto inputParam     = new MNN::InputT;
            inputParam->dformat = MNN::MNN_DATA_FORMAT_NHWC;
            inputParam->dims = inputTensor->shape;
            inputParam->dtype = _convertType(inputTensor->type);
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
            if (needExtractInput(opCode)) {
                for (auto input : ops[j]->inputs) {
                    if (input < 0 || extractedTensors[input]) {
                        continue;
                    }
                    extractedTensors[input] = true;
                    auto& tensor = tfliteModel->subgraphs[i]->tensors[input];
                    auto& buffer = buffers[tensor->buffer];
                    if (buffer->data.empty()) {
                        continue;
                    }
                    std::unique_ptr<MNN::OpT> newOp(new MNN::OpT);
                    newOp->type = MNN::OpType_Const;
                    newOp->name = tensor->name;
                    newOp->outputIndexes = {input};
                    newOp->main.type = MNN::OpParameter_Blob;
                    newOp->main.value = new MNN::BlobT;
                    auto blob = newOp->main.AsBlob();
                    blob->dims = tensor->shape;
                    blob->dataFormat = MNN::MNN_DATA_FORMAT_NHWC;
                    blob->dataType = _convertType(tensor->type);
                    if (MNN::DataType_DT_INVALID == blob->dataType) {
                        MNN_ERROR("Don't support tensor type for %s\n", tflite::EnumNameTensorType(tensor->type));
                        MNNNetT.reset();
                        return 0;
                    }
                    int size = 1;
                    for (auto s : blob->dims) {
                        size *= s;
                    }
                    void* dst = nullptr;
                    switch (blob->dataType) {
                        case MNN::DataType_DT_FLOAT:
                            blob->float32s.resize(size);
                            dst = blob->float32s.data();
                            break;
                        case MNN::DataType_DT_INT32:
                            blob->int32s.resize(size);
                            dst = blob->int32s.data();
                            break;
                        case MNN::DataType_DT_INT8:
                            blob->int8s.resize(size);
                            dst = blob->int8s.data();
                            break;
                        case MNN::DataType_DT_UINT8:
                            blob->uint8s.resize(size);
                            dst = blob->uint8s.data();
                            break;
                        case MNN::DataType_DT_HALF:
                            blob->uint8s.resize(size * 2);
                            dst = blob->uint8s.data();
                            break;
                        default:
                            break;
                    }
                    auto func = _getConvertFunction(tensor->type);
                    if (nullptr == func) {
                        ::memcpy(dst, buffer->data.data(), buffer->data.size());
                    } else {
                        func(buffer->data.data(), dst, buffer->data.size());
                    }
                    MNNNetT->oplists.emplace_back(std::move(newOp));
                }
            }


            if (opCode == tflite::BuiltinOperator_CUSTOM) {
                const int inputSize = ops[j]->inputs.size();
                for (int k = 0; k < inputSize; ++k) {
                    _converteConstantDataToMNNConstantNode(ops[j]->inputs[k], tensors, tfliteModelBuffer, MNNNetT);
                }
            }

            MNN::OpT* op = new MNN::OpT;
            auto creator = liteOpConverterSuit::get()->search(opCode);
            DCHECK(creator) << "NOT_SUPPORTED_OP: [ " << tflite::EnumNameBuiltinOperator(opCode) << " ]";
            if (nullptr == creator) {
                // Has error, reset net
                MNNNetT.reset();
                return 0;
            }
            // tflite op to MNN op
            op->name      = tensors[ops[j]->outputs[0]]->name;
            op->type      = creator->opType(quantizedModel);
            op->main.type = creator->type(quantizedModel);
            // set default input output index
            op->inputIndexes.resize(ops[j]->inputs.size());
            op->outputIndexes.resize(ops[j]->outputs.size());
            auto insertQuantinfo = [&](int idx) {
                if (quantizedModel != 2) {
                    return;
                }
                if (tensors[idx]->type != tflite::TensorType_INT8) {
                    return;
                }
                auto quant = tensors[idx]->quantization.get();
                if (!quant) {
                    return;
                }
                std::unique_ptr<MNN::TensorDescribeT> tensorDescribe(new MNN::TensorDescribeT);
                tensorDescribe->index = idx;
                tensorDescribe->name = MNNNetT->tensorName[idx];
                tensorDescribe->quantInfo.reset(new MNN::TensorQuantInfoT);
                tensorDescribe->quantInfo->type = MNN::DataType_DT_INT8;
                tensorDescribe->quantInfo->scale = quant->scale[0];
                tensorDescribe->quantInfo->zero = quant->zero_point[0];
                MNNNetT->extraTensorDescribe.emplace_back(std::move(tensorDescribe));
            };
            for (int i = 0; i < ops[j]->inputs.size(); i++) {
                op->inputIndexes[i] = ops[j]->inputs[i];
            }
            for (int i = 0; i < ops[j]->outputs.size(); i++) {
                op->outputIndexes[i] = ops[j]->outputs[i];
                insertQuantinfo(ops[j]->outputs[i]);
            }
            // Run actual conversion
            creator->run(op, ops[j], tensors, tfliteModelBuffer, tfliteOpSet, quantizedModel);
            if (op->type == MNN::OpType_MAX) {
                // Has error, reset net
                MNNNetT.reset();
                return 0;
            }
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
