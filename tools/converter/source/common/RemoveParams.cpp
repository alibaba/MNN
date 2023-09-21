//
//  RemoveParams.cpp
//  MNNConverter
//
//  Created by MNN on 2021/08/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CommonUtils.hpp"
#include "../source/core/FileLoader.hpp"
#include <fstream>

auto RemoveParams = [](std::unique_ptr<MNN::OpT>& op) {
    const auto opType = op->type;
    switch (opType) {
        case MNN::OpType_Convolution:
        case MNN::OpType_Deconvolution:
        case MNN::OpType_ConvolutionDepthwise: {
            auto param = op->main.AsConvolution2D();
            param->weight.clear();
            param->bias.clear();
            break;
        }
        case MNN::OpType_TfQuantizedConv2D: {
            auto param = op->main.AsTfQuantizedConv2D();
            param->weight.clear();
            param->bias.clear();
            break;
        }
        case MNN::OpType_MatMul: {
            auto param = op->main.AsMatMul();
            param->weight.clear();
            param->bias.clear();
            break;
        }
        case MNN::OpType_BatchNorm: {
            auto param = op->main.AsBatchNorm();
            param->slopeData.clear();
            param->meanData.clear();
            param->varData.clear();
            param->biasData.clear();
            param->Adata.clear();
            param->Bdata.clear();
            break;
        }
        case MNN::OpType_Scale: {
            auto param = op->main.AsScale();
            param->scaleData.clear();
            param->biasData.clear();
            break;
        }
        case MNN::OpType_LayerNorm: {
            auto param = op->main.AsLayerNorm();
            param->gamma.clear();
            param->beta.clear();
        }
        default:
            break;
    }
};

template <typename T>
static void storeWeight(std::ofstream* fs, std::vector<T>& weight, std::vector<int64_t>& external, int64_t& offset) {
    if (external.empty()) {
        external.push_back(offset);
    }
    int64_t size = weight.size() * sizeof(T);
    fs->write(reinterpret_cast<const char*>(weight.data()), size);
    weight.clear();
    external.push_back(size);
    offset += size;
}

void RemoveAndStoreParam(std::unique_ptr<MNN::OpT>& op, std::ofstream* fs, int64_t& offset) {
    const auto opType = op->type;
    switch (opType) {
        case MNN::OpType_Convolution:
        case MNN::OpType_Deconvolution:
        case MNN::OpType_ConvolutionDepthwise:
        {
            auto param = op->main.AsConvolution2D();
            if (param->quanParameter) {
                storeWeight<int8_t>(fs, param->quanParameter->buffer, param->external, offset);
                storeWeight<float>(fs, param->quanParameter->alpha, param->external, offset);
            } else {
                storeWeight<float>(fs, param->weight, param->external, offset);
                storeWeight<float>(fs, param->bias, param->external, offset);
            }
            break;
        }
        case MNN::OpType_Scale: {
            auto param = op->main.AsScale();
            storeWeight<float>(fs, param->scaleData, param->external, offset);
            if (!param->biasData.empty()) {
                storeWeight<float>(fs, param->biasData, param->external, offset);
            }
            break;
        }
        case MNN::OpType_LayerNorm: {
            auto param = op->main.AsLayerNorm();
            if (!param->gamma.empty() && !param->beta.empty()) {
                storeWeight<float>(fs, param->gamma, param->external, offset);
                storeWeight<float>(fs, param->beta, param->external, offset);
            }
            break;
        }
        case MNN::OpType_TrainableParam:
        case MNN::OpType_Const: {
            auto param = op->main.AsBlob();
            size_t totalSize = 1;
            for (auto dim : param->dims) {
                totalSize *= dim;
            }
            if (totalSize <= 1024) {
                break;
            }
            switch (param->dataType) {
                case MNN::DataType_DT_FLOAT:
                    storeWeight<float>(fs, param->float32s, param->external, offset);
                    break;
               case MNN::DataType_DT_BFLOAT16:
                    storeWeight<uint8_t>(fs, param->uint8s, param->external, offset);
                    break;
                case MNN::DataType_DT_INT32:
                    storeWeight<int>(fs, param->int32s, param->external, offset);
                    break;
                case MNN::DataType_DT_UINT8:
                    storeWeight<uint8_t>(fs, param->uint8s, param->external, offset);
                    break;
                case MNN::DataType_DT_INT8:
                    storeWeight<int8_t>(fs, param->int8s, param->external, offset);
                    break;
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
}

void removeParams(std::unique_ptr<MNN::NetT>& netT) {
    for (auto& op : netT->oplists) {
        RemoveParams(op);
    }
    for (auto& subgraph : netT->subgraphs) {
        for (auto& op : subgraph->nodes) {
            RemoveParams(op);
        }
    }
}

bool saveExternalData(std::unique_ptr<MNN::NetT>& netT, const std::string& extraFileName) {
    std::ofstream extraFile(extraFileName, std::ios::binary);
    if (!extraFile.is_open()) {
        return false;
    }
    int64_t offset = 0;
    for (auto& op : netT->oplists) {
        RemoveAndStoreParam(op, &extraFile, offset);
    }
    for (auto& subgraph : netT->subgraphs) {
        for (auto& op : subgraph->nodes) {
            RemoveAndStoreParam(op, &extraFile, offset);
        }
    }
    extraFile.close();
    return true;
}

template <typename T>
static void loadExternalData(MNN::FileLoader& fl, std::vector<T>& data, int64_t size) {
    data.resize(size / sizeof(T));
    fl.read(reinterpret_cast<char*>(data.data()), size);
}

void loadExternalParam(std::unique_ptr<MNN::OpT>& op, MNN::FileLoader& fl) {
    const auto opType = op->type;
    switch (opType) {
        case MNN::OpType_Convolution:
        case MNN::OpType_Deconvolution:
        case MNN::OpType_ConvolutionDepthwise:
        {
            auto param = op->main.AsConvolution2D();
            if (param->external.size() != 3) {
                return;
            }
            fl.offset(param->external[0]);
            if (param->quanParameter) {
                loadExternalData<int8_t>(fl, param->quanParameter->buffer, param->external[1]);
                loadExternalData<float>(fl, param->quanParameter->alpha, param->external[2]);
            } else {
                loadExternalData<float>(fl, param->weight, param->external[1]);
                loadExternalData<float>(fl, param->bias, param->external[2]);
            }
            param->external.clear();
            break;
        }
        case MNN::OpType_Scale: {
            auto param = op->main.AsScale();
            break;
        }
        case MNN::OpType_LayerNorm: {
            auto param = op->main.AsLayerNorm();
            break;
        }
        case MNN::OpType_TrainableParam:
        case MNN::OpType_Const: {
            auto param = op->main.AsBlob();
            if (param->external.size() != 2) {
                return;
            }
            size_t totalSize = 1;
            for (auto dim : param->dims) {
                totalSize *= dim;
            }
            fl.offset(param->external[0]);
            switch (param->dataType) {
                case MNN::DataType_DT_FLOAT:
                    loadExternalData<float>(fl, param->float32s, param->external[1]);
                    break;
                case MNN::DataType_DT_INT32:
                    loadExternalData<int>(fl, param->int32s, param->external[1]);
                    break;
                case MNN::DataType_DT_UINT8:
                    loadExternalData<uint8_t>(fl, param->uint8s, param->external[1]);
                    break;
                case MNN::DataType_DT_INT8:
                    loadExternalData<int8_t>(fl, param->int8s, param->external[1]);
                    break;
                default:
                    break;
            }
            param->external.clear();
            break;
        }
        default:
            break;
    }
}

bool loadExternalData(std::unique_ptr<MNN::NetT>& netT, const char* extraFileName) {
    MNN::FileLoader fileloader(extraFileName);
    for (auto& op : netT->oplists) {
        loadExternalParam(op, fileloader);
    }
    for (auto& subgraph : netT->subgraphs) {
        for (auto& op : subgraph->nodes) {
            loadExternalParam(op, fileloader);
        }
    }
    return true;
}