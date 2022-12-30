//
//  RemoveParams.cpp
//  MNNConverter
//
//  Created by MNN on 2021/08/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CommonUtils.hpp"
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
static void storeWeight(std::ofstream& fs, std::vector<T>& weight, std::vector<int64_t>& external, int64_t& offset) {
    if (external.empty()) {
        external.push_back(offset);
    }
    int64_t size = weight.size() * sizeof(T);
    fs.write(reinterpret_cast<const char*>(weight.data()), size);
    weight.clear();
    external.push_back(size);
    offset += size;
}

static void RemoveAndStoreParam(std::unique_ptr<MNN::OpT>& op, std::ofstream& fs, int64_t& offset) {
    const auto opType = op->type;
    switch (opType) {
        case MNN::OpType_Convolution:
        case MNN::OpType_Deconvolution:
        case MNN::OpType_ConvolutionDepthwise:
        {
            auto param = op->main.AsConvolution2D();
            storeWeight<float>(fs, param->weight, param->external, offset);
            storeWeight<float>(fs, param->bias, param->external, offset);
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
            switch (param->dataType) {
                case MNN::DataType_DT_FLOAT:
                    storeWeight<float>(fs, param->float32s, param->external, offset);
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
        RemoveAndStoreParam(op, extraFile, offset);
    }
    for (auto& subgraph : netT->subgraphs) {
        for (auto& op : subgraph->nodes) {
            RemoveAndStoreParam(op, extraFile, offset);
        }
    }
    extraFile.close();
    return true;
}
