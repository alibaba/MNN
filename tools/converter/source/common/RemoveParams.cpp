//
//  RemoveParams.cpp
//  MNNConverter
//
//  Created by MNN on 2021/08/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CommonUtils.hpp"

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
        default:
            break;
    }
};

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
