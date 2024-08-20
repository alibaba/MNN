//
//  ConvertToFullQuant.hpp
//  MNN
//
//  Created by MNN on 2021/04/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CONVERTTOFULLQUANT_HPP
#define CONVERTTOFULLQUANT_HPP

#include <vector>
#include <map>
#include <string>
#include <fstream>
#include <sstream>
#include "MNN_generated.h"
#include "core/IDSTEncoder.hpp"

using namespace MNN;

namespace ConvertToFullQuant {

void ConvertOp(std::unique_ptr<OpT>& op, int opIndex, NetT* net, SubGraphProtoT* subgraph, std::vector<int>& needEraseIndices) {
    auto opType = op->type;
    if ((opType != OpType_FloatToInt8) && (opType != OpType_Int8ToFloat) && (opType != OpType_ConvInt8) && (opType != OpType_DepthwiseConvInt8)) {
        return;
    }

    auto& tensorNames = subgraph ? subgraph->tensors : net->tensorName;
    auto& tensorDescribe = subgraph ? subgraph->extraTensorDescribe : net->extraTensorDescribe;

    auto findReferenceOpsAndIndices = [&](int outputIndex) {
        std::map<OpT*, std::vector<int> > refOps;
        if (subgraph != nullptr) {
            for (auto& node : subgraph->nodes) {
                for (int i = 0; i < node->inputIndexes.size(); i++) {
                    int index = node->inputIndexes[i];
                    if (index == outputIndex) {
                        refOps[node.get()].emplace_back(i);
                    }
                }
            }
        } else {
            for (auto& node : net->oplists) {
                for (int i = 0; i < node->inputIndexes.size(); i++) {
                    int index = node->inputIndexes[i];
                    if (index == outputIndex) {
                        refOps[node.get()].emplace_back(i);
                    }
                }
            }
        }
        return refOps;
    };

    auto inputIndex = op->inputIndexes[0];
    int outputIndex = op->outputIndexes[0];
    if ((opType == OpType_FloatToInt8) || (opType == OpType_Int8ToFloat)) {
        auto params = op->main.AsQuantizedFloatParam();
        std::unique_ptr<MNN::TensorDescribeT> describe(new MNN::TensorDescribeT);
        describe->index = inputIndex;
        std::unique_ptr<MNN::TensorQuantInfoT> qInfo(new MNN::TensorQuantInfoT);
        qInfo->zero = params->zeroPoint;
        if (opType == OpType_FloatToInt8) {
            qInfo->scale = 1. / params->tensorScale[0];
        } else {
            qInfo->scale = params->tensorScale[0];
        }
        qInfo->min = params->clampMin;
        qInfo->max = params->clampMax;
        qInfo->type = MNN::DataType_DT_INT8;
        describe->quantInfo = std::move(qInfo);

        if (opType == OpType_FloatToInt8) {
            tensorDescribe.emplace_back(std::move(describe));
        } else {
            bool found = false;
            for (int i = 0; i < tensorDescribe.size(); i++) {
                if (tensorDescribe[i]->index == inputIndex) {
                    found = true;
                    tensorDescribe[i]->index = inputIndex;
                    tensorDescribe[i]->quantInfo->zero = params->zeroPoint;
                    tensorDescribe[i]->quantInfo->scale = params->tensorScale[0];
                    tensorDescribe[i]->quantInfo->min = params->clampMin;
                    tensorDescribe[i]->quantInfo->max = params->clampMax;
                    tensorDescribe[i]->quantInfo->type = MNN::DataType_DT_INT8;
                    break;
                }
            }
            if (!found) {
                tensorDescribe.emplace_back(std::move(describe));
            }
        }

        tensorNames[outputIndex] = "notused";
        // reference op change input indexes
        auto referenceOps = findReferenceOpsAndIndices(outputIndex);
        for (auto& refOps : referenceOps) {
            for (int i = 0; i < refOps.second.size(); i++) {
                refOps.first->inputIndexes[refOps.second[i]] = inputIndex;
            }
        }

        needEraseIndices.emplace_back(opIndex);
    }

    if ((opType == OpType_ConvInt8) || (opType == OpType_DepthwiseConvInt8)) {
        if (opType == OpType_ConvInt8) {
            op->type = OpType_Convolution;
        } else {
            op->type = OpType_ConvolutionDepthwise;
        }
        auto conv2D = op->main.AsConvolution2D();

        // encoding
        if (conv2D->symmetricQuan && (!conv2D->symmetricQuan->weight.empty())) {
            // full quant support for train quant in NN.cpp
            if (conv2D->quanParameter && conv2D->quanParameter->buffer.empty()) {
                auto aMin = conv2D->quanParameter->aMin;
                auto scaleIn = conv2D->quanParameter->scaleIn;
                auto scaleOut = conv2D->quanParameter->scaleOut;
                auto weightScale = conv2D->quanParameter->alpha;

                if (aMin != 0 && scaleIn != 0 && scaleOut != 0 && weightScale.size() > 0) {
                    auto weight = conv2D->symmetricQuan->weight;
                    const int kn = conv2D->common->outputCount;
                    const int ks = weight.size() / kn;
                    std::vector<float> scales(kn, 1.0f);
                    std::vector<float> weightFloat;

                    for (int i = 0; i < weight.size(); i++) {
                        weightFloat.emplace_back(weight[i] * weightScale[i / ks]);
                    }

                    conv2D->quanParameter = IDSTEncoder::encode(weightFloat.data(), weightScale, ks, kn, false, weight.data(), aMin);
                    conv2D->quanParameter->scaleIn = scaleIn;
                    conv2D->quanParameter->scaleOut = scaleOut;
                    conv2D->symmetricQuan->weight.clear();

                    std::unique_ptr<MNN::TensorDescribeT> describe(new MNN::TensorDescribeT);
                    describe->index = outputIndex;
                    std::unique_ptr<MNN::TensorQuantInfoT> qInfo(new MNN::TensorQuantInfoT);
                    qInfo->zero = conv2D->symmetricQuan->outputZeroPoint;
                    qInfo->scale = scaleOut;
                    qInfo->min = conv2D->symmetricQuan->clampMin;
                    qInfo->max = conv2D->symmetricQuan->clampMax;
                    qInfo->type = MNN::DataType_DT_INT8;
                    describe->quantInfo = std::move(qInfo);
                    tensorDescribe.emplace_back(std::move(describe));

                    return;
                }
            }
        }

        // fake info
        std::unique_ptr<MNN::TensorDescribeT> describe(new MNN::TensorDescribeT);
        describe->index = outputIndex;
        std::unique_ptr<MNN::TensorQuantInfoT> qInfo(new MNN::TensorQuantInfoT);
        qInfo->zero = 0;
        qInfo->scale = 0;
        qInfo->min = -127;
        qInfo->max = 127;
        qInfo->type = MNN::DataType_DT_INT8;
        describe->quantInfo = std::move(qInfo);
        tensorDescribe.emplace_back(std::move(describe));
    }
}

void convert(std::string modelFile) {
    std::unique_ptr<MNN::NetT> netT;
    std::ifstream input(modelFile);
    std::ostringstream outputOs;
    outputOs << input.rdbuf();
    netT = MNN::UnPackNet(outputOs.str().c_str());
    auto net = netT.get();

    std::vector<int> netNeedEraseIndices;
    for (int i = 0; i < net->oplists.size(); i++) {
        auto& op = net->oplists[i];
        ConvertOp(op, i, net, nullptr, netNeedEraseIndices);
    }
    std::reverse(netNeedEraseIndices.begin(), netNeedEraseIndices.end());
    for (int i = 0; i < netNeedEraseIndices.size(); i++) {
        net->oplists.erase(net->oplists.begin() + netNeedEraseIndices[i]);
    }

    for (auto& subgraph : net->subgraphs) {
        std::vector<int> subgraphNeedEraseIndices;
        for (int i = 0; i < subgraph->nodes.size(); i++) {
            auto& op = subgraph->nodes[i];
            ConvertOp(op, i, net, subgraph.get(), subgraphNeedEraseIndices);
        }
        std::reverse(subgraphNeedEraseIndices.begin(), subgraphNeedEraseIndices.end());
        for (int i = 0; i < subgraphNeedEraseIndices.size(); i++) {
            subgraph->nodes.erase(subgraph->nodes.begin() + subgraphNeedEraseIndices[i]);
        }
    }

    flatbuffers::FlatBufferBuilder builderOutput(1024);
    builderOutput.ForceDefaults(true);
    auto len = MNN::Net::Pack(builderOutput, net);
    builderOutput.Finish(len);
    std::ofstream output(modelFile);
    output.write((const char*)builderOutput.GetBufferPointer(), builderOutput.GetSize());
}

} // namespace ConvertToFullQuant

#endif // CONVERTTOFULLQUANT_HPP
