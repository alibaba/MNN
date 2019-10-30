//
//  writeFb.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <fstream>
#include <iostream>

#include "logkit.h"
#include "writeFb.hpp"
#include <set>

int writeFb(std::unique_ptr<MNN::NetT>& netT, const std::string& MNNModelFile, bool benchmarkModel) {
    if (benchmarkModel) {
        for (auto& op : netT->oplists) {
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
                }
                case MNN::OpType_MatMul: {
                    auto param = op->main.AsMatMul();
                    param->weight.clear();
                    param->bias.clear();
                }
                case MNN::OpType_BatchNorm: {
                    auto param = op->main.AsBatchNorm();
                    param->slopeData.clear();
                    param->meanData.clear();
                    param->varData.clear();
                    param->biasData.clear();
                    param->Adata.clear();
                    param->Bdata.clear();
                }
                case MNN::OpType_Scale: {
                    auto param = op->main.AsScale();
                    param->scaleData.clear();
                    param->biasData.clear();
                }
                default:
                    break;
            }
        }
    }
    std::set<std::string> notSupportOps;
    for (auto& op : netT->oplists) {
        if (op->type == MNN::OpType_Extra) {
            if (op->main.AsExtra()->engine != "MNN") {
                notSupportOps.insert(op->main.AsExtra()->engine + "::" + op->main.AsExtra()->type);
            }
        }
    }
    std::ostringstream notSupportInfo;
    if (!notSupportOps.empty()) {
        for (auto name : notSupportOps) {
            notSupportInfo << name << " | ";
        }
    }
    DCHECK_EQ(notSupportOps.size(), 0) << "These Op Not Support: " << notSupportInfo.str();
    flatbuffers::FlatBufferBuilder builderOutput(1024);
    builderOutput.ForceDefaults(true);
    auto len = MNN::Net::Pack(builderOutput, netT.get());
    builderOutput.Finish(len);
    int sizeOutput    = builderOutput.GetSize();
    auto bufferOutput = builderOutput.GetBufferPointer();

    std::ofstream output(MNNModelFile, std::ofstream::binary);
    output.write((const char*)bufferOutput, sizeOutput);

    return 0;
}
