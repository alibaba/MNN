//
//  writeFb.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "writeFb.hpp"
#include <fstream>
#include <iostream>

int writeFb(std::unique_ptr<MNN::NetT>& netT, const std::string& MNNModelFile, bool benchmarkModel) {
    flatbuffers::FlatBufferBuilder builder(1024);

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

    auto offset = MNN::Net::Pack(builder, netT.get());
    builder.Finish(offset);

    int size      = builder.GetSize();
    auto ocontent = builder.GetBufferPointer();
    std::ofstream output(MNNModelFile);
    output.write((const char*)ocontent, size);

    return 0;
}
