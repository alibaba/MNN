//
//  MNNDump2Json.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include "MNN_generated.h"
#include "flatbuffers/idl.h"
#include "flatbuffers/minireflect.h"
#include "flatbuffers/util.h"
#include <string.h>

int main(int argc, const char** argv) {
    if (argc <= 2) {
        printf("Usage: ./MNNDump2Json.out XXX.MNN XXX.json\n");
        return 0;
    }
    std::ifstream inputFile(argv[1], std::ios::binary);
    inputFile.seekg(0, std::ios::end);
    auto size = inputFile.tellg();
    inputFile.seekg(0, std::ios::beg);

    char* buffer = new char[size];

    inputFile.read((char*)buffer, size);
    std::ofstream output(argv[2]);

    if (argc > 3) {
        printf("Dont't add convweight\n");
        auto netT = MNN::UnPackNet((void*)buffer);
        auto treatFunction = [&](MNN::OpT* opParam) {
            auto type = opParam->main.type;
            if (type == MNN::OpParameter::OpParameter_Convolution2D) {
                auto param = opParam->main.AsConvolution2D();
                param->weight.clear();
                param->bias.clear();
                if (param->symmetricQuan) {
                    param->symmetricQuan->weight.clear();
                }
                if (param->quanParameter) {
                    param->quanParameter->buffer.clear();
                }
            } else if (type == MNN::OpParameter::OpParameter_Blob) {
                auto blobT = opParam->main.AsBlob();
                blobT->float32s.clear();
                blobT->int8s.clear();
                blobT->uint8s.clear();
                blobT->int32s.clear();
                blobT->int64s.clear();
            } else if (type == MNN::OpParameter::OpParameter_Convolution2D) {
                opParam->main.AsConvolution2D()->weight.clear();
                opParam->main.AsConvolution2D()->bias.clear();
            } else if (type == MNN::OpParameter::OpParameter_MatMul) {
                opParam->main.AsMatMul()->weight.clear();
                opParam->main.AsMatMul()->bias.clear();
            } else if (type == MNN::OpParameter::OpParameter_PRelu) {
                opParam->main.AsPRelu()->slope.clear();
            } else if (type == MNN::OpParameter::OpParameter_Extra) {
                auto extra = opParam->main.AsExtra();
                extra->info.clear();
            } else if(type == MNN::OpParameter::OpParameter_LSTM){
                auto param = opParam->main.AsLSTM();
                if (param->weightH) {
                    param->weightH->float32s.clear();
                }
                if (param->weightI) {
                    param->weightI->float32s.clear();
                }
                if (param->bias) {
                    param->bias->float32s.clear();
                }
            }
        };
        for (int i = 0; i < netT->oplists.size(); ++i) {
            treatFunction(netT->oplists[i].get());
        }
        for (int i = 0; i < netT->subgraphs.size(); ++i) {
            for (int j=0; j<netT->subgraphs[i]->nodes.size(); ++j) {
                treatFunction(netT->subgraphs[i]->nodes[j].get());
            }
        }
        if (argc > 4) {
            printf("Seperate dump subgraph\n");
            for (int i=0; i<netT->subgraphs.size(); ++i) {
                auto& g = netT->subgraphs[i];
                flatbuffers::FlatBufferBuilder newBuilder(1024);
                auto root = MNN::SubGraphProto::Pack(newBuilder, g.get());
                newBuilder.Finish(root);
                auto content = newBuilder.GetBufferPointer();
                char subGraphNameStr[128];
                sprintf(subGraphNameStr, "%s_%d", argv[2], i);
                printf("Dump subgraph %s to %s\n", g->name.c_str(), subGraphNameStr);
                std::ofstream tempOutput(subGraphNameStr);
                auto s       = flatbuffers::FlatBufferToString((const uint8_t*)content, MNN::SubGraphProtoTypeTable());
                tempOutput << s;
            }
            netT->subgraphs.clear();
        }
        flatbuffers::FlatBufferBuilder newBuilder(1024);
        auto root = MNN::Net::Pack(newBuilder, netT.get());
        MNN::FinishNetBuffer(newBuilder, root);
        {
            auto content = newBuilder.GetBufferPointer();
            auto s       = flatbuffers::FlatBufferToString((const uint8_t*)content, MNN::NetTypeTable());
            output << s;
        }
    } else {
        auto s = flatbuffers::FlatBufferToString((const uint8_t*)buffer, MNN::NetTypeTable());
        output << s;
    }

    delete[] buffer;
    return 0;
}
