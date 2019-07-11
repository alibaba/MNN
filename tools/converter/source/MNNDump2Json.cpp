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
        for (int i = 0; i < netT->oplists.size(); ++i) {
            auto type     = netT->oplists[i]->main.type;
            auto& opParam = netT->oplists[i];
            if (type == MNN::OpParameter::OpParameter_Convolution2D) {
                auto param = opParam->main.AsConvolution2D();
                param->weight.clear();
                param->bias.clear();
                if (param->symmetricQuan) {
                    param->symmetricQuan->weight.clear();
                }
            }
            else if (type == MNN::OpParameter::OpParameter_Blob) {
                auto blobT = opParam->main.AsBlob();
                blobT->float32s.clear();
                blobT->int8s.clear();
            }
            else if (type == MNN::OpParameter::OpParameter_Convolution2D) {
                opParam->main.AsConvolution2D()->weight.clear();
                opParam->main.AsConvolution2D()->bias.clear();
            } else if (type == MNN::OpParameter::OpParameter_MatMul) {
                opParam->main.AsMatMul()->weight.clear();
                opParam->main.AsMatMul()->bias.clear();
            } else if (type == MNN::OpParameter::OpParameter_PRelu) {
                opParam->main.AsPRelu()->slope.clear();
            }
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
