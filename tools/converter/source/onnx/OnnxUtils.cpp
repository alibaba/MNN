//
//  OnnxUtils.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OnnxUtils.hpp"
#include <stdio.h>
#include <stdint.h>
#include <fstream>

bool onnx_read_proto_from_binary(const char* filepath, google::protobuf::Message* message) {
    std::ifstream fs(filepath, std::ifstream::in | std::ifstream::binary);
    if (!fs.is_open()) {
        fprintf(stderr, "open failed %s\n", filepath);
        return false;
    }

    google::protobuf::io::IstreamInputStream input(&fs);
    google::protobuf::io::CodedInputStream codedstr(&input);
#if GOOGLE_PROTOBUF_VERSION >= 3011000
    codedstr.SetTotalBytesLimit(INT_MAX);
#else
    codedstr.SetTotalBytesLimit(INT_MAX, INT_MAX/2);
#endif

    bool success = message->ParseFromCodedStream(&codedstr);

    fs.close();

    return success;
}
bool onnx_write_proto_from_binary(const char* filepath, const google::protobuf::Message* message) {
    std::ofstream fs(filepath);
    if (fs.fail()) {
        fprintf(stderr, "open failed %s\n", filepath);
        return false;
    }
    message->SerializeToOstream(&fs);
    fs.close();
    return true;
}
