//
//  TfUtils.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include <algorithm>
#include <fstream>
#include <limits>
#include <set>

#include "TfUtils.hpp"
#include "logkit.h"

bool tf_read_proto_from_binary(const char* filepath, google::protobuf::Message* message) {
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

bool find_attr_value(const tensorflow::NodeDef* node, const char* key, tensorflow::AttrValue& value) {
    const google::protobuf::Map<std::string, tensorflow::AttrValue>& attr = node->attr();

    const google::protobuf::Map<std::string, tensorflow::AttrValue>::const_iterator it = attr.find(key);
    if (it != attr.end()) {
        value = it->second;
        return true;
    }

    return false;
}

bool convertDataFormat(const float* src, float* dst, int planeNumber, int CI, int CO) {
    // H W CI CO --> CO CI H W
    assert(planeNumber > 0);
    assert(CI > 0);
    assert(CO > 0);
    assert(src != nullptr);
    for (int coi = 0; coi < CO; coi++) {
        for (int cii = 0; cii < CI; cii++) {
            for (int i = 0; i < planeNumber; ++i) {
                dst[(coi * CI + cii) * planeNumber + i] = src[(i * CI + cii) * CO + coi];
            }
        }
    }

    return true;
}
