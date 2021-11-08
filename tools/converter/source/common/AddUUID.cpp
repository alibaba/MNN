//
//  AddUUID.cpp
//  MNNConverter
//
//  Created by MNN on 2021/08/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CommonUtils.hpp"
#include <random>
#include <sstream>

static std::string uuid4() {
    static std::random_device              rd;
    static std::mt19937_64                 gen(rd());
    static std::uniform_int_distribution<> dis(0, 15);
    static std::uniform_int_distribution<> dis2(8, 11);

    std::stringstream ss;
    int i;
    ss << std::hex;
    for (i = 0; i < 8; i++) {
       ss << dis(gen);
    }
    ss << "-";
    for (i = 0; i < 4; i++) {
       ss << dis(gen);
    }
    ss << "-4";
    for (i = 0; i < 3; i++) {
       ss << dis(gen);
    }
    ss << "-";
    ss << dis2(gen);
    for (i = 0; i < 3; i++) {
       ss << dis(gen);
    }
    ss << "-";
    for (i = 0; i < 12; i++) {
       ss << dis(gen);
    };
    return ss.str();
}

void addUUID(std::unique_ptr<MNN::NetT>& netT, MNN::Compression::Pipeline proto) {
    if (netT->mnn_uuid.empty()) {
        // set uuid from compress file
        if (proto.has_mnn_uuid()) {
            netT->mnn_uuid = proto.mnn_uuid();
        } else {
            netT->mnn_uuid = uuid4();
        }
    }
}
