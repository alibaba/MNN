//
//  tfOpConverter.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "tfOpConverter.hpp"
#include <stdlib.h>

tfOpConverterSuit *tfOpConverterSuit::global = nullptr;

tfOpConverter *tfOpConverterSuit::search(const std::string &name) {
    auto iter = mTests.find(name);
    if (iter == mTests.end()) {
        return nullptr;
    }
    return iter->second;
}

tfOpConverterSuit *tfOpConverterSuit::get() {
    if (global == nullptr)
        global = new tfOpConverterSuit;
    return global;
}

tfOpConverterSuit::~tfOpConverterSuit() {
    for (auto &iter : mTests) {
        delete iter.second;
    }
    mTests.clear();
}

void tfOpConverterSuit::insert(tfOpConverter *t, const char *name) {
    mTests.insert(std::make_pair(name, t));
}
