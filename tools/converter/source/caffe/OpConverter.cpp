//
//  OpConverter.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpConverter.hpp"
#include <stdlib.h>

OpConverterSuit* OpConverterSuit::global = NULL;

OpConverter* OpConverterSuit::search(const std::string& name) {
    auto iter = mTests.find(name);
    if (iter == mTests.end()) {
        return NULL;
    }
    return iter->second;
}

OpConverterSuit* OpConverterSuit::get() {
    if (global == NULL)
        global = new OpConverterSuit;
    return global;
}

OpConverterSuit::~OpConverterSuit() {
    for (auto& iter : mTests) {
        delete iter.second;
    }
    mTests.clear();
}

void OpConverterSuit::insert(OpConverter* t, const char* name) {
    mTests.insert(std::make_pair(name, t));
}
