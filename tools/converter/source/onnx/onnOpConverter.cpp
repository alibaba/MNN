//
//  onnOpConverter.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "onnxOpConverter.hpp"

onnxOpConverterSuit::onnxOpConverterSuit() {
}

onnxOpConverterSuit::~onnxOpConverterSuit() {
    for (auto& iter : mConverterContainer) {
        delete iter.second;
    }
    mConverterContainer.clear();
}

onnxOpConverterSuit* onnxOpConverterSuit::global = nullptr;

onnxOpConverterSuit* onnxOpConverterSuit::get() {
    if (global == nullptr) {
        global = new onnxOpConverterSuit;
    }
    return global;
}

void onnxOpConverterSuit::insert(onnxOpConverter* t, const char* name) {
    mConverterContainer.insert(std::make_pair(name, t));
}

onnxOpConverter* onnxOpConverterSuit::search(const std::string& name) {
    auto iter = mConverterContainer.find(name);
    if (iter == mConverterContainer.end()) {
        return nullptr;
    }
    return iter->second;
}
