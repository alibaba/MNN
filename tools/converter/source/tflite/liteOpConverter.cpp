//
//  liteOpConverter.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "liteOpConverter.hpp"

liteOpConverterSuit* liteOpConverterSuit::_uniqueSuit = nullptr;

liteOpConverter* liteOpConverterSuit::search(const tflite::BuiltinOperator opIndex) {
    auto iter = _liteOpConverters.find(opIndex);
    if (iter == _liteOpConverters.end()) {
        return nullptr;
    }
    return iter->second;
}

liteOpConverterSuit* liteOpConverterSuit::get() {
    if (_uniqueSuit == nullptr) {
        _uniqueSuit = new liteOpConverterSuit;
    }
    return _uniqueSuit;
}

liteOpConverterSuit::~liteOpConverterSuit() {
    for (auto& it : _liteOpConverters) {
        delete it.second;
    }
    _liteOpConverters.clear();
}

void liteOpConverterSuit::insert(liteOpConverter* t, tflite::BuiltinOperator opIndex) {
    _liteOpConverters.insert(std::make_pair(opIndex, t));
}
