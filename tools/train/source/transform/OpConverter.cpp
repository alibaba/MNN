//
//  OpConverter.cpp
//  MNN
//
//  Created by MNN on 2019/05/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpConverter.hpp"
#include <map>

static std::map<MNN::OpType, OpConverter*>& getConverter() {
    static std::map<MNN::OpType, OpConverter*> gConverterMap;
    return gConverterMap;
}

OpConverter* OpConverter::get(MNN::OpType type) {
    auto& converterMap = getConverter();
    auto iter          = converterMap.find(type);
    if (iter != converterMap.end()) {
        return iter->second;
    }
    return nullptr;
}

void OpConverter::insert(MNN::OpType type, OpConverter* converter) {
    auto& converterMap = getConverter();
    converterMap.insert(std::make_pair(type, converter));
}
void OpConverter::merge(MNN::NetT* dest, OpConverter::Result& result) {
    dest->tensorName.insert(dest->tensorName.end(), result.tensorNames.begin(), result.tensorNames.end());
    for (auto& op : result.opLists) {
        dest->oplists.emplace_back(std::move(op));
    }
}
