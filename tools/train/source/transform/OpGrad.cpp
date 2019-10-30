//
//  OpGrad.cpp
//  MNN
//
//  Created by MNN on 2019/05/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpGrad.hpp"
#include <sstream>
using namespace std;
namespace MNN {
static std::map<int, OpGrad*>& getConverter() {
    static std::map<int, OpGrad*> gConverterMap;
    return gConverterMap;
}

OpGrad* OpGrad::get(int type) {
    auto& converterMap = getConverter();
    auto iter          = converterMap.find(type);
    if (iter != converterMap.end()) {
        return iter->second;
    }
    return nullptr;
}

void OpGrad::insert(int type, OpGrad* converter) {
    auto& converterMap = getConverter();
    converterMap.insert(std::make_pair(type, converter));
}

std::string numberToString(int index) {
    std::ostringstream os;
    os << index;
    return os.str();
}
}
