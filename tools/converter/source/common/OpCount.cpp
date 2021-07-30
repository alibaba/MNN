//
//  OpCount.cpp
//  MNNConverter
//
//  Created by MNN on b'2021/06/26'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpCount.hpp"
#include <memory>
namespace MNN {
OpCount::OpCount() {
    // Do nothing
}
OpCount::~OpCount() {
    // Do nothing
}

OpCount* OpCount::get() {
    static std::shared_ptr<OpCount> gOpCount;
    if (gOpCount == nullptr) {
        gOpCount.reset(new OpCount);
    }
    return gOpCount.get();
}

void OpCount::insertOp(const std::string& framework, const std::string& name) {
    auto fr = mOps.find(framework);
    if (fr == mOps.end()) {
        mOps.insert(std::make_pair(framework, std::set<std::string>{name}));
        return;
    }
    fr->second.insert(name);
}
const std::map<std::string, std::set<std::string>>& OpCount::getMap() {
    return mOps;
}

};
