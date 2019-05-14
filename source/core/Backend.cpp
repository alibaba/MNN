//
//  Backend.cpp
//  MNN
//
//  Created by MNN on 2018/07/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Backend.hpp"
#include <stdio.h>
#include <mutex>
#include "MNN_generated.h"
#include "Macro.h"

namespace MNN {

void registerBackend();

static std::map<MNNForwardType, std::pair<const BackendCreator*, bool>>& GetExtraCreator() {
    static std::once_flag flag;
    static std::map<MNNForwardType, std::pair<const BackendCreator*, bool>>* gExtraCreator;
    std::call_once(flag,
                   [&]() { gExtraCreator = new std::map<MNNForwardType, std::pair<const BackendCreator*, bool>>; });
    return *gExtraCreator;
}

const BackendCreator* MNNGetExtraBackendCreator(MNNForwardType type) {
    registerBackend();

    auto& gExtraCreator = GetExtraCreator();
    auto iter           = gExtraCreator.find(type);
    if (iter == gExtraCreator.end()) {
        return nullptr;
    }
    if (!iter->second.second) {
        return iter->second.first;
    }
    Backend::Info info;
    info.type = type;
    std::shared_ptr<Backend> bn(iter->second.first->onCreate(info));
    if (nullptr != bn.get()) {
        return iter->second.first;
    }
    return nullptr;
}

bool MNNInsertExtraBackendCreator(MNNForwardType type, const BackendCreator* creator, bool needCheck) {
    auto& gExtraCreator = GetExtraCreator();
    if (gExtraCreator.find(type) != gExtraCreator.end()) {
        MNN_ASSERT(false && "duplicate type");
        return false;
    }
    gExtraCreator.insert(std::make_pair(type, std::make_pair(creator, needCheck)));
    return true;
}
} // namespace MNN
