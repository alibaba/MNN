//
//  Execution.cpp
//  MNN
//
//  Created by MNN on 2018/07/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Execution.hpp"
#include <map>
#include <mutex>

namespace MNN {

typedef std::map<MNNForwardType, std::map<std::string, std::shared_ptr<Execution::Creator>>> ExtraMap;
static ExtraMap* gExtra = nullptr;
static std::mutex gMutex;

static void _init() {
    if (nullptr == gExtra) {
        gExtra = new ExtraMap;
    }
}

const Execution::Creator* Execution::searchExtraCreator(const std::string& key, MNNForwardType type) {
    std::unique_lock<std::mutex> __l(gMutex);
    _init();

    auto fwd = gExtra->find(type);
    if (fwd == gExtra->end()) {
        return nullptr;
    }
    auto iter = fwd->second.find(key);
    if (iter == fwd->second.end()) {
        return nullptr;
    }
    return iter->second.get();
}

bool Execution::insertExtraCreator(std::shared_ptr<Creator> creator, const std::string& key, MNNForwardType type) {
    std::unique_lock<std::mutex> __l(gMutex);
    _init();

    auto iter = gExtra->find(type);
    if (iter == gExtra->end()) {
        std::map<std::string, std::shared_ptr<Creator>> iterMap;
        gExtra->insert(std::make_pair(type, iterMap));
        iter = gExtra->find(type);
    }
    if (iter->second.find(key) != iter->second.end()) {
        return false;
    }
    iter->second.insert(std::make_pair(key, creator));
    return true;
}

bool Execution::removeExtraCreator(const std::string& key, MNNForwardType type) {
    std::unique_lock<std::mutex> __l(gMutex);
    _init();

    auto fwd = gExtra->find(type);
    if (fwd == gExtra->end()) {
        return false;
    }
    auto iter = fwd->second.find(key);
    if (iter == fwd->second.end()) {
        return false;
    }
    fwd->second.erase(iter);
    return true;
}
} // namespace MNN
