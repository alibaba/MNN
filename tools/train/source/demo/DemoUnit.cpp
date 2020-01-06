//
//  DemoUnit.cpp
//  MNN
//
//  Created by MNN on 2019/11/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "DemoUnit.hpp"
#include <stdlib.h>

DemoUnitSet* DemoUnitSet::gInstance = NULL;

DemoUnitSet* DemoUnitSet::get() {
    if (gInstance == NULL)
        gInstance = new DemoUnitSet;
    return gInstance;
}

DemoUnitSet::~DemoUnitSet() {
    for (auto iter : mUnit) {
        delete iter.second;
    }
}

void DemoUnitSet::add(DemoUnit* test, const char* name) {
    test->name = name;
    mUnit.insert(std::make_pair(name, test));
}

DemoUnit* DemoUnitSet::search(const char* key) {
    std::string prefix = key;
    std::vector<std::string> wrongs;
    auto iter = mUnit.find(prefix);
    if (iter == mUnit.end()) {
        return nullptr;
    }
    return iter->second;
}
