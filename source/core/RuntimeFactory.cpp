//
//  RuntimeFactory.cpp
//  MNN
//
//  Created by MNN on 2018/07/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "core/RuntimeFactory.hpp"
//#include <MNN/core/CPUBackend.hpp
#include "core/Macro.h"

namespace MNN {
Runtime* RuntimeFactory::create(const Backend::Info& info) {
    auto creator = MNNGetExtraRuntimeCreator(info.type);
    if (nullptr == creator) {
        MNN_PRINT("Create Runtime Failed because no creator for %d\n", info.type);
        return nullptr;
    }
    auto runtime = creator->onCreate(info);
    if (nullptr == runtime) {
        MNN_PRINT("Create Runtime failed, the creator return nullptr, type = %d\n", info.type);
    }
    return runtime;
}
} // namespace MNN
