//
//  BackendFactory.cpp
//  MNN
//
//  Created by MNN on 2018/07/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "BackendFactory.hpp"
#include "CPUBackend.hpp"
#include "Macro.h"

namespace MNN {
Backend* BackendFactory::create(const Backend::Info& info) {
    auto creator = MNNGetExtraBackendCreator(info.type);
    if (nullptr == creator) {
        MNN_PRINT("Create Backend Failed because no creator for %d\n", info.type);
        return nullptr;
    }
    auto backend = creator->onCreate(info);
    if (nullptr == backend) {
        MNN_PRINT("Create Backend failed, the creator return nullptr, type = %d\n", info.type);
    }
    return backend;
}
} // namespace MNN
