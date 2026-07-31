//
//  MNNRvvFastPathRegistration.cpp
//  MNN
//
//  Created by MNN on 2026/07/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/compute/CommonOptFunction.h"
#include "backend/cpu/compute/CPUExtension.hpp"

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
#include "MNNRvvAttention.hpp"
#endif

namespace MNN {

void MNNRvvInitializeFastPathFunctions(CoreFunctions* core) {
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
    static constexpr CPUExtension extension(nullptr, MNNRvvCreateAttentionExecution);
#else
    static constexpr CPUExtension extension(nullptr, nullptr);
#endif
    core->kvUpdateConcurrent = true;
    core->extension = &extension;
}

void MNNRvvInitializeInt8FastPathFunctions(CoreInt8Functions*) {}

} // namespace MNN
