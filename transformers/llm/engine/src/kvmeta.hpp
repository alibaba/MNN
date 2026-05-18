//
//  kvmeta.hpp
//
//  Created by MNN on 2025/04/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef KVMETA_hpp
#define KVMETA_hpp

// Unified KVMeta definition is in source/core/KVMeta.hpp
#include "core/KVMeta.hpp"

// Import MNN::KVMeta into MNN::Transformer namespace for backward compatibility
namespace MNN {
using namespace Express;
namespace Transformer {
    using MNN::KVMeta;
}
}

#endif // KVMETA_hpp
