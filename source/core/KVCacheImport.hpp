//
//  KVCacheImport.hpp
//  MNN
//
//  Created by MNN on 2026/11/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_KV_CACHE_IMPORT_HPP
#define MNN_KV_CACHE_IMPORT_HPP
#include <MNN/Tensor.hpp>
#include <map>
#include <memory>
#include <string>

namespace MNN {
// Owned dense [token, kv_head, head_dim] states, consumed once by native attention.
struct KVCacheImport {
    struct Layer {
        std::shared_ptr<Tensor> key, value;
    };
    std::map<std::string, Layer> layers;
    size_t tokens = 0;
    const void* syncedBackend = nullptr;
};
} // namespace MNN
#endif
