// Copyright (c) Alibaba Group Holding Limited. All rights reserved.
#ifndef MNN_SHARED_ATTENTION_MODULE_HPP
#define MNN_SHARED_ATTENTION_MODULE_HPP
#include <MNN/expr/Module.hpp>
#include "core/KVMeta.hpp"
namespace MNN {
namespace Express {
// Internal, serialized loading/execution only. Own the actual objects supplying the shared KV state.
struct SharedAttentionOwners {
    std::shared_ptr<Module> base;
    std::shared_ptr<Executor::RuntimeManager> runtime;
    std::shared_ptr<KVMeta> meta;
    std::shared_ptr<Executor> executor;
};
MNN_PUBLIC Module* loadSharedAttentionModule(const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs, const char* fileName,
    const SharedAttentionOwners& owners, const Module::Config* config);
MNN_PUBLIC Module* loadSharedAttentionModule(const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs, const uint8_t* buffer, size_t length,
    const SharedAttentionOwners& owners, const Module::Config* config);
}
}
#endif
