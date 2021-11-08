//
//  ShapeInference.cpp
//  MNN
//
//  Created by MNN on 2020/04/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_WITH_PLUGIN

#include "MNN/plugin/PluginShapeInference.hpp"
#include "MNN/plugin/PluginContext.hpp"

namespace MNN {
namespace plugin {

InferShapeContext::InferShapeContext(const std::vector<Tensor*>& inputs,  // NOLINT
                                     const std::vector<Tensor*>& outputs) // NOLINT
    : PluginContext(inputs, outputs) {
}

typedef InferShapeKernelRegister::Factory Factory;

/*static*/ auto InferShapeKernelRegister::getFactoryMap() // NOLINT
    -> std::unordered_map<std::string, Factory>* {
    static std::unordered_map<std::string, Factory> gFactoryMap;
    return &gFactoryMap;
}

/*static*/ bool InferShapeKernelRegister::add( // NOLINT
    const std::string& name, Factory factory) {
    auto* gFactoryMap = InferShapeKernelRegister::getFactoryMap();
    if (gFactoryMap->count(name)) {
        MNN_PRINT("Factory has been registered for name %s.", name.c_str());
    }
    return gFactoryMap->emplace(name, factory).second;
}

/*static*/ InferShapeKernel* InferShapeKernelRegister::get( // NOLINT
    const std::string& name) {
    auto* gFactoryMap = InferShapeKernelRegister::getFactoryMap();
    if (!gFactoryMap->count(name)) {
        MNN_PRINT("Factory has not been registered for name %s.", name.c_str());
        return nullptr;
    }
    return gFactoryMap->at(name)();
}

} // namespace plugin
} // namespace MNN

#endif  // #ifdef MNN_WITH_PLUGIN
