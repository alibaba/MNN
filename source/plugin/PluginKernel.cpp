//
//  PluginKernel.cpp
//  MNN
//
//  Created by MNN on 2020/04/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_WITH_PLUGIN

#include <string>
#include <unordered_map>

#include "MNN/plugin/PluginKernel.hpp"

namespace MNN {
namespace plugin {

CPUKernelContext::CPUKernelContext(const std::string& op_type,          // NOLINT
                                   Backend* backend,                    // NOLINT
                                   const std::vector<Tensor*>& inputs,  // NOLINT
                                   const std::vector<Tensor*>& outputs) // NOLINT
    : op_type_(op_type), backend_(backend), PluginContext(inputs, outputs) {
}

template <typename PluginKernelT>
using Factory = typename ComputeKernelRegistry<PluginKernelT>::Factory;

template <typename PluginKernelT>
/*static*/ auto ComputeKernelRegistry<PluginKernelT>::getFactoryMap() // NOLINT
    -> std::unordered_map<std::string, Factory>* {
    static std::unordered_map<std::string, Factory> gFactoryMap;
    return &gFactoryMap;
}

template <typename PluginKernelT>
/*static*/ bool ComputeKernelRegistry<PluginKernelT>::add( // NOLINT
    const std::string& name, Factory factory) {
    auto* gFactoryMap = ComputeKernelRegistry<PluginKernelT>::getFactoryMap();
    if (gFactoryMap->count(name)) {
        MNN_PRINT("Factory has been registered for name %s.", name.c_str());
    }
    return gFactoryMap->emplace(name, factory).second;
}

template <typename PluginKernelT>
/*static*/ PluginKernelT* ComputeKernelRegistry<PluginKernelT>::get( // NOLINT
    const std::string& name) {
    auto* gFactoryMap = ComputeKernelRegistry<PluginKernelT>::getFactoryMap();
    if (!gFactoryMap->count(name)) {
        MNN_PRINT("Factory has not been registered for name %s.", name.c_str());
        return nullptr;
    }
    return gFactoryMap->at(name)();
}

template class ComputeKernelRegistry<CPUComputeKernel>;

} // namespace plugin
} // namespace MNN

#endif  // #ifdef MNN_WITH_PLUGIN
