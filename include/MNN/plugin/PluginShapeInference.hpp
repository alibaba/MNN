//
//  ShapeInference.h
//  MNN
//
//  Created by MNN on 2020/04/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_PLUGIN_PLUGIN_SHAPE_INFERENCE_HPP_
#define MNN_PLUGIN_PLUGIN_SHAPE_INFERENCE_HPP_

#include <functional>
#include <string>
#include <unordered_map>

#include <MNN/plugin/PluginContext.hpp>

namespace MNN {
namespace plugin {

class MNN_PUBLIC InferShapeKernel {
public:
    virtual ~InferShapeKernel()                  = default;
    virtual bool compute(InferShapeContext* ctx) = 0;
};

class MNN_PUBLIC InferShapeKernelRegister {
public:
    // typedef InferShapeKernel* (*Factory)();
    typedef std::function<InferShapeKernel*()> Factory;
    static std::unordered_map<std::string, Factory>* getFactoryMap();

    static bool add(const std::string& name, Factory factory);

    static InferShapeKernel* get(const std::string& name);
};

template <typename PluginKernel>
struct InferShapeKernelRegistrar {
    InferShapeKernelRegistrar(const std::string& name) {
        InferShapeKernelRegister::add(name, []() { // NOLINT
            return new PluginKernel;               // NOLINT
        });
    }
};

#define REGISTER_PLUGIN_OP(name, inferShapeKernel)                      \
    namespace {                                                         \
    static auto _plugin_infer_shape_##name##_ __attribute__((unused)) = \
        InferShapeKernelRegistrar<inferShapeKernel>(#name);             \
    } // namespace

} // namespace plugin
} // namespace MNN

#endif // MNN_PLUGIN_PLUGIN_SHAPE_INFERENCE_HPP_
