//
//  ShapeInference.h
//  MNN
//
//  Created by MNN on 2020/04/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_PLUGIN_PLUGIN_CONTEXT_HPP_
#define MNN_PLUGIN_PLUGIN_CONTEXT_HPP_

#include <unordered_map>
#include <vector>

#include <MNN/Interpreter.hpp> // Backend
#include <MNN/Tensor.hpp>
#include "Tensor_generated.h"

namespace MNN {
namespace plugin {

class MNN_PUBLIC PluginContext {
public:
    PluginContext() = delete;
    PluginContext(const std::vector<Tensor*>& inputs, // NOLINT
                  const std::vector<Tensor*>& outputs);

    virtual ~PluginContext() = default;

    const std::vector<Tensor*>& inputs() const;
    const std::vector<Tensor*>& outputs() const;

    const Tensor* input(const int index) const;
    const Tensor* output(const int index) const;

    Tensor* output(const int index);

    bool hasAttr(const std::string& name) const;

    bool setAttr(const std::string& name, const Attribute* attr);

    void setAttrs(const std::unordered_map<std::string, // NOLINT
                                           const Attribute*>& attrs);

    const Attribute* getAttr(const std::string& name) const;

    const std::unordered_map<std::string, const Attribute*>& getAttrs() const;

protected:
    const std::vector<Tensor*>& inputs_;
    const std::vector<Tensor*>& outputs_;
    std::unordered_map<std::string, const Attribute*> attrs_;
};

class MNN_PUBLIC InferShapeContext : public PluginContext {
public:
    InferShapeContext() = delete;
    InferShapeContext(const std::vector<Tensor*>& inputs, // NOLINT
                      const std::vector<Tensor*>& outputs);

    virtual ~InferShapeContext() = default;
};

class MNN_PUBLIC CPUKernelContext : public PluginContext {
public:
    CPUKernelContext() = delete;
    CPUKernelContext(const std::string& op_type,         // NOLINT
                     Backend* backend,                   // NOLINT
                     const std::vector<Tensor*>& inputs, // NOLINT
                     const std::vector<Tensor*>& outputs);

    virtual ~CPUKernelContext() = default;

    Backend* backend() const;
    const std::string& op_type() const;

private:
    const std::string op_type_ = "";
    Backend* backend_          = nullptr;
};


} // namespace plugin
} // namespace MNN

#endif // MNN_PLUGIN_PLUGIN_CONTEXT_HPP_
