#include "MNN/plugin/Plugin.hpp"

namespace MNN {
namespace plugin {
    
PluginContext::PluginContext(const std::vector<Tensor*>& inputs,  // NOLINT
                                    const std::vector<Tensor*>& outputs) // NOLINT
    : inputs_(inputs), outputs_(outputs) {
}

const std::vector<Tensor*>& PluginContext::inputs() const {
    return inputs_;
}
const std::vector<Tensor*>& PluginContext::outputs() const {
    return outputs_;
}

const Tensor* PluginContext::input(const int index) const {
    MNN_ASSERT(index < inputs_.size());
    return inputs_.at(index);
}

const Tensor* PluginContext::output(const int index) const {
    MNN_ASSERT(index < outputs_.size());
    return outputs_.at(index);
}

Tensor* PluginContext::output(const int index) {
    MNN_ASSERT(index < outputs_.size());
    return outputs_.at(index);
}

bool PluginContext::hasAttr(const std::string& name) const {
    return attrs_.count(name) > 0;
}

bool PluginContext::setAttr(const std::string& name, // NOLINT
                                   const Attribute* attr) {
    return attrs_.emplace(name, attr).second;
}

void PluginContext::setAttrs( // NOLINT
    const std::unordered_map<std::string, const Attribute*>& attrs) {
    attrs_ = attrs;
}

const Attribute* PluginContext::getAttr(const std::string& name) const {
    const auto& it = attrs_.find(name);
    MNN_ASSERT(it != attrs_.end());
    return it->second;
}

const std::unordered_map<std::string, const Attribute*>& // NOLINT
PluginContext::getAttrs() const {
    return attrs_;
}

} // namespace plugin
} // namespace MNN
