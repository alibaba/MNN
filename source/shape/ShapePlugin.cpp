//  ShapePlugin.cpp
//  MNN
//
//  Created by MNN on 2020/04/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//


#include "shape/SizeComputer.hpp"
#include "core/TensorUtils.hpp"

#ifdef MNN_WITH_PLUGIN
#include "MNN/plugin/PluginShapeInference.hpp"
#endif  // MNN_WITH_PLUGIN

namespace MNN {

#ifdef MNN_WITH_PLUGIN
static std::shared_ptr<plugin::InferShapeKernel> getInferShapeKernel( // NOLINT
    const std::string& name) {                                        // NOLINT
    return std::shared_ptr<plugin::InferShapeKernel>(                 // NOLINT
        plugin::InferShapeKernelRegister::get(name));
}
#endif  // MNN_WITH_PLUGIN

class PluginSizeComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        // Plugin op should has inputs or outputs, or both.
        MNN_CHECK(inputs.size() > 0 || outputs.size() > 0, // NOLINT
                  "Plugin op should has inputs or outputs, or both of them.");

#ifdef MNN_WITH_PLUGIN
        const Plugin* plugin_param = op->main_as<Plugin>();
        std::shared_ptr<plugin::InferShapeKernel> kernel = // NOLINT
            getInferShapeKernel(plugin_param->type()->str());
        MNN_CHECK(nullptr != kernel.get(), // NOLINT
                  "Shape inference kernel has not been registered for plugin op.");

        plugin::InferShapeContext ctx(inputs, outputs);
        for (const Attribute* attr : *(plugin_param->attr())) {
            ctx.setAttr(attr->key()->str(), attr);
        }
        bool status = kernel->compute(&ctx);
        if (!status) {
            MNN_ERROR("Plugin op infer shape failed with false returned.");
        }
        return status;
#else
        MNN_ERROR("Plugin is not supported. Please recompile with `MNN_WITH_PLUGIN` enabled.");
        return false;
#endif  // MNN_WITH_PLUGIN
    }
};

REGISTER_SHAPE(PluginSizeComputer, OpType_Plugin);

} // namespace MNN
