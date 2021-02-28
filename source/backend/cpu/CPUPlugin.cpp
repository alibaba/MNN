//
//  CPUPlugin.cpp
//  MNN
//
//  Created by MNN on 2020/04/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUBackend.hpp"
#include "core/AutoStorage.h"
#include "core/Execution.hpp"

#ifdef MNN_WITH_PLUGIN
#include "MNN/plugin/PluginContext.hpp"
#include "MNN/plugin/PluginKernel.hpp"
#endif  // MNN_WITH_PLUGIN

namespace MNN {

#ifdef MNN_WITH_PLUGIN
static std::shared_ptr<plugin::CPUComputeKernel> getCPUComputeKernel( // NOLINT
    const std::string& name) {                                        // NOLINT
    return std::shared_ptr<plugin::CPUComputeKernel>(                 // NOLINT
        plugin::ComputeKernelRegistry<plugin::CPUComputeKernel>::get(name));
}

class CPUPlugin : public Execution {
public:
    CPUPlugin(std::unique_ptr<plugin::CPUKernelContext> ctx) // NOLINT
        : Execution(ctx->backend()), ctx_(std::move(ctx)) {
        kernel_ = getCPUComputeKernel(ctx_->op_type());
        MNN_CHECK(nullptr != kernel_.get(), // NOLINT
                  "CPU compute kernel has not been registered for plugin op.");
        kernel_->init(ctx_.get());
    }
    virtual ~CPUPlugin() = default;

    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, // NOLINT
                                const std::vector<Tensor*>& outputs) override;

private:
    std::unique_ptr<plugin::CPUKernelContext> ctx_;
    std::shared_ptr<plugin::CPUComputeKernel> kernel_;
};

ErrorCode CPUPlugin::onExecute(const std::vector<Tensor*>& inputs, // NOLINT
                               const std::vector<Tensor*>& outputs) {
    // Setup new context with inputs and outputs.
    plugin::CPUKernelContext ctx( // NOLINT
        ctx_->op_type(), ctx_->backend(), inputs, outputs);
    ctx.setAttrs(ctx_->getAttrs());
    if (kernel_->compute(&ctx)) {
        return NO_ERROR;
    } else {
        MNN_ERROR("Plugin kernel compute failed with false returned.");
        return INVALID_VALUE;
    }
}
#endif  // MNN_WITH_PLUGIN

class CPUPluginCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs,  // NOLINT
                                const std::vector<Tensor*>& outputs, // NOLINT
                                const MNN::Op* op, Backend* backend) const {
#ifdef MNN_WITH_PLUGIN
        MNN_ASSERT(op->type() == OpType_Plugin);
        // Plugin op should has inputs or outputs, or both of them.
        MNN_CHECK(inputs.size() > 0 || outputs.size() > 0, // NOLINT
                  "Plugin op should has inputs or outputs, or both of them.");

        const Plugin* plugin_param = op->main_as<Plugin>();

        const std::string& op_type = plugin_param->type()->str();
        std::unique_ptr<plugin::CPUKernelContext> ctx( // NOLINT
            new plugin::CPUKernelContext(op_type, backend, inputs, outputs));

        for (const Attribute* attr : *(plugin_param->attr())) {
            ctx->setAttr(attr->key()->str(), attr);
        }
        return new CPUPlugin(std::move(ctx));
#else
        MNN_ERROR("Plugin is not supported. Please recompile with `MNN_WITH_PLUGIN` enabled.");
        return nullptr;
#endif  // MNN_WITH_PLUGIN
    }
};

REGISTER_CPU_OP_CREATOR(CPUPluginCreator, OpType_Plugin);

} // namespace MNN
