//
//  PluginWrapper.cpp
//  Codegen
//
//  Created by MNN on 2020/09/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "kernel.h"
#include "MNN/plugin/PluginKernel.hpp"
#include <vector>

MNN_PUBLIC int _intPluginWrapper = 10; // Just for linking successfully.

namespace MNN {
namespace plugin {

namespace backend {
class PluginWrapper : public CPUComputeKernel {
public:
    bool init(CPUKernelContext*) override { return true; }
    bool compute(CPUKernelContext* ctx) override;
};

bool PluginWrapper::compute(CPUKernelContext* ctx) {
    int kernelIdx = 0;
    if (ctx->hasAttr("kernel")) {
        kernelIdx = ctx->getAttr("kernel")->i();
    }
    int I = ctx->inputs().size();
    float** inputs = new float*[I];
    for (int i = 0; i < I; i++) {
        inputs[i] = reinterpret_cast<float*>(ctx->input(i)->buffer().host);
    }
    int O = ctx->outputs().size();
    float** outputs = new float*[O];
    for (int i = 0; i < O; i++) {
        outputs[i] = reinterpret_cast<float*>(ctx->output(i)->buffer().host);
    }
    kernels[kernelIdx](inputs, outputs);
    return true;
}
} // namespace backend

REGISTER_PLUGIN_COMPUTE_KERNEL(PluginWrapper, backend::PluginWrapper);

} // namespace plugin
} // namespace MNN
