//
//  JitPluginWrapper.cpp
//  Codegen
//
//  Created by MNN on 2021/01/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include "jit/LLVMJit.hpp"
#include "MNN/plugin/PluginKernel.hpp"
#include "cpu/CPUAst.hpp"
#include <vector>

MNN_PUBLIC int _intPluginWrapper = 10; // Just for linking successfully.

using namespace llvm;
using namespace llvm::orc;

namespace MNN {
namespace plugin {

namespace backend {
class JitPluginWrapper : public CPUComputeKernel {
public:
    bool init(CPUKernelContext*) override { return true; }
    bool compute(CPUKernelContext* ctx) override;
};

bool JitPluginWrapper::compute(CPUKernelContext* ctx) {
    int kernelIdx = 0;
    if (ctx->hasAttr("kernel")) {
        kernelIdx = ctx->getAttr("kernel")->i();
    }

    LLVMJIT* jit = LLVMJIT::createLLVMJIT();
    MNN_ASSERT(jit != nullptr);

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
    void (*kernel)(float**, float**) = (void (*)(float**, float**))jit->getFuncByIdx(kernelIdx);
    kernel(inputs, outputs);
    return true;
}
} // namespace backend

REGISTER_PLUGIN_COMPUTE_KERNEL(JitPluginWrapper, backend::JitPluginWrapper);

} // namespace plugin
} // namespace MNN
