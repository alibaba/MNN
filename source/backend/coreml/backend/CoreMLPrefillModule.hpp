//
//  CoreMLPrefillModule.hpp
//  MNN
//
//  Created by MNN on 2026/11/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifndef MNN_COREML_PREFILL_MODULE_HPP
#define MNN_COREML_PREFILL_MODULE_HPP
#include <MNN/expr/Module.hpp>
namespace MNN {
struct KVMeta;
namespace Transformer {
class LlmConfig;
std::shared_ptr<Express::Module>
createCoreMLPrefillModule(const LlmConfig& config, const std::string& model, const std::string& weights,
                          const std::shared_ptr<Express::Executor::RuntimeManager>& runtime, KVMeta* meta);
} // namespace Transformer
} // namespace MNN
#endif
