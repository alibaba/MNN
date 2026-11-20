#include "LlmDelegate.hpp"
#include "llmconfig.hpp"

namespace MNN {
namespace Transformer {

#ifdef MNN_LLM_TORQ
Llm* createTorqDelegate(const std::shared_ptr<LlmConfig>& config);
#endif

Llm* createLlmDelegate(const std::shared_ptr<LlmConfig>& config) {
    const auto name = config->delegate();
#ifdef MNN_LLM_TORQ
    if (name == "TORQ") {
        return createTorqDelegate(config);
    }
#endif
    MNN_ERROR("Unsupported LLM delegate: %s\n", name.c_str());
    return nullptr;
}

} // namespace Transformer
} // namespace MNN
