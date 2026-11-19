#pragma once

#include "llm/llm.hpp"

namespace MNN {
namespace Transformer {

class LlmDelegate : public Llm {
protected:
    using Llm::Llm;
    using Llm::mBlockSize;
    using Llm::mGenerateParam;
    using Llm::mGenerationStrategy;
    using Llm::mPrefixCacheMode;
};

Llm* createLlmDelegate(const std::shared_ptr<LlmConfig>& config);

} // namespace Transformer
} // namespace MNN
