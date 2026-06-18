#ifdef MNN_LLM_SUPPORT_SEGMENT

#ifndef LLM_SEGMENT_HPP
#define LLM_SEGMENT_HPP

#include <memory>

#include "llm/llm.hpp"

namespace MNN {
namespace Transformer {

Llm* createSegmentLlm(std::shared_ptr<LlmConfig> config);

} // namespace Transformer
} // namespace MNN

#endif // LLM_SEGMENT_HPP

#endif // MNN_LLM_SUPPORT_SEGMENT
