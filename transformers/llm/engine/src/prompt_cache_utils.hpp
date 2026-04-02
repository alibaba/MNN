#ifndef PROMPT_CACHE_UTILS_HPP
#define PROMPT_CACHE_UTILS_HPP

#include <string>

namespace MNN {
namespace Transformer {

// Strip <think>...</think> blocks from cached prompt text.
// When enable_thinking is active, the template appends <think>...</think>
// to the last assistant message only. Since it won't be last on the next
// turn, stripping ensures the cached text matches the next turn's rendering.
inline void stripThinkBlocks(std::string& text) {
    while (true) {
        auto start = text.find("<think>");
        if (start == std::string::npos)
            break;
        auto end = text.find("</think>", start);
        if (end == std::string::npos)
            break;
        end += 8;
        while (end < text.size() && (text[end] == '\n' || text[end] == '\r'))
            end++;
        text.erase(start, end - start);
    }
}

}  // namespace Transformer
}  // namespace MNN

#endif  // PROMPT_CACHE_UTILS_HPP
