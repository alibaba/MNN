

#include <vector>
#include <cstddef>
#include "evaluation/evaluation.hpp"

namespace MNN {
namespace Transformer {

void clearPerformance(struct TimePerformance* perf) {
    perf->prefill_record_.clear();
    perf->decode_record_.clear();
    perf->prompt_record_.clear();
}
void appendNewPromptRecord(struct TimePerformance* perf, int input_len, bool reuse_kv) {
    if (reuse_kv) {
        perf->prompt_record_.push_back(input_len);
    } else {
        // not reuse kv
        if (!perf->decode_record_.empty()) {
            perf->prompt_record_.push_back(input_len - (perf->decode_record_.back().decode_prev_token_+1));
        } else {
            // first prefill
            perf->prompt_record_.push_back(input_len);
        }
    }
}

} // Transformer
} // MNN