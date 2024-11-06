

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

} // Transformer
} // MNN