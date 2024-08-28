#include "evaluation/evaluation.hpp"

namespace MNN {
namespace Transformer {

void mergePerformance(struct TimePerformance* dst, struct TimePerformance* src) {
    dst->prefill_record_.insert(dst->prefill_record_.end(), src->prefill_record_.begin(), src->prefill_record_.end());
    dst->decode_record_.insert(dst->decode_record_.end(), src->decode_record_.begin(), src->decode_record_.end());
}

void clearPerformance(struct TimePerformance* perf) {
    perf->prefill_record_.clear();
    perf->decode_record_.clear();
}

void mergePerformance(struct MemPerformance* dst, struct MemPerformance* src) {
    dst->prefill_record_.insert(dst->prefill_record_.end(), src->prefill_record_.begin(), src->prefill_record_.end());
    dst->decode_record_.insert(dst->decode_record_.end(), src->decode_record_.begin(), src->decode_record_.end());
}

void clearPerformance(struct MemPerformance* perf) {
    perf->prefill_record_.clear();
    perf->decode_record_.clear();
}


} // namespace Transformer
} // namespace MNN