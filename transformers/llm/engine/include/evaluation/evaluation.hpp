

#ifndef TRANSFORMER_EVALUATION_hpp
#define TRANSFORMER_EVALUATION_hpp

#include <vector>
#include <cstddef>

namespace MNN {
namespace Transformer {

#define MICRO_TO_MILLI 1e-3f
#define MILLI_TO_MICRO 1000
#define MICRO_TO_SEC 1e-6f
#define SEC_TO_MICRO 1000000

struct PrefillTimePerformance {
    size_t prefill_prev_token_ = 0;
    size_t prefill_token_ = 0;
    size_t prefill_us_ = 0;
};

struct DecodeTimePerformance {
    size_t decode_prev_token_ = 0;
    size_t decode_us_ = 0;
};

struct TimePerformance {
    std::vector<PrefillTimePerformance> prefill_record_;
    std::vector<DecodeTimePerformance> decode_record_;
};

void mergePerformance(struct TimePerformance* dst, struct TimePerformance* src);
void clearPerformance(struct TimePerformance* perf);
} // namespace Transformer
} // namespace MNN
#endif // TRANSFORMER_EVALUATION_hpp