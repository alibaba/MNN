

#ifndef TRANSFORMER_EVALUATION_hpp
#define TRANSFORMER_EVALUATION_hpp

#include <vector>
#include <cstddef>
#include "MemMonitor.hpp"

namespace MNN {
namespace Transformer {

#define MICRO_TO_MILLI 1e-3f
#define MILLI_TO_MICRO 1000
#define MICRO_TO_SEC 1e-6f
#define SEC_TO_MICRO 1000000

#define MEGA_TO_GIGA (1/1024.f)
#define GIGA_TO_MEGA 1024.f
#define KILLO_TO_GIGA (1/1024.f/1024.f)
#define GIGA_TO_KILLO (1024.f*1024.f)
#define KILLO_TO_MEGA (1/1024.f)
#define MEGA_TO_KILLO 1024.f
#define BYTE_TO_MEGA (1/1024.f/1024.f)
#define MEGA_TO_BYTE (1024.f*1024.f)

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
    std::vector<int> prompt_record_;
};

void appendNewPromptRecord(struct TimePerformance* perf, int input_len, bool reuse_kv);

struct PrefillMemPerformance {
    size_t prefill_prev_token_ = 0;
    size_t prefill_token_ = 0;
    float prefill_MB_ = 0;
};

struct DecodeMemPerformance {
    size_t decode_prev_token_ = 0;
    float decode_MB_ = 0;
};

struct MemPerformance {
    std::vector<PrefillMemPerformance> prefill_record_;
    std::vector<DecodeMemPerformance> decode_record_;
};

void mergePerformance(struct TimePerformance* dst, struct TimePerformance* src);
void mergePerformance(struct MemPerformance* dst, struct MemPerformance* src);
void clearPerformance(struct TimePerformance* perf);
void clearPerformance(struct MemPerformance* perf);
} // namespace Transformer
} // namespace MNN
#endif // TRANSFORMER_EVALUATION_hpp