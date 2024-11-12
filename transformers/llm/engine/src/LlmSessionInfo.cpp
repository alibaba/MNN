
#include "llm/llm.hpp"

namespace MNN {
namespace Transformer {

// LlmSessionInfo starts
void LlmSessionInfo::resetSamplerFields() {
    all_seq_len_ = 0;
    gen_seq_len_ = 0;
    tokens.clear();
}
void LlmSessionInfo::resetPromptFields() {
    mHistory.clear();
    mInputs.clear();
}
void LlmSessionInfo::resetPerformanceFields() {
    clearPerformance(&mTimePerformance);
}
float LlmSessionInfo::average_total_speed() {
    return (getTotalPromptLen()+getTotalDecodeLen())/(getTotalPrefillTime()+getTotalDecodeTime());
}
float LlmSessionInfo::average_prefill_speed() {
    // prefill response rate
    return getTotalPromptLen()/getTotalPrefillTime();
}
float LlmSessionInfo::average_decode_speed() {
    return getTotalDecodeLen()/getTotalDecodeTime();
}
float LlmSessionInfo::getTotalPrefillTime() {
    float sum = 0.f;
    for (auto record : mTimePerformance.prefill_record_) {
        sum += ((float)record.prefill_us_)*MICRO_TO_SEC;
    }
    return sum;
}
float LlmSessionInfo::getTotalDecodeTime() {
    float sum = 0.0f;
    for (auto record : mTimePerformance.decode_record_) {
        sum += ((float)record.decode_us_)*MICRO_TO_SEC;
    }
    return sum;
}
int LlmSessionInfo::getTotalPromptLen() {
    int prompt_len = 0;
    if (mTimePerformance.prefill_record_.size() != mTimePerformance.prompt_record_.size()) {
        for (auto record : mTimePerformance.prefill_record_) {
            prompt_len += record.prefill_token_;
        }
    } else {
        for (int r=0; r < mTimePerformance.prompt_record_.size(); ++r) {
            prompt_len += mTimePerformance.prompt_record_[r];
        }
    } 
    return prompt_len;
}
int LlmSessionInfo::getTotalDecodeLen() {
    return mTimePerformance.decode_record_.size();
}
void LlmSessionInfo::print_speed(std::ostream* os) {
    (*os) << "prefill " << mTimePerformance.prefill_record_.size() << std::endl;
    if (mTimePerformance.prefill_record_.size() != mTimePerformance.prompt_record_.size()) {
        (*os) << "prev_token input_token speed(token/s)" << std::endl;
        for (auto record : mTimePerformance.prefill_record_) {
            (*os) << record.prefill_prev_token_ << " " << record.prefill_token_ << " " << record.prefill_token_/(((float)record.prefill_us_)*MICRO_TO_SEC) << std::endl;
        }
    } else {
        (*os) << "prev_token input_token prompt_token response_speed(token/s)" << std::endl;
        for (int r=0; r < mTimePerformance.prompt_record_.size(); ++r) {
            auto record = mTimePerformance.prefill_record_[r];
            auto prompt_len = mTimePerformance.prompt_record_[r];
            (*os) << record.prefill_prev_token_ << " " << record.prefill_token_ << " " << prompt_len << " " << prompt_len/(((float)record.prefill_us_)*MICRO_TO_SEC) << std::endl;
        }
    }
    (*os) << "decode " << mTimePerformance.decode_record_.size() << std::endl;
    (*os) << "prev_token speed(token/s)" << std::endl;
    for (auto record : mTimePerformance.decode_record_) {
        (*os) << record.decode_prev_token_ << " " << 1./(((float)record.decode_us_)*MICRO_TO_SEC) << std::endl;
    }
}

} // Transformer
} // MNN