
#include <MNN/expr/ExecutorScope.hpp>
#include "llm/llm.hpp"
#include "sampler/sampler.hpp"

namespace MNN{
namespace Transformer{

LocalSampler::LocalSampler(Llm* llm, StateCacheManager* manager, int max_new_tokens, std::string type) {
    mLlm = llm;
    mStateCacheManager.reset(manager);
    std::vector<int> history_ids_;
    std::shared_ptr<StateCacheReference> reference = manager->onCreateReference();
    mCandidates.emplace_back(std::make_pair(history_ids_, reference)); // for LocalSampler, reference have never been modified manually.
    mCommonPrefix = history_ids_;
    mMaxNewTokens = max_new_tokens;
    mType = type;
}


int LocalSampler::argmax(MNN::Express::VARP logits) {
    auto scores = (float*)(logits->readMap<float>());
    float max_score = 0;
    int token_id = 0;
    for (int i = 0; i < logits->getInfo()->size; i++) {
        float score = scores[i];
        if (score > max_score) {
            max_score = score;
            token_id = i;
        }
    }
    return token_id;
}


int LocalSampler::algorithm(MNN::Express::VARP logits) {
    if (mType == "greedy") return argmax(logits);
}

std::string LocalSampler::handleToken(int token, std::ostream* os, const char* end_with) {
    // CommonPrefix and Candidates managements
    mCandidates[0].first.push_back(token);
    mCommonPrefix.push_back(token);
    std::string output_str = mLlm->decode(mCommonPrefix.back());
    // print
    *os << output_str << std::flush;
    return output_str;
}

std::string LocalSampler::sample(const std::vector<int>& input_ids, std::ostream* os, const char* end_with, struct timePerformance* perf) {
    // initialization
    std::string output_str; 
    mStateCacheManager->setCurrentReference(mCandidates[0].second);
    mCommonPrefix.insert(mCommonPrefix.begin(), input_ids.begin(), input_ids.end());
    // prefill 
    auto st = std::chrono::system_clock::now();
    auto logits = mLlm->forward(input_ids, true);
    if (nullptr == logits.get()) {
        return "";
    }
    int token = algorithm(logits);
    auto et = std::chrono::system_clock::now();
    perf->prefill_us_ = std::chrono::duration_cast<std::chrono::microseconds>(et - st).count();
    output_str += handleToken(token, os, end_with);
    // decode
    while (getGenLength(0, output_str.size()) < mMaxNewTokens) {
        st = std::chrono::system_clock::now();
        // next token
        logits = mLlm->forward({mCandidates[0].first.back()}, false);
        if (nullptr == logits.get()) {
            return output_str;
        }
        if (logits->getInfo()->size == 0) {
            return output_str;
        }
        token = algorithm(logits);
        et = std::chrono::system_clock::now();
        perf->decode_us_ += std::chrono::duration_cast<std::chrono::microseconds>(et - st).count();
        if (mLlm->is_stop(token)) {
            *os << end_with << std::flush;
            break;
        } else {
            output_str += handleToken(token);
        }
    }
    // return output_str
    return output_str;
}


} // Transformer
} // MNN