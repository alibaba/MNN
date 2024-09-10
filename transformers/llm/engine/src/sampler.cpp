#include <random>
#include <queue>
#include <algorithm>
#include <cmath>
#include <unordered_map>

#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include "llm/llm.hpp"
#include "sampler/sampler.hpp"

namespace MNN{
namespace Transformer{

LocalSampler::LocalSampler(Llm* llm, StateCacheManager* manager, int max_new_tokens, struct LocalSamplerConfig config) {
    mLlm = llm;
    mStateCacheManager = manager;
    std::vector<int> history_ids_;
    std::shared_ptr<StateCacheReference> reference = manager->onCreateReference();
    mCandidates.emplace_back(std::make_pair(history_ids_, reference)); // for LocalSampler, reference have never been modified manually.
    mCommonPrefix = history_ids_;
    mMaxNewTokens = max_new_tokens;
    mConfig = config;
}

int LocalSampler::randomSelect(float* probs, size_t size) {
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<float> distribution(0.0, 1.0);
    float target = distribution(generator);
    float cumulative = 0.0;
    for (int i = 0; i < size; i++) {
        cumulative += probs[i];
        if (target < cumulative) {
            return i;
        }
    }
    return size - 1;
}

int LocalSampler::temperature(MNN::Express::VARP logits, float temperature) {
    logits = MNN::Express::_TempratureSoftmax(logits, temperature);
    return randomSelect((float*)(logits->readMap<float>()), logits->getInfo()->size);
}

int LocalSampler::reSoftmaxSelect(std::vector<int> index, std::vector<float> scores, float temperature) {
    auto varp = MNN::Express::_Input({(int)index.size()}, MNN::Express::NHWC);
    auto scoresMap = (float*)(varp->writeMap<float>());
    for (int i = 0; i < index.size(); ++i) {
        scoresMap[i] = scores[i];
    }
    int token_index_id = randomSelect((float*)(MNN::Express::_TempratureSoftmax(varp, temperature)->readMap<float>()), index.size());
    return index[token_index_id];
}

void LocalSampler::topK(MNN::Express::VARP logits, int K, std::vector<int>& index, std::vector<float>& topKprob) {
    auto scores = (float*)(logits->readMap<float>());
    auto size = logits->getInfo()->size;
    // 1. time complexity: O(nlogk)
    std::priority_queue<IndexProb, std::vector<IndexProb>, IndexProbCmpGreater> heap;
    for (int i = 0; i < size; i++) {
        IndexProb m;
        m.index = i;
        m.prob = scores[i];
        if (heap.size() < K) {
            heap.push(m);
        } 
        else {
            if (heap.top().prob < m.prob) {
                heap.pop();
                heap.push(m);
            }
        }
    }
    // 2. store top K results
    index.clear();
    index.resize(K);
    topKprob.clear();
    topKprob.resize(K);
    for (int i = 0; i < K; i++) {
        index[K-i-1] = heap.top().index;
        topKprob[K-i-1]  = heap.top().prob;
        heap.pop();
    }
}

void LocalSampler::topP(MNN::Express::VARP logits, float p, float temperature, std::vector<int>& index, std::vector<float>& topPprob) {
    auto prob = MNN::Express::_TempratureSoftmax(logits, temperature);
    // 1. make max heap
    auto scores = (float*)(prob->readMap<float>());
    auto size = prob->getInfo()->size;
    std::vector<IndexProb> score_vector;
    score_vector.resize(size);
    for (int i = 0; i < size; i++) {
        IndexProb m;
        m.index = i;
        m.prob = scores[i];
        score_vector[i] = m;
    }
    std::make_heap(score_vector.begin(), score_vector.end(), IndexProbCmpLess());
    // 2. top p algorithm
    scores = (float*)(logits->readMap<float>());
    float cumulative = 0.0f; 
    while (cumulative < p && !score_vector.empty()) {
        std::pop_heap(score_vector.begin(), score_vector.end(), IndexProbCmpLess());
        IndexProb m = score_vector.back();
        score_vector.pop_back();
        index.push_back(m.index);
        topPprob.push_back(scores[m.index]);
        cumulative += m.prob;
    }
}

void LocalSampler::minP(MNN::Express::VARP logits, float p, float temperature, std::vector<int>& index, std::vector<float>& minPprob) {
    auto prob = MNN::Express::_TempratureSoftmax(logits, temperature);
    // 1. make max heap
    auto scores = (float*)(prob->readMap<float>());
    auto size = prob->getInfo()->size;
    std::vector<IndexProb> score_vector;
    score_vector.resize(size);
    for (int i = 0; i < size; i++) {
        IndexProb m;
        m.index = i;
        m.prob = scores[i];
        score_vector[i] = m;
    }
    std::make_heap(score_vector.begin(), score_vector.end(), IndexProbCmpLess());
    // 2. min p algorithm
    scores = (float*)(logits->readMap<float>());
    for (int i = 0; i < size; ++i) {
        std::pop_heap(score_vector.begin(), score_vector.end(), IndexProbCmpLess());
        IndexProb m = score_vector.back();
        if (m.prob < p && !index.empty()) break;
        score_vector.pop_back();
        index.push_back(m.index);
        minPprob.push_back(scores[m.index]);
    }
}

void LocalSampler::tfs(MNN::Express::VARP logits, float z, float temperature, std::vector<int>& index, std::vector<float>& tfsprob) {
    // tfs algorithm
    auto prob = MNN::Express::_TempratureSoftmax(logits, temperature);
    // 1. softmax
    auto scores = (float*)(prob->readMap<float>());
    auto size = prob->getInfo()->size;
    std::vector<IndexProb> score_vector;
    score_vector.resize(size);
    for (int i = 0; i < size; i++) {
        IndexProb m;
        m.index = i;
        m.prob = scores[i];
        score_vector[i] = m;
    }
    // 2. sort
    std::sort(score_vector.begin(), score_vector.end(), IndexProbCmpGreater());
    scores = (float*)(logits->readMap<float>());
    // 3. calculate derivatives
    std::vector<float> derivatives(size - 2, 0.0f);
    float first = score_vector[0].prob - score_vector[1].prob;
    float second = score_vector[1].prob - score_vector[2].prob;
    for (int i = 0; i < size - 2; ++i) {
        second = score_vector[i+1].prob - score_vector[i+2].prob;
        derivatives[i] = std::fabs(first - second);
        first = second;
    }
    // 4. normalize derivatives
    float derivatives_sum = 0.0;
    for (int i = 0; i < size - 2; ++i) derivatives_sum += derivatives[i];
    float derivatives_sum_rec = 1.0f / derivatives_sum;
    for (int i = 0; i < size - 2; ++i) derivatives[i] *= derivatives_sum_rec;
    // 5. cumulate, discard last 2 for sure.
    float cumulative = 0.0; 
    for (int i = 0; i < size - 2; ++i) {
        IndexProb m = score_vector[i];
        cumulative += derivatives[i];
        if (cumulative >= z && !index.empty()) break;
        index.push_back(m.index);
        tfsprob.push_back(scores[m.index]);
    }
}

void LocalSampler::typical(MNN::Express::VARP logits, float p, float temperature, std::vector<int>& index, std::vector<float>& minPprob) {
    auto prob = MNN::Express::_TempratureSoftmax(logits, temperature);
    auto scores = (float*)(prob->readMap<float>());
    auto size = prob->getInfo()->size;
    std::vector<IndexProb> score_vector;
    score_vector.resize(size);
    // 1. calcaluate dist
    float entropy = 0.0f;
    for (int i = 0; i < size; i++) entropy -= scores[i] * std::log(scores[i]);
    for (int i = 0; i < size; i++) {
        IndexProb m;
        m.index = i;
        m.prob = std::fabs(entropy + std::log(scores[i]));
        score_vector[i] = m;
    }
    // 2. make min heap for dist
    std::make_heap(score_vector.begin(), score_vector.end(), IndexProbCmpGreater());
    // 3. typical p algorithm
    auto probs = (float*)(prob->readMap<float>());
    scores = (float*)(logits->readMap<float>());
    float cumulative = 0.0f;
    for (int i = 0; i < size; ++i) {
        std::pop_heap(score_vector.begin(), score_vector.end(), IndexProbCmpGreater());
        IndexProb m = score_vector.back();
        cumulative += probs[m.index];
        if (cumulative >= p && !index.empty()) break;
        score_vector.pop_back();
        index.push_back(m.index);
        minPprob.push_back(scores[m.index]);
    }
}

int LocalSampler::topK(MNN::Express::VARP logits, int K, float temperature) {
    // top K operation
    std::vector<int> index;
    std::vector<float> topKscores;
    topK(logits, K, index, topKscores);
    // apply Softmax and select
    return reSoftmaxSelect(index, topKscores, temperature);
}

int LocalSampler::topP(MNN::Express::VARP logits, float p, float temperature) {
    // top p operation
    std::vector<int> index;
    std::vector<float> topPscores;
    topP(logits, p, temperature, index, topPscores);
    // apply Softmax and select
    return reSoftmaxSelect(index, topPscores, temperature);
}

int LocalSampler::minP(MNN::Express::VARP logits, float p, float temperature) {
    // top p operation
    std::vector<int> index;
    std::vector<float> minPscores;
    minP(logits, p, temperature, index, minPscores);
    // apply Softmax and select
    return reSoftmaxSelect(index, minPscores, temperature);
}

int LocalSampler::tfs(MNN::Express::VARP logits, float z, float temperature) {
    // top p operation
    std::vector<int> index;
    std::vector<float> scores;
    tfs(logits, z, temperature, index, scores);
    // apply Softmax and select
    return reSoftmaxSelect(index, scores, temperature);
}

int LocalSampler::typical(MNN::Express::VARP logits, float p, float temperature) {
    // top p operation
    std::vector<int> index;
    std::vector<float> scores;
    typical(logits, p, temperature, index, scores);
    // apply Softmax and select
    return reSoftmaxSelect(index, scores, temperature);
}

int LocalSampler::argmax(MNN::Express::VARP logits) {
    auto scores = (float*)(logits->readMap<float>());
    auto size = logits->getInfo()->size;
    float max_score = 0;
    int token_id = 0;
    for (int i = 0; i < size; i++) {
        float score = scores[i];
        if (score > max_score) {
            max_score = score;
            token_id = i;
        }
    }
    return token_id;
}

// no frequency penalty now!
void LocalSampler::penalty(MNN::Express::VARP logits, float penalty, bool penalizeNgram, int ngram, float ngram_factor) {
    if (penalty <= 1.0f) return; // no penalty!
    if (ngram_factor <= 1.0f) penalizeNgram = false;
    penalty = std::min(penalty, mConfig.max_penalty);
    // initialization
    std::vector<int>& prev = mCandidates[0].first;
    std::unordered_map<int, float> penalty_map;
    // 1. local ngram info, reversed order
    std::vector<int> ngram_info(ngram-1);
    if (penalizeNgram) {
        for (int n = 0; n < ngram_info.size(); ++n) {
            ngram_info[n] = prev[prev.size()-1-n];
        }
    }
    // 2. generate penalty map
    for (int i = 0; i < prev.size(); ++i) {
        if (penalty_map.count(prev[i]) == 0) penalty_map[prev[i]] = penalty;
        if (penalizeNgram) {
            float ngram_penalty = penalty;
            for (int j = i-1; i-j < ngram && j>=0; --j) {
                int idx = i-j-1;
                if (prev[j] != ngram_info[idx]) break;
                ngram_penalty *= ngram_factor;
                // no repeat larger than ngram!
                if (idx == ngram_info.size()-1) ngram_penalty = mConfig.max_penalty;
            }
            if (ngram_penalty > penalty_map[prev[i]]) penalty_map[prev[i]] = ngram_penalty;
        }
    }
    // 3. penalize logits according to penalty_map
    auto scoresMap = (float*)(logits->writeMap<float>());
    for (auto it = penalty_map.begin(); it != penalty_map.end(); ++it) {
        scoresMap[it->first] = (scoresMap[it->first] >= 0.0f) ? (scoresMap[it->first]/it->second) : (scoresMap[it->first]*it->second);
    }
}


int LocalSampler::penalty(MNN::Express::VARP logits, float penalty, int ngram, float ngram_factor, float temperature) {
    bool penalizeNgram = (mConfig.type == "penalize_ngram");
    this->penalty(logits, penalty, penalizeNgram, ngram, ngram_factor);
    return this->temperature(logits, temperature);
}

int LocalSampler::algorithm(MNN::Express::VARP logits) {
    int res = 0;
    if (mConfig.type == "greedy") res = argmax(logits);
    if (mConfig.type == "temperature") res = temperature(logits, mConfig.temperature);
    if (mConfig.type == "topK") res = topK(logits, mConfig.topK);
    if (mConfig.type == "topP") res = topP(logits, mConfig.topP);
    if (mConfig.type == "minP") res = minP(logits, mConfig.minP);
    if (mConfig.type == "tfs") res = tfs(logits, mConfig.tfsZ);
    if (mConfig.type == "typical") res = typical(logits, mConfig.typical);
    if (mConfig.type == "penalty" || mConfig.type == "penalize_ngram") res = penalty(logits, mConfig.penalty, mConfig.ngram, mConfig.ngram_factor, mConfig.temperature);
    // if (mConfig.type == "mixed") res = mixed(logits);
    Express::ExecutorScope::Current()->gc(Express::Executor::FULL);
    return res;
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

std::string LocalSampler::sample(const std::vector<int>& input_ids, std::ostream* os, const char* end_with, struct TimePerformance* time_perf, struct MemPerformance* mem_perf, struct MemoryInfo* init_mem) {
    // initialization for time and memory performance
    PrefillTimePerformance prefill_time;
    PrefillMemPerformance prefill_mem;
    prefill_mem.prefill_prev_token_ = prefill_time.prefill_prev_token_ = mCommonPrefix.size();
    prefill_mem.prefill_token_ = prefill_time.prefill_token_ = input_ids.size();
    MemoryInfo now_mem;
    // initialization
    std::string output_str; 
    mStateCacheManager->setCurrentReference(mCandidates[0].second);
    mCandidates[0].first.insert(mCandidates[0].first.end(), input_ids.begin(), input_ids.end());
    mCommonPrefix.insert(mCommonPrefix.end(), input_ids.begin(), input_ids.end());
    // prefill 
    auto st = std::chrono::system_clock::now();
    auto logits = mLlm->forward(input_ids, true);
    if (nullptr == logits.get()) {
        return "";
    }
    int token = algorithm(logits);
    // record time and memory
    auto et = std::chrono::system_clock::now();
    prefill_time.prefill_us_ = std::chrono::duration_cast<std::chrono::microseconds>(et - st).count();
    readMemInfo(&now_mem);
    readProcStatus(&now_mem);
    // prefill_mem.prefill_MB_ = getSysMemInc(init_mem, &now_mem);
    prefill_mem.prefill_MB_ = getProcMem(&now_mem);
    time_perf->prefill_record_.push_back(prefill_time);
    mem_perf->prefill_record_.push_back(prefill_mem);
    // handle the new token
    output_str += handleToken(token, os, end_with);
    // decode
    while (getGenLength(0, output_str.size()) < mMaxNewTokens) {
        DecodeTimePerformance decode_time;
        DecodeMemPerformance decode_mem;
        decode_mem.decode_prev_token_ = decode_time.decode_prev_token_ = mCandidates[0].first.size();
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
        decode_time.decode_us_ = std::chrono::duration_cast<std::chrono::microseconds>(et - st).count();
        readMemInfo(&now_mem);
        readProcStatus(&now_mem);
        // decode_mem.decode_MB_ = getSysMemInc(init_mem, &now_mem); 
        decode_mem.decode_MB_ = getProcMem(&now_mem); 
        time_perf->decode_record_.push_back(decode_time);
        mem_perf->decode_record_.push_back(decode_mem);
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

void LocalSampler::reset() {
    // in the future, only reset its own.
    mStateCacheManager->clear();
    mCandidates.clear();
    std::vector<int> history_ids_;
    std::shared_ptr<StateCacheReference> reference = mStateCacheManager->onCreateReference();
    mCandidates.emplace_back(std::make_pair(history_ids_, reference)); // for LocalSampler, reference have never been modified manually.
    mCommonPrefix = history_ids_;
}


} // Transformer
} // MNN