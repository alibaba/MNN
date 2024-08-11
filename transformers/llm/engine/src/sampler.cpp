#include <random>
#include <queue>
#include <algorithm>

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
    std::cout << "in select" << std::endl;
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<float> distribution(0.0, 1.0);
    float target = distribution(generator);
    float cumulative = 0.0;
    std::cout << size << " " << target << " " << cumulative << std::endl;
    for (int i = 0; i < size; i++) {
        cumulative += probs[i];
        if (target < cumulative) {
            return i;
        }
    }
    return size - 1;
}

int LocalSampler::temperature(MNN::Express::VARP logits, float temperature) {
    std::cout << "temperature" << std::endl;
    std::cout << temperature << std::endl;
    std::cout << logits->readMap<float>()[0] << std::endl;
    std::cout << logits->readMap<float>()[1] << std::endl;
    std::cout << logits->readMap<float>()[2] << std::endl;
    std::cout << logits->getInfo()->size << std::endl;
    // logits = MNN::Express::_TempratureSoftmax(logits, 100.);
    logits = MNN::Express::_Softmax(logits);
    std::cout << "before random" << std::endl;
    std::cout << logits->readMap<float>()[0] << std::endl;
    return randomSelect((float*)(logits->readMap<float>()), logits->getInfo()->size);
}

int LocalSampler::reSoftmaxSelect(std::vector<int> index, std::vector<float> scores, float temperature) {
    auto varp = MNN::Express::_Input({index.size()}, MNN::Express::NHWC);
    auto scoresMap = (float*)(varp->writeMap<float>());
    for (int i = 0; i < index.size(); ++i) {
        scoresMap[i] = scores[i];
    }
    int token_index_id = randomSelect((float*)(MNN::Express::_TempratureSoftmax(varp, temperature)->readMap<float>()), index.size());
    return index[token_index_id];
}

void LocalSampler::topK(MNN::Express::VARP logits, int K, std::vector<int>& topKindex, std::vector<float>& topKprob) {
    auto scores = (float*)(logits->readMap<float>());
    auto size = logits->getInfo()->size;
    // time complexity: O(nlogk)
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
    // store top K results
    topKindex.clear();
    topKindex.resize(K);
    topKprob.clear();
    topKprob.resize(K);
    for (int i = 0; i < K; i++) {
        topKindex[K-i-1] = heap.top().index;
        topKprob[K-i-1]  = heap.top().prob;
        heap.pop();
    }
}

int LocalSampler::topK(MNN::Express::VARP logits, int K, float temperature) {
    // top K operation
    std::vector<int> topKindex;
    std::vector<float> topKscores;
    topK(logits, K, topKindex, topKscores);
    // apply Softmax and select
    return reSoftmaxSelect(topKindex, topKscores, temperature);
}

void LocalSampler::topP(MNN::Express::VARP logits, float p, float temperature, std::vector<int>& topPindex, std::vector<float>& topPprob) {
    auto prob = MNN::Express::_TempratureSoftmax(logits, temperature);
    // make max heap
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
    // top p algorithm
    scores = (float*)(logits->readMap<float>());
    float cumulative = 0.0; 
    while (cumulative < p) {
        std::pop_heap(score_vector.begin(), score_vector.end(), IndexProbCmpLess());
        IndexProb m = score_vector.back();
        score_vector.pop_back();
        topPindex.push_back(m.index);
        topPprob.push_back(scores[m.index]);
        cumulative += m.prob;
    }
}


int LocalSampler::topP(MNN::Express::VARP logits, float p, float temperature) {
    // top p operation
    std::vector<int> topPindex;
    std::vector<float> topPscores;
    topP(logits, p, temperature, topPindex, topPscores);
    // apply Softmax and select
    return reSoftmaxSelect(topPindex, topPscores, temperature);
}

void LocalSampler::minP(MNN::Express::VARP logits, float p, float temperature, std::vector<int>& minPindex, std::vector<float>& minPprob) {
    auto prob = MNN::Express::_TempratureSoftmax(logits, temperature);
    // make max heap
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
    // min p algorithm
    scores = (float*)(logits->readMap<float>());
    for (int i = 0; i < size; ++i) {
        std::pop_heap(score_vector.begin(), score_vector.end(), IndexProbCmpLess());
        IndexProb m = score_vector.back();
        if (m.prob < p && minPindex.size() != 0) break;
        score_vector.pop_back();
        minPindex.push_back(m.index);
        minPprob.push_back(scores[m.index]);
    }
}


int LocalSampler::minP(MNN::Express::VARP logits, float p, float temperature) {
    // top p operation
    std::vector<int> minPindex;
    std::vector<float> minPscores;
    minP(logits, p, temperature, minPindex, minPscores);
    // apply Softmax and select
    return reSoftmaxSelect(minPindex, minPscores, temperature);
}


int LocalSampler::argmax(MNN::Express::VARP logits) {
    std::cout << logits->readMap<float>()[0] << std::endl;
    std::cout << logits->readMap<float>()[1] << std::endl;
    std::cout << logits->readMap<float>()[2] << std::endl;
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


int LocalSampler::algorithm(MNN::Express::VARP logits) {
    if (mConfig.type == "greedy") return argmax(logits);
    if (mConfig.type == "temperature") return temperature(logits, mConfig.temperature);
    if (mConfig.type == "topK") return topK(logits, mConfig.topK);
    if (mConfig.type == "topP") return topP(logits, mConfig.topP);
    if (mConfig.type == "minP") return minP(logits, mConfig.minP);
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
    printf("start prefill\n");
    auto logits = mLlm->forward(input_ids, true);
    // printf("sampler algorithm prefill\n");
    if (nullptr == logits.get()) {
        return "";
    }
    std::cout << "pointer valid" << std::endl;
    int token = algorithm(logits);
    auto et = std::chrono::system_clock::now();
    perf->prefill_us_ = std::chrono::duration_cast<std::chrono::microseconds>(et - st).count();
    std::cout << "sampler algorithm prefill finish" << std::endl;
    output_str += handleToken(token, os, end_with);
    // std::cout << output_str << std::endl;
    // std::vector<int> no;
    // std::cout << no[10] << std::endl;
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