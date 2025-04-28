#include <random>
#include <queue>
#include <algorithm>
#include <cmath>
#include <unordered_map>

#include <MNN/AutoTime.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExecutorScope.hpp>

#include "llm/llm.hpp"
#include "sampler.hpp"
#include "llmconfig.hpp"

namespace MNN{
namespace Transformer{

// sampler compute struct start
// a index and its corresponding score
struct IndexScore {
    int index;
    float score;
};
struct IndexScoreCmpLess{
    bool operator()(IndexScore a, IndexScore b) {
        return a.score < b.score;
    }
};
struct IndexScoreCmpGreater{
    bool operator()(IndexScore a, IndexScore b) {
        return a.score > b.score;
    }
};
// a series of index and their corresponding logits
struct SubsetLogits{
    std::vector<int> index;
    MNN::Express::VARP logits;
    bool is_subset;
};
// sampler compute struct end

// sampler compute functions start
Express::VARP _TempratureSoftmax(Express::VARP logits, float temperature, int axis = -1) {
    return Express::_Softmax(logits * Express::_Scalar<float>(1.0f / temperature), axis);
}

SubsetLogits createSubsetLogits(Express::VARP logits) {
    struct SubsetLogits subset;
    subset.logits = logits;
    subset.is_subset = false;
    return subset;
}

SubsetLogits createSubsetLogits(Express::VARP logits, const std::vector<int>& index) {
    struct SubsetLogits subset;
    subset.logits = logits;
    subset.index = index;
    subset.is_subset = true;
    return subset;
}

SubsetLogits createSubsetLogits(int size) {
    struct SubsetLogits subset;
    subset.logits = Express::_Input({size}, Express::NHWC);
    subset.index.resize(size);
    subset.is_subset = true;
    return subset;
}

SubsetLogits createSubsetLogits(const std::vector<float>& scores, const std::vector<int>& index) {
    int size = (int)(index.size());
    struct SubsetLogits subset;
    subset.logits = Express::_Input({size}, Express::NHWC);
    auto pointer = (float*)(subset.logits->writeMap<float>());
    for (int i = 0; i < size; ++i) {
        pointer[i] = scores[i];
    }
    subset.index = index;
    subset.is_subset = true;
    return subset;
}

void transformIndex(struct SubsetLogits& superset, struct SubsetLogits& subset) {
    if (!(superset.is_subset)) return;
    for (auto& id : subset.index) {
        id = superset.index[id];
    }
}

int select(struct SubsetLogits& subset, int id) {
    if (!(subset.is_subset)) {
        return id;
    }
    return subset.index[id];
}

int argmaxSelect(struct SubsetLogits superset) {
    auto scores = (float*)(superset.logits->readMap<float>());
    auto size = superset.logits->getInfo()->size;
    float max_score = scores[0];
    int token_id = 0;
    for (int i = 0; i < size; i++) {
        float score = scores[i];
        if (score > max_score) {
            max_score = score;
            token_id = i;
        }
    }
    return select(superset, token_id);
}

int randomSelect(float* probs, size_t size) {
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

int randomSelect(Express::VARP probs) {
    return randomSelect((float*)(probs->readMap<float>()), probs->getInfo()->size);
}

int reSoftmaxSelect(struct SubsetLogits subset, float temperature) {
    int token_index_id = randomSelect(_TempratureSoftmax(subset.logits, temperature));
    return ((subset.is_subset) ? subset.index[token_index_id] : token_index_id);
}

int packSoftmax(Express::VARP logits, std::vector<IndexScore>& index_scores, float temperature) {
    auto prob_varp = _TempratureSoftmax(logits, temperature);
    auto probs = (float*)(prob_varp->readMap<float>());
    auto size = prob_varp->getInfo()->size;
    index_scores.resize(size);
    for (int i = 0; i < size; i++) {
        IndexScore m;
        m.index = i;
        m.score = probs[i];
        index_scores[i] = m;
    }
    return size;
}
// sampler compute functions end

Sampler* Sampler::createSampler(std::shared_ptr<LlmContext> context, std::shared_ptr<LlmConfig> config) {
    return new Sampler(context, config);
}

Sampler::Sampler(std::shared_ptr<LlmContext> context, std::shared_ptr<LlmConfig> config) {
    mContext = context;
    // mConfig = getSamplerConfig(config);
    mConfig.max_all_tokens = config->max_all_tokens();
    mConfig.max_new_tokens = config->max_new_tokens();
    mConfig.type = config->sampler_type();
    mConfig.configSampler(mConfig.type, config);
}

/* ----------Sampler's members---------- */



/* ----------SamplerConfig---------- */
void Sampler::SamplerConfig::configSampler( std::string sampler_type, std::shared_ptr<LlmConfig> llmConfig) {
    if (sampler_type == "greedy"){
        this->configGreedy(llmConfig);
    } else if (sampler_type == "temperature"){
        this->configTemperature(llmConfig);
    } else if (sampler_type == "topK"){
        this->configTopK(llmConfig);
    } else if (sampler_type == "topP"){
        this->configTopP(llmConfig);
    } else if (sampler_type == "minP"){
        this->configMinP(llmConfig);
    } else if (sampler_type == "tfs"){
        this->configTFS(llmConfig);
    } else if (sampler_type == "typical"){
        this->configTypical(llmConfig);
    } else if (sampler_type == "penalty"){
        this->configPenalty(llmConfig);
    } else if (sampler_type == "mixed"){
        this->configMixed(llmConfig);
    }
}
void Sampler::SamplerConfig::configGreedy(std::shared_ptr<LlmConfig> llmConfig) {
    select_type = "greedy";
}
void Sampler::SamplerConfig::configTemperature(std::shared_ptr<LlmConfig> llmConfig) {
    temperature = llmConfig->temperature();
    select_type = "temperature";
}
void Sampler::SamplerConfig::configTopK(std::shared_ptr<LlmConfig> llmConfig) {
    topK = llmConfig->topK();
    select_type = "temperature";
}
void Sampler::SamplerConfig::configTopP(std::shared_ptr<LlmConfig> llmConfig) {
    topP = llmConfig->topP();
    temperature = llmConfig->temperature();
    select_type = "temperature";
}
void Sampler::SamplerConfig::configMinP(std::shared_ptr<LlmConfig> llmConfig) {
    minP = llmConfig->minP();
    temperature = llmConfig->temperature();
    select_type = "temperature";
}
void Sampler::SamplerConfig::configTFS(std::shared_ptr<LlmConfig> llmConfig) {
    tfsZ = llmConfig->tfsZ();
    temperature = llmConfig->temperature();
    select_type = "temperature";
}
void Sampler::SamplerConfig::configTypical(std::shared_ptr<LlmConfig> llmConfig) {
    typical = llmConfig->typical();
    temperature = llmConfig->temperature();
    select_type = "temperature";
}
void Sampler::SamplerConfig::configPenalty(std::shared_ptr<LlmConfig> llmConfig) {
    penalty = llmConfig->penalty();
    ngram = llmConfig->ngram();
    ngram_factor = llmConfig->ngram_factor();
    sampler = llmConfig->penalty_sampler();
    select_type = sampler;
}
void Sampler::SamplerConfig::configMixed(std::shared_ptr<LlmConfig> llmConfig) {
    mixedSamplers = llmConfig->mixed_samplers();
    // std::cout << "Mixed Sampler Sequence: " << std::flush;
    for (auto samplerName : mixedSamplers) {
        this->configSampler(samplerName, llmConfig);
        // std::cout << samplerName << " " << std::flush;
    }
    // std::cout << std::endl;
    // set select type
    // the final sampler select the token
    if (mixedSamplers.back() == "greedy") select_type = "greedy";
    else if(mixedSamplers.back()=="temperature") select_type = "temperature";
    else select_type = "temperature"; // By default temperature is used.
}

struct SubsetLogits Sampler::topK(struct SubsetLogits superset) {
    int K = mConfig.topK;
    auto scores = (float*)(superset.logits->readMap<float>());
    auto size = superset.logits->getInfo()->size;
    // 1. time complexity: O(nlogk)
    std::priority_queue<IndexScore, std::vector<IndexScore>, IndexScoreCmpGreater> heap;
    for (int i = 0; i < size; i++) {
        IndexScore m;
        m.index = i;
        m.score = scores[i];
        if (heap.size() < K) {
            heap.push(m);
        }
        else {
            if (heap.top().score < m.score) {
                heap.pop();
                heap.push(m);
            }
        }
    }
    // 2. store top K results
    auto subset = createSubsetLogits(K);
    float* topKscores = (float*)(subset.logits->writeMap<float>());
    for (int i = 0; i < K; i++) {
        subset.index[K-i-1] = heap.top().index;
        topKscores[K-i-1]  = heap.top().score;
        heap.pop();
    }
    transformIndex(superset, subset);
    return subset;
}

struct SubsetLogits Sampler::topP(struct SubsetLogits superset) {
    float p = mConfig.topP, temperature = mConfig.temperature;
    std::vector<IndexScore> index_scores;
    int size = packSoftmax(superset.logits, index_scores, temperature);
    // 1. make max heap
    std::make_heap(index_scores.begin(), index_scores.end(), IndexScoreCmpLess());
    // 2. top p algorithm
    auto scores = (float*)(superset.logits->readMap<float>());
    std::vector<int> index;
    std::vector<float> subset_logits;
    float cumulative = 0.0f;
    while (cumulative < p && !index_scores.empty()) {
        std::pop_heap(index_scores.begin(), index_scores.end(), IndexScoreCmpLess());
        IndexScore m = index_scores.back();
        index_scores.pop_back();
        index.push_back(m.index);
        subset_logits.push_back(scores[m.index]);
        cumulative += m.score;
    }
    auto subset = createSubsetLogits(subset_logits, index);
    transformIndex(superset, subset);
    return subset;
}

struct SubsetLogits Sampler::minP(struct SubsetLogits superset) {
    float p = mConfig.minP, temperature = mConfig.temperature;
    std::vector<IndexScore> index_scores;
    int size = packSoftmax(superset.logits, index_scores, temperature);
    // 1. make max heap
    std::make_heap(index_scores.begin(), index_scores.end(), IndexScoreCmpLess());
    // 2. min p algorithm
    auto scores = (float*)(superset.logits->readMap<float>());
    std::vector<int> index;
    std::vector<float> subset_logits;
    for (int i = 0; i < size; ++i) {
        std::pop_heap(index_scores.begin(), index_scores.end(), IndexScoreCmpLess());
        IndexScore m = index_scores.back();
        if (m.score < p && !index.empty()) break;
        index_scores.pop_back();
        index.push_back(m.index);
        subset_logits.push_back(scores[m.index]);
    }
    auto subset = createSubsetLogits(subset_logits, index);
    transformIndex(superset, subset);
    return subset;
}

struct SubsetLogits Sampler::tfs(struct SubsetLogits superset) {
    float z = mConfig.tfsZ, temperature = mConfig.temperature;
    // tfs algorithm
    // 1. softmax
    std::vector<IndexScore> index_scores;
    int size = packSoftmax(superset.logits, index_scores, temperature);
    // 2. sort
    std::sort(index_scores.begin(), index_scores.end(), IndexScoreCmpGreater());
    auto scores = (float*)(superset.logits->readMap<float>());
    // 3. calculate derivatives
    std::vector<float> derivatives(size - 2, 0.0f);
    float first = index_scores[0].score - index_scores[1].score;
    float second = index_scores[1].score - index_scores[2].score;
    for (int i = 0; i < size - 2; ++i) {
        second = index_scores[i+1].score - index_scores[i+2].score;
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
    std::vector<int> index;
    std::vector<float> subset_logits;
    for (int i = 0; i < size - 2; ++i) {
        IndexScore m = index_scores[i];
        cumulative += derivatives[i];
        if (cumulative >= z && !index.empty()) break;
        index.push_back(m.index);
        subset_logits.push_back(scores[m.index]);
    }
    auto subset = createSubsetLogits(subset_logits, index);
    transformIndex(superset, subset);
    return subset;
}

struct SubsetLogits Sampler::typical(struct SubsetLogits superset) {
    float p = mConfig.typical, temperature = mConfig.temperature;
    auto prob_varp = _TempratureSoftmax(superset.logits, temperature);
    auto probs = (float*)(prob_varp->readMap<float>());
    auto size = prob_varp->getInfo()->size;
    std::vector<IndexScore> index_scores;
    index_scores.resize(size);
    // 1. calcaluate dist
    float entropy = 0.0f;
    for (int i = 0; i < size; i++) entropy -= probs[i] * std::log(probs[i]);
    for (int i = 0; i < size; i++) {
        IndexScore m;
        m.index = i;
        m.score = std::fabs(entropy + std::log(probs[i]));
        index_scores[i] = m;
    }
    // 2. make min heap for dist
    std::make_heap(index_scores.begin(), index_scores.end(), IndexScoreCmpGreater());
    // 3. typical p algorithm
    auto scores = (float*)(superset.logits->readMap<float>());
    float cumulative = 0.0f;
    std::vector<int> index;
    std::vector<float> subset_logits;
    for (int i = 0; i < size; ++i) {
        std::pop_heap(index_scores.begin(), index_scores.end(), IndexScoreCmpGreater());
        IndexScore m = index_scores.back();
        cumulative += probs[m.index];
        if (cumulative >= p && !index.empty()) break;
        index_scores.pop_back();
        index.push_back(m.index);
        subset_logits.push_back(scores[m.index]);
    }
    auto subset = createSubsetLogits(subset_logits, index);
    transformIndex(superset, subset);
    return subset;
}

// presence penalty
// no frequency penalty now!
struct SubsetLogits Sampler::penalty(struct SubsetLogits subset) {
    float penalty = mConfig.penalty;
    int ngram = mConfig.ngram;
    float ngram_factor = mConfig.ngram_factor;
    float temperature = mConfig.temperature;
    bool penalizeNgram = (ngram_factor > 1.0f);
    if (penalty <= 1.0f) return subset; // no penalty!
    penalty = std::min(penalty, mConfig.max_penalty);
    // initialization
    std::vector<int>& prev = mContext->history_tokens;
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
    auto scoresMap = (float*)(subset.logits->writeMap<float>());
    for (auto it = penalty_map.begin(); it != penalty_map.end(); ++it) {
        scoresMap[it->first] = (scoresMap[it->first] >= 0.0f) ? (scoresMap[it->first]/it->second) : (scoresMap[it->first]*it->second);
    }
    return subset;
}

struct SubsetLogits Sampler::mixed(struct SubsetLogits subset) {
    for (auto sampler : mConfig.mixedSamplers) {
        subset = subsetSampler(sampler, subset);
    }
    return subset;
}

struct SubsetLogits Sampler::subsetSampler(std::string sampler_type, struct SubsetLogits subset) {
    if (sampler_type == "penalty") subset = penalty(subset);
    if (sampler_type == "topK") subset = topK(subset);
    if (sampler_type == "topP") subset = topP(subset);
    if (sampler_type == "minP") subset = minP(subset);
    if (sampler_type == "tfs") subset = tfs(subset);
    if (sampler_type == "typical") subset = typical(subset);
    if (sampler_type == "mixed") subset = mixed(subset);
    // if greedy and temperate, just let the Selector handle it.
    return subset;
}

int Sampler::handleSelect(struct SubsetLogits subset) {
    if (mConfig.select_type == "greedy") {
        return argmaxSelect(subset);
    } else if(mConfig.select_type =="temperature") {
        return reSoftmaxSelect(subset, mConfig.temperature);
    }
    return 0;
}

int Sampler::sample(Express::VARP logits) {
    Timer _t;
    struct SubsetLogits subset = createSubsetLogits(logits);
    // process subsetSampler
    subset = subsetSampler(mConfig.type, subset);
    // select token from the subset
    int res = handleSelect(subset);
    mContext->sample_us += _t.durationInUs();
    return res;
}

} // Transformer
} // MNN
