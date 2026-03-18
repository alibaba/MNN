#include <random>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_map>
#include <limits>

#include <MNN/AutoTime.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExecutorScope.hpp>

#include "llm/llm.hpp"
#include "sampler.hpp"
#include "llmconfig.hpp"

namespace MNN {
namespace Transformer {

// SamplerState methods
void SamplerState::ensureProbs(float temperature) {
    if (!probs.empty()) return;
    float invTemp = 1.0f / temperature;
    // find max for numerical stability
    float maxLogit = *std::max_element(logits.begin(), logits.end());
    probs.resize(logits.size());
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp((logits[i] - maxLogit) * invTemp);
        sum += probs[i];
    }
    float invSum = 1.0f / sum;
    for (auto& p : probs) p *= invSum;
}

void SamplerState::invalidateProbs() {
    probs.clear();
}

// Helper: build index→position map for subset
static std::unordered_map<int, int> buildIndexMap(const SamplerState& state) {
    std::unordered_map<int, int> map;
    if (state.is_subset) {
        map.reserve(state.indices.size());
        for (int i = 0; i < (int)state.indices.size(); ++i) {
            map[state.indices[i]] = i;
        }
    }
    return map;
}

// SamplerConfig methods
void Sampler::SamplerConfig::configSampler(std::string sampler_type, std::shared_ptr<LlmConfig> llmConfig) {
    if (sampler_type == "greedy") {
        configGreedy(llmConfig);
    } else if (sampler_type == "temperature") {
        configTemperature(llmConfig);
    } else if (sampler_type == "topK") {
        configTopK(llmConfig);
    } else if (sampler_type == "topP") {
        configTopP(llmConfig);
    } else if (sampler_type == "minP" || sampler_type == "min_p") {
        configMinP(llmConfig);
    } else if (sampler_type == "tfs") {
        configTFS(llmConfig);
    } else if (sampler_type == "typical") {
        configTypical(llmConfig);
    } else if (sampler_type == "penalty") {
        configPenalty(llmConfig);
    } else if (sampler_type == "mixed") {
        configMixed(llmConfig);
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
    repetition_penalty = llmConfig->repetition_penalty();
    presence_penalty = llmConfig->presence_penalty();
    frequency_penalty = llmConfig->frequency_penalty();
    penalty_window = llmConfig->penalty_window();
    ngram = llmConfig->ngram();
    ngram_factor = llmConfig->ngram_factor();
    sampler = llmConfig->penalty_sampler();
    select_type = sampler;
}

void Sampler::SamplerConfig::configMixed(std::shared_ptr<LlmConfig> llmConfig) {
    mixedSamplers = llmConfig->mixed_samplers();
    for (const auto& samplerName : mixedSamplers) {
        configSampler(samplerName, llmConfig);
    }
    // move penalty to front if present
    std::vector<std::string> newSamplers;
    bool hasPenalty = false;
    for (const auto& s : mixedSamplers) {
        if (s != "penalty") {
            newSamplers.push_back(s);
        } else {
            hasPenalty = true;
        }
    }
    if (hasPenalty) {
        newSamplers.insert(newSamplers.begin(), "penalty");
    }
    mixedSamplers = std::move(newSamplers);
    // set select type from last sampler
    if (mixedSamplers.back() == "greedy") {
        select_type = "greedy";
    } else {
        select_type = "temperature";
    }
    // load new config fields
    logit_bias = llmConfig->logit_bias();
    banned_tokens = llmConfig->banned_tokens();
    repetition_penalty = llmConfig->repetition_penalty();
    presence_penalty = llmConfig->presence_penalty();
    frequency_penalty = llmConfig->frequency_penalty();
    penalty_window = llmConfig->penalty_window();
}

Sampler* Sampler::createSampler(std::shared_ptr<LlmContext> context, std::shared_ptr<LlmConfig> config) {
    return new Sampler(context, config);
}

Sampler::Sampler(std::shared_ptr<LlmContext> context, std::shared_ptr<LlmConfig> config)
    : mContext(context), mRng(std::random_device{}()) {
    mConfig.max_all_tokens = config->max_all_tokens();
    mConfig.max_new_tokens = config->max_new_tokens();
    mConfig.type = config->sampler_type();
    mConfig.configSampler(mConfig.type, config);
    buildPipeline();
}

SamplerState Sampler::createState(Express::VARP logits) {
    SamplerState state;
    auto ptr = logits->readMap<float>();
    int lastDim = logits->getInfo()->dim.back();
    state.vocab_size = lastDim;
    state.logits.assign(ptr, ptr + lastDim);
    state.is_subset = false;
    return state;
}

void Sampler::buildPipeline() {
    auto addStep = [this](const std::string& name) {
        if (name == "penalty") {
            mPipeline.push_back([this](SamplerState& s) { stepPenalty(s); });
        } else if (name == "topK") {
            mPipeline.push_back([this](SamplerState& s) { stepTopK(s); });
        } else if (name == "topP") {
            mPipeline.push_back([this](SamplerState& s) { stepTopP(s); });
        } else if (name == "minP" || name == "min_p") {
            mPipeline.push_back([this](SamplerState& s) { stepMinP(s); });
        } else if (name == "tfs") {
            mPipeline.push_back([this](SamplerState& s) { stepTfs(s); });
        } else if (name == "typical") {
            mPipeline.push_back([this](SamplerState& s) { stepTypical(s); });
        }
        // greedy/temperature handled by stepSelect
    };

    if (mConfig.type == "mixed") {
        // logit_bias and banned_tokens before other steps
        if (!mConfig.logit_bias.empty()) {
            mPipeline.push_back([this](SamplerState& s) { stepLogitBias(s); });
        }
        if (!mConfig.banned_tokens.empty()) {
            mPipeline.push_back([this](SamplerState& s) { stepBannedTokens(s); });
        }
        for (const auto& name : mConfig.mixedSamplers) {
            addStep(name);
        }
    } else if (mConfig.type == "greedy") {
        // no filtering steps needed
    } else {
        // single sampler type
        addStep(mConfig.type);
    }
    // final select step
    mPipeline.push_back([this](SamplerState& s) { stepSelect(s); });
}

int Sampler::sample(Express::VARP logits) {
    Timer _t;
    SamplerState state = createState(logits);
    for (auto& step : mPipeline) {
        step(state);
    }
    mContext->sample_us += _t.durationInUs();
    return state.selected_token;
}

// --- Step implementations ---

void Sampler::stepPenalty(SamplerState& state) {
    float repPenalty = mConfig.repetition_penalty;
    float presPenalty = mConfig.presence_penalty;
    float freqPenalty = mConfig.frequency_penalty;
    int ngram = mConfig.ngram;
    float ngram_factor = mConfig.ngram_factor;
    bool penalizeNgram = (ngram_factor > 1.0f);
    if (repPenalty <= 1.0f && presPenalty <= 0.0f && freqPenalty <= 0.0f) return;
    repPenalty = std::min(repPenalty, mConfig.max_penalty);

    const std::vector<int>& prev = mContext->history_tokens;
    if (prev.empty()) return;

    // determine window
    int start = 0;
    if (mConfig.penalty_window > 0 && (int)prev.size() > mConfig.penalty_window) {
        start = (int)prev.size() - mConfig.penalty_window;
    }

    // count occurrences and compute penalty
    std::unordered_map<int, float> penalty_map;
    std::unordered_map<int, int> freq_map;

    // ngram info (reversed order)
    std::vector<int> ngram_info;
    if (penalizeNgram && (int)prev.size() >= ngram) {
        ngram_info.resize(ngram - 1);
        for (int n = 0; n < (int)ngram_info.size(); ++n) {
            ngram_info[n] = prev[prev.size() - 1 - n];
        }
    } else {
        penalizeNgram = false;
    }

    for (int i = start; i < (int)prev.size(); ++i) {
        int tok = prev[i];
        freq_map[tok]++;
        if (penalty_map.count(tok) == 0) {
            penalty_map[tok] = repPenalty;
        }
        if (penalizeNgram) {
            float ngram_penalty = repPenalty;
            for (int j = i - 1; i - j < ngram && j >= 0; --j) {
                int idx = i - j - 1;
                if (prev[j] != ngram_info[idx]) break;
                ngram_penalty *= ngram_factor;
                if (idx == (int)ngram_info.size() - 1) ngram_penalty = mConfig.max_penalty;
            }
            if (ngram_penalty > penalty_map[tok]) penalty_map[tok] = ngram_penalty;
        }
    }

    // build reverse index for subset mode
    auto indexMap = buildIndexMap(state);

    for (auto& kv : penalty_map) {
        int tokenId;
        if (state.is_subset) {
            auto it = indexMap.find(kv.first);
            if (it == indexMap.end()) continue;
            tokenId = it->second;
        } else {
            tokenId = kv.first;
        }
        if (tokenId < 0 || tokenId >= (int)state.logits.size()) continue;
        float& logit = state.logits[tokenId];
        // repetition_penalty: multiplicative
        if (kv.second > 1.0f) {
            logit = (logit >= 0.0f) ? (logit / kv.second) : (logit * kv.second);
        }
        // presence_penalty: additive, applied once per token
        logit -= presPenalty;
        // frequency_penalty: additive, proportional to count
        if (freqPenalty > 0.0f) {
            logit -= freqPenalty * freq_map[kv.first];
        }
    }
    state.invalidateProbs();
}

void Sampler::stepTopK(SamplerState& state) {
    int K = mConfig.topK;
    int size = (int)state.logits.size();
    if (K >= size) return;

    // Use nth_element for O(n) average
    std::vector<int> idx(size);
    std::iota(idx.begin(), idx.end(), 0);
    std::nth_element(idx.begin(), idx.begin() + K, idx.end(),
        [&](int a, int b) { return state.logits[a] > state.logits[b]; });

    std::vector<int> newIndices(K);
    std::vector<float> newLogits(K);
    for (int i = 0; i < K; ++i) {
        newLogits[i] = state.logits[idx[i]];
        newIndices[i] = state.is_subset ? state.indices[idx[i]] : idx[i];
    }
    state.logits = std::move(newLogits);
    state.indices = std::move(newIndices);
    state.is_subset = true;
    state.invalidateProbs();
}

void Sampler::stepTopP(SamplerState& state) {
    float p = mConfig.topP;
    state.ensureProbs(mConfig.temperature);
    int size = (int)state.probs.size();

    // sort by prob descending
    std::vector<int> idx(size);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
        [&](int a, int b) { return state.probs[a] > state.probs[b]; });

    std::vector<int> newIndices;
    std::vector<float> newLogits;
    float cumulative = 0.0f;
    for (int i = 0; i < size; ++i) {
        int id = idx[i];
        cumulative += state.probs[id];
        newIndices.push_back(state.is_subset ? state.indices[id] : id);
        newLogits.push_back(state.logits[id]);
        if (cumulative >= p && !newIndices.empty()) break;
    }
    state.logits = std::move(newLogits);
    state.indices = std::move(newIndices);
    state.is_subset = true;
    state.invalidateProbs();
}

void Sampler::stepMinP(SamplerState& state) {
    float p = mConfig.minP;
    state.ensureProbs(mConfig.temperature);
    int size = (int)state.probs.size();

    // sort by prob descending
    std::vector<int> idx(size);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
        [&](int a, int b) { return state.probs[a] > state.probs[b]; });

    std::vector<int> newIndices;
    std::vector<float> newLogits;
    for (int i = 0; i < size; ++i) {
        int id = idx[i];
        if (state.probs[id] < p && !newIndices.empty()) break;
        newIndices.push_back(state.is_subset ? state.indices[id] : id);
        newLogits.push_back(state.logits[id]);
    }
    state.logits = std::move(newLogits);
    state.indices = std::move(newIndices);
    state.is_subset = true;
    state.invalidateProbs();
}

void Sampler::stepTfs(SamplerState& state) {
    float z = mConfig.tfsZ;
    state.ensureProbs(mConfig.temperature);
    int size = (int)state.probs.size();
    if (size < 3) return;

    // sort by prob descending
    std::vector<int> idx(size);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
        [&](int a, int b) { return state.probs[a] > state.probs[b]; });

    // second derivatives
    std::vector<float> derivatives(size - 2);
    for (int i = 0; i < size - 2; ++i) {
        float first = state.probs[idx[i]] - state.probs[idx[i + 1]];
        float second = state.probs[idx[i + 1]] - state.probs[idx[i + 2]];
        derivatives[i] = std::fabs(first - second);
    }
    // normalize
    float sum = 0.0f;
    for (auto d : derivatives) sum += d;
    if (sum > 0.0f) {
        float invSum = 1.0f / sum;
        for (auto& d : derivatives) d *= invSum;
    }

    std::vector<int> newIndices;
    std::vector<float> newLogits;
    float cumulative = 0.0f;
    for (int i = 0; i < size - 2; ++i) {
        cumulative += derivatives[i];
        if (cumulative >= z && !newIndices.empty()) break;
        int id = idx[i];
        newIndices.push_back(state.is_subset ? state.indices[id] : id);
        newLogits.push_back(state.logits[id]);
    }
    if (newIndices.empty()) {
        // keep at least the top token
        int id = idx[0];
        newIndices.push_back(state.is_subset ? state.indices[id] : id);
        newLogits.push_back(state.logits[id]);
    }
    state.logits = std::move(newLogits);
    state.indices = std::move(newIndices);
    state.is_subset = true;
    state.invalidateProbs();
}

void Sampler::stepTypical(SamplerState& state) {
    float p = mConfig.typical;
    state.ensureProbs(mConfig.temperature);
    int size = (int)state.probs.size();

    // entropy
    float entropy = 0.0f;
    for (int i = 0; i < size; ++i) {
        if (state.probs[i] > 0.0f) {
            entropy -= state.probs[i] * std::log(state.probs[i]);
        }
    }

    // distance from entropy for each token
    std::vector<int> idx(size);
    std::iota(idx.begin(), idx.end(), 0);
    // sort by distance ascending (closest to entropy first)
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        float da = std::fabs(entropy + std::log(state.probs[a]));
        float db = std::fabs(entropy + std::log(state.probs[b]));
        return da < db;
    });

    std::vector<int> newIndices;
    std::vector<float> newLogits;
    float cumulative = 0.0f;
    for (int i = 0; i < size; ++i) {
        int id = idx[i];
        cumulative += state.probs[id];
        newIndices.push_back(state.is_subset ? state.indices[id] : id);
        newLogits.push_back(state.logits[id]);
        if (cumulative >= p && !newIndices.empty()) break;
    }
    state.logits = std::move(newLogits);
    state.indices = std::move(newIndices);
    state.is_subset = true;
    state.invalidateProbs();
}

void Sampler::stepLogitBias(SamplerState& state) {
    if (mConfig.logit_bias.empty()) return;
    if (state.is_subset) {
        auto indexMap = buildIndexMap(state);
        for (const auto& kv : mConfig.logit_bias) {
            auto it = indexMap.find(kv.first);
            if (it != indexMap.end()) {
                state.logits[it->second] += kv.second;
            }
        }
    } else {
        for (const auto& kv : mConfig.logit_bias) {
            if (kv.first >= 0 && kv.first < (int)state.logits.size()) {
                state.logits[kv.first] += kv.second;
            }
        }
    }
    state.invalidateProbs();
}

void Sampler::stepBannedTokens(SamplerState& state) {
    if (mConfig.banned_tokens.empty()) return;
    const float NEG_INF = -std::numeric_limits<float>::infinity();
    if (state.is_subset) {
        auto indexMap = buildIndexMap(state);
        for (int tok : mConfig.banned_tokens) {
            auto it = indexMap.find(tok);
            if (it != indexMap.end()) {
                state.logits[it->second] = NEG_INF;
            }
        }
    } else {
        for (int tok : mConfig.banned_tokens) {
            if (tok >= 0 && tok < (int)state.logits.size()) {
                state.logits[tok] = NEG_INF;
            }
        }
    }
    state.invalidateProbs();
}

void Sampler::stepSelect(SamplerState& state) {
    if (mConfig.select_type == "greedy") {
        // argmax
        int bestIdx = 0;
        float bestScore = state.logits[0];
        for (int i = 1; i < (int)state.logits.size(); ++i) {
            if (state.logits[i] > bestScore) {
                bestScore = state.logits[i];
                bestIdx = i;
            }
        }
        state.selected_token = state.is_subset ? state.indices[bestIdx] : bestIdx;
    } else {
        // temperature random sampling
        state.ensureProbs(mConfig.temperature);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float target = dist(mRng);
        float cumulative = 0.0f;
        int idx = (int)state.probs.size() - 1;
        for (int i = 0; i < (int)state.probs.size(); ++i) {
            cumulative += state.probs[i];
            if (target < cumulative) {
                idx = i;
                break;
            }
        }
        state.selected_token = state.is_subset ? state.indices[idx] : idx;
    }
}

} // Transformer
} // MNN
