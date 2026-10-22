//
//  token_compressor.hpp
//
//  Created by MNN on 2026/10/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_LLM_TOKEN_COMPRESSOR_HPP
#define MNN_LLM_TOKEN_COMPRESSOR_HPP
#ifdef MNN_LLM_TOKEN_COMPRESS

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <queue>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace MNN {
namespace Transformer {

class Tokenizer;

namespace TokenCompression {

std::vector<int> encodePrompt(Tokenizer* tokenizer, const std::string& text, int maxTokens);
std::vector<std::vector<int>> sentenceDelimiters(Tokenizer* tokenizer);

// Private token-only sentence selection. Delimiters are encoded once by the caller.
class Compressor {
public:
    using Tokens = std::vector<int>;
    using Span = std::pair<size_t, size_t>;
    struct Options {
        size_t maxTokens = 64000;
        size_t boundaryTokens = 512;
        size_t blockTokens = 4096;
        size_t blocksPerGroup = 4;
    };
    struct Result {
        Tokens tokens;
        std::vector<Span> spans;
        std::vector<size_t> selected;
        size_t reserveUnits = 0;
    };

    Compressor(const std::vector<Tokens>& delimiters, Tokens gap) : mGap(std::move(gap)) {
        for (const auto& ids : delimiters) {
            if (ids.empty()) {
                continue;
            }
            auto& entries = mDelimiters[ids.front()];
            if (std::find(entries.begin(), entries.end(), ids) == entries.end()) {
                entries.push_back(ids);
            }
        }
    }

    bool compress(const Tokens& source, const Options& options, Result& result) const {
        result = Result();
        if (!options.maxTokens || !options.blockTokens || !options.blocksPerGroup ||
            std::any_of(source.begin(), source.end(), [](int id) { return id < 0; })) {
            return false;
        }
        if (source.size() <= options.maxTokens) {
            result.tokens = source;
            if (!source.empty()) {
                result.spans.push_back({0, source.size()});
                result.selected.push_back(0);
            }
            return true;
        }
        result.spans = split(source);
        std::vector<Unit> units(result.spans.size());
        std::vector<Group> groups;
        std::unordered_map<uint64_t, int> featureIds;
        std::vector<size_t> globalDf;
        std::vector<bool> bigrams;
        size_t block = 0, blockCost = 0;
        for (size_t i = 0; i < units.size(); ++i) {
            auto& u = units[i];
            u.cost = result.spans[i].second - result.spans[i].first;
            if (blockCost && blockCost + u.cost > options.blockTokens) {
                ++block;
                blockCost = 0;
            }
            blockCost += u.cost;
            size_t group = block / options.blocksPerGroup;
            if (groups.size() <= group) {
                groups.resize(group + 1);
            }
            groups[group].members.push_back(i);
            groups[group].cost += u.cost;
            for (size_t p = result.spans[i].first; p < result.spans[i].second; ++p) {
                for (int order = 0; order < 2 && p + order < result.spans[i].second; ++order) {
                    uint64_t key =
                        (uint64_t(uint32_t(source[p])) << 32) | (order ? uint32_t(source[p + 1]) : UINT32_MAX);
                    auto entry = featureIds.emplace(key, static_cast<int>(featureIds.size()));
                    if (entry.second) {
                        globalDf.push_back(0);
                        bigrams.push_back(order != 0);
                    }
                    u.features.push_back(entry.first->second);
                }
            }
            std::sort(u.features.begin(), u.features.end());
            u.features.erase(std::unique(u.features.begin(), u.features.end()), u.features.end());
            for (int f : u.features) {
                ++globalDf[f];
            }
        }
        for (auto& group : groups) {
            prepare(group, units, bigrams);
        }

        std::set<size_t> chosen;
        int64_t used = 0;
        auto take = [&](size_t i, int64_t limit) {
            if (chosen.count(i)) {
                return true;
            }
            auto right = chosen.lower_bound(i);
            bool leftAdjacent = right != chosen.begin() && *std::prev(right) + 1 == i;
            bool rightAdjacent = right != chosen.end() && i + 1 == *right;
            int64_t extra = static_cast<int64_t>(units[i].cost);
            if (!chosen.empty()) {
                extra += static_cast<int64_t>(mGap.size()) * (1 - int(leftAdjacent) - int(rightAdjacent));
            }
            if (used + extra > limit) {
                return false;
            }
            chosen.insert(i);
            used += extra;
            return true;
        };
        for (int direction = 0; direction < 2; ++direction) {
            size_t spent = 0;
            for (size_t n = 0; n < units.size() && spent < options.boundaryTokens; ++n) {
                size_t i = direction ? units.size() - 1 - n : n;
                if (!take(i, options.maxTokens)) {
                    return false;
                }
                spent += units[i].cost;
            }
        }

        struct Rarity {
            double score;
            double efficiency;
            size_t index;
        };
        std::vector<Rarity> rare;
        for (size_t i = 0; i < units.size(); ++i) {
            double top[4] = {};
            double totalIdf = 0;
            for (int f : units[i].features) {
                double idf = std::log(double(units.size() + 1) / (globalDf[f] + 1));
                totalIdf += idf;
                auto least = std::min_element(top, top + 4);
                if (idf > *least) {
                    *least = idf;
                }
            }
            double score = 0;
            for (double value : top) {
                score += value;
            }
            score /= std::max<size_t>(1, std::min<size_t>(4, units[i].features.size()));
            double efficiency = totalIdf / std::max<size_t>(1, units[i].features.size()) /
                                std::sqrt(double(units[i].cost) + mGap.size());
            rare.push_back({score, efficiency, i});
        }
        std::sort(rare.begin(), rare.end(), [](const Rarity& a, const Rarity& b) {
            if (a.score != b.score) {
                return a.score > b.score;
            }
            return a.efficiency != b.efficiency ? a.efficiency > b.efficiency : a.index < b.index;
        });
        // Count actual incremental tokens, including gaps and adjacent-run merges.
        int64_t reserveLimit = std::min<int64_t>(options.maxTokens, used + options.maxTokens / 5);
        for (const auto& rank : rare) {
            if (!chosen.count(rank.index) && take(rank.index, reserveLimit)) {
                ++result.reserveUnits;
            }
        }

        std::vector<size_t> positions(groups.size(), 0);
        std::priority_queue<Rank> queue;
        for (size_t g = 0; g < groups.size(); ++g) {
            size_t spent = 0;
            size_t floor = static_cast<size_t>(double(groups[g].cost) / source.size() * options.maxTokens / 2);
            auto& p = positions[g];
            const auto& steps = groups[g].steps;
            while (p < steps.size() && spent + units[steps[p].index].cost <= floor) {
                if (take(steps[p].index, options.maxTokens)) {
                    spent += units[steps[p].index].cost;
                }
                ++p;
            }
            if (p < steps.size()) {
                queue.push({steps[p].score, steps[p].index, g});
            }
        }
        while (!queue.empty()) {
            auto next = queue.top();
            queue.pop();
            take(next.index, options.maxTokens);
            const auto& steps = groups[next.group].steps;
            auto& p = positions[next.group];
            if (++p < steps.size()) {
                queue.push({steps[p].score, steps[p].index, next.group});
            }
        }
        result.selected.assign(chosen.begin(), chosen.end());
        size_t previous = 0;
        for (size_t n = 0; n < result.selected.size(); ++n) {
            size_t i = result.selected[n];
            if (n && i != previous + 1) {
                result.tokens.insert(result.tokens.end(), mGap.begin(), mGap.end());
            }
            result.tokens.insert(result.tokens.end(), source.begin() + result.spans[i].first,
                                 source.begin() + result.spans[i].second);
            previous = i;
        }
        return !result.selected.empty() && result.tokens.size() <= options.maxTokens;
    }

private:
    struct Unit {
        size_t cost = 0;
        std::vector<int> features;
    };
    struct Rank {
        double score;
        size_t index;
        size_t group;
        bool operator<(const Rank& other) const {
            return score != other.score ? score < other.score : index > other.index;
        }
    };
    struct Group {
        size_t cost = 0;
        std::vector<size_t> members;
        std::vector<Rank> steps;
    };
    std::unordered_map<int, std::vector<Tokens>> mDelimiters;
    Tokens mGap;

    std::vector<Span> split(const Tokens& source) const {
        std::vector<Span> spans;
        size_t start = 0;
        for (size_t i = 0; i < source.size();) {
            size_t length = 0;
            auto found = mDelimiters.find(source[i]);
            if (found != mDelimiters.end()) {
                for (const auto& ids : found->second) {
                    if (ids.size() > length && ids.size() <= source.size() - i &&
                        std::equal(ids.begin(), ids.end(), source.begin() + i)) {
                        length = ids.size();
                    }
                }
            }
            if (length) {
                i += length;
                spans.push_back({start, i});
                start = i;
            } else {
                ++i;
            }
        }
        if (start < source.size()) {
            spans.push_back({start, source.size()});
        }
        return spans;
    }

    static void prepare(Group& group, const std::vector<Unit>& units, const std::vector<bool>& bigrams) {
        std::unordered_map<int, size_t> df;
        for (auto i : group.members) {
            for (int f : units[i].features) {
                ++df[f];
            }
        }
        std::unordered_map<int, double> weights;
        for (const auto& entry : df) {
            weights[entry.first] =
                std::log(double(group.members.size() + 1) / (entry.second + 1)) * (bigrams[entry.first] ? .25 : 1);
        }
        std::unordered_set<int> covered;
        auto gain = [&](size_t i) {
            double value = 0;
            for (int f : units[i].features) {
                if (!covered.count(f)) {
                    value += weights.at(f);
                }
            }
            return value / units[i].cost;
        };
        std::priority_queue<Rank> queue;
        for (auto i : group.members) {
            double value = gain(i);
            if (value > 0) {
                queue.push({value, i, 0});
            }
        }
        size_t spent = 0;
        while (!queue.empty()) {
            auto next = queue.top();
            queue.pop();
            if (spent + units[next.index].cost > group.cost / 4) {
                continue;
            }
            double value = gain(next.index);
            if (value <= 0) {
                continue;
            }
            if (value != next.score) {
                queue.push({value, next.index, 0});
                continue;
            }
            group.steps.push_back(next);
            spent += units[next.index].cost;
            covered.insert(units[next.index].features.begin(), units[next.index].features.end());
        }
    }
};

} // namespace TokenCompression
} // namespace Transformer
} // namespace MNN
#endif // MNN_LLM_TOKEN_COMPRESS
#endif // MNN_LLM_TOKEN_COMPRESSOR_HPP
