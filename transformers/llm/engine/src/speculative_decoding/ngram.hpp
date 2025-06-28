//
//  n-gram.hpp
//
//  Created by MNN on 2025/04/09.
//

#ifndef MNN_NGRAM_HPP
#define MNN_NGRAM_HPP

#include "llm/llm.hpp"
#include "lookahead.hpp"
// Ngram key match max length
#define MNN_NGRAM_KEY_MAX 8
/**
  the least key times occur in prompts
  can adjust acccording tasks
 */
const int match_ngram_size_balance[MNN_NGRAM_KEY_MAX] = {2, 2, 1, 1, 1, 1, 1, 1};
const int match_ngram_size_strict[MNN_NGRAM_KEY_MAX] = {4, 3, 2, 2, 1, 1, 1, 1};

/**
  candidate token concentration degree
  can adjust acccording tasks
 */
const int min_draft_percent_balance[MNN_NGRAM_KEY_MAX] = {60, 50, 50, 40, 40, 40, 40, 40};
const int min_draft_percent_strict[MNN_NGRAM_KEY_MAX] = {90, 80, 70, 60, 50, 50, 50, 50};

// #define ADOPT_STRICT_DRAFT_CONCENTRATION
namespace MNN {
namespace Transformer {

struct ngram_key {
    int keys[MNN_NGRAM_KEY_MAX];
    ngram_key() {
        for(int i = 0; i < MNN_NGRAM_KEY_MAX; i++) {
            keys[i] = -1;
        }
    }
    ngram_key(int *ptr, const int ngram_key_size) {
        for(int i = 0; i < ngram_key_size; i++) {
            keys[i] = ptr[i];
        }
        for(int i = ngram_key_size; i < MNN_NGRAM_KEY_MAX; i++) {
            keys[i] = -1;
        }
    }
    bool operator==(const ngram_key & b) const {
        for (int i = 0; i < MNN_NGRAM_KEY_MAX; ++i) {
            if (keys[i] != b.keys[i]) {
                return false;
            }
        }
        return true;
    }
};

struct token_hash_function {
    size_t operator()(const int token) const {
        return token * 11400714819323198485llu;
    }
};
    
struct ngram_hash_function {
    size_t operator()(const ngram_key & ngram) const {
        size_t hash = token_hash_function{}(ngram.keys[0]);
        for (int i = 1; i < MNN_NGRAM_KEY_MAX; ++i) {
            hash ^= token_hash_function{}(ngram.keys[i]);
        }
        return hash;
    }
};
// token & token_count, not consider order
typedef std::unordered_map<int, int> ngram_value;
// save token id in order
typedef std::vector<int> ngram_ordered_value;
//typedef std::unordered_map<ngram_key, ngram_value, ngram_hash_function> ngram_cache;
template<typename ValueType>
using ngram_cache = std::unordered_map<ngram_key, ValueType, ngram_hash_function>;

    
template<typename ValueType>
void ngram_add(ngram_cache<ValueType> &data, ngram_key ngram, int token) {
    // do nothing currently
}

// template specialize
template<>
void ngram_add<ngram_value>(ngram_cache<ngram_value> &data, ngram_key ngram, int token) {
    auto iter = data.find(ngram);
    if(iter != data.end()) {
        auto iter_value = iter->second.find(token);
        if(iter_value != iter->second.end()) {
            iter_value->second++;
        } else {
            iter->second.emplace(token, 1);
        }
    } else {
        ngram_value value;
        value.emplace(token, 1);
        data.emplace(ngram, value);
    }
}

// template specialize
template<>
void ngram_add<ngram_ordered_value>(ngram_cache<ngram_ordered_value> &data, ngram_key ngram, int token) {
    auto iter = data.find(ngram);
    if(iter != data.end()) {
        iter->second.emplace_back(token);
    } else {
        ngram_ordered_value value;
        value.emplace_back(token);
        data.emplace(ngram, value);
    }
}
    
template<typename ValueType>
void ngram_cache_update(ngram_cache<ValueType> &data, int ngram_key_min, int ngram_key_max, std::vector<int> tokens, int new_count) {
    const int token_num = tokens.size();
    for(int ngram_key_size = ngram_key_min; ngram_key_size <= ngram_key_max; ngram_key_size++) {
        int pos_start = std::max(token_num - new_count, ngram_key_size);
        for(int i = pos_start; i < token_num; i++) {
            int token = tokens[i];
            int ngram_start_idx = i - ngram_key_size;
            ngram_key ngram(tokens.data() + ngram_start_idx, ngram_key_size);
            if(std::is_same<ValueType, ngram_value>::value) {
                ngram_add(*(ngram_cache<ngram_value> *)(&data), ngram, token);
            } else {
                // save in order
                ngram_add(*(ngram_cache<ValueType> *)(&data), ngram, token);
            }
        }
    }
}

template<typename ValueType>
    void ngram_cache_search(ngram_cache<ValueType> &data, int ngram_key_min, int ngram_key_max, std::vector<int> tokens, std::vector<int> &drafts, int max_draft_size, MatchStrictLevel adopt_strictness) {
    MNN_ASSERT(drafts.size() == 1);

    while(drafts.size() < max_draft_size) {
        // if get first draft, then set adopt_strictness = 0.
        // reason is :  only support seq_len = 1 and MAX_DRAFT_SIZE two kinds of decoding currenctly
        if(drafts.size() > 1) {
            adopt_strictness = MatchStrictLevel::LOW_LEVEL;
        }
        int token_draft = -1;
        int max_score = 0;
        for(int ngram_key_size = ngram_key_max; ngram_key_size >= ngram_key_min; ngram_key_size--) {
            int ngram_start_idx = tokens.size() - ngram_key_size + drafts.size();
            ngram_key ngram;
            int i = ngram_start_idx;
            if(i < 0) {
                i = 0;
            }
            for(; i < tokens.size(); i++) {
                ngram.keys[i - ngram_start_idx] = tokens[i];
            }
            for(; i < ngram_start_idx + ngram_key_size; i++) {
                ngram.keys[i - ngram_start_idx] = drafts[i - tokens.size()];
            }
            
            auto iter = data.find(ngram);
            if(iter == data.end()) {
                continue;
            }
            
            
            if(std::is_same<ValueType, ngram_ordered_value>::value) {
                ngram_ordered_value real_value = *((ngram_ordered_value *)&(iter->second));
                token_draft = real_value.at(0);
                real_value.erase(real_value.begin());
                break;
            } else {
                
                int total_count = 0;
                int max_count = 0;
                
                ngram_value real_value = *((ngram_value *)&(iter->second));
                for(auto &iter_value : real_value) {
                    int token = iter_value.first;
                    int count = iter_value.second;
                    /** adopt_strictness:
                     0 -> adopt draft as long as ngram matched
                     1 -> adopt draft not only ngram matched, but also check ngram match size and concentration degree
                     2 -> adopt draft not only ngram matched, but also check ngram match size and concentration degree strictly
                     not adopted currenctly
                     */
                    #ifdef ADOPT_STRICT_DRAFT_CONCENTRATION
                    bool abandon = false;
                    switch (adopt_strictness) {
                        case MatchStrictLevel::MEDIUM_LEVEL:
                            if(count < match_ngram_size_balance[ngram_key_size - 1]) {
                                abandon = true;
                            }
                            break;
                        case MatchStrictLevel::HIGH_LEVEL:
                            if(count < match_ngram_size_strict[ngram_key_size - 1]) {
                                abandon = true;
                            }
                            break;
                        default:
                            break;
                    }
                    if(abandon) {
                        continue;
                    }
                    #endif
                    total_count += count;
                    // occur times * matched length -> the score, higher is better
                    int score = count * ngram_key_size;
                    if(score > max_score) {
                        token_draft = token;
                        max_score = score;
                        max_count = count;
                    }
                }
                
                if(max_count != 0 && adopt_strictness != MatchStrictLevel::LOW_LEVEL) {
                    float ratio = 100.0 * max_count / total_count;
                    switch (adopt_strictness) {
                        case MatchStrictLevel::MEDIUM_LEVEL:
                            if(ratio < min_draft_percent_balance[ngram_key_size - 1]) {
                                token_draft = -1;
                            }
                            break;
                        case MatchStrictLevel::HIGH_LEVEL:
                            if(ratio < min_draft_percent_strict[ngram_key_size - 1]) {
                                token_draft = -1;
                            }
                            break;
                        default:
                            break;
                    }
                }
            }
        }

        if(token_draft == -1) {
            break;
        }
        drafts.push_back(token_draft);
    }

    // if not one, pad to max_draft_size
    if(drafts.size() > 1) {
        for(int i = drafts.size(); i < max_draft_size; i++) {
            drafts.push_back(0);
        }
    }

}
    
} // Transformer
} // MNN
#endif
