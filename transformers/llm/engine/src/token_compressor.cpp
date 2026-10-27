//
//  token_compressor.cpp
//
//  Created by MNN on 2026/10/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "token_compressor.hpp"
#ifdef MNN_LLM_TOKEN_COMPRESS
#include "tokenizer/tokenizer.hpp"
#include <MNN/MNNDefine.h>

namespace MNN {
namespace Transformer {
namespace TokenCompression {
std::vector<std::vector<int>> sentenceDelimiters(Tokenizer* tokenizer) {
    struct Mark {
        const char* text;
        bool requireSpace;
    };
    const Mark marks[] = {{".", true},  {"!", true},  {"?", true},   {"...", true},  {"。", true},
                          {"！", true}, {"？", true}, {"\n", false}, {"\r\n", false}};
    const char* quotes[] = {"", "\"", "'", "”", "’"};
    const char* spaces[] = {"", " ", "\t", "\n", "\n\n", "\r\n"};
    std::vector<std::vector<int>> patterns;
    for (const auto& mark : marks) {
        for (const char* quote : quotes) {
            for (const char* space : spaces) {
                if (mark.requireSpace && !*space) {
                    continue;
                }
                for (const char* lead : {"", " "}) {
                    std::string text = std::string(lead) + mark.text + quote + space;
                    auto ids = tokenizer->encode(text, false);
                    if (tokenizer->decode(ids) == text) {
                        patterns.push_back(std::move(ids));
                    }
                }
            }
        }
    }
    return patterns;
}

static bool compressPrompt(Tokenizer* tokenizer, std::vector<int>& ids, size_t max_tokens) {
    if (!max_tokens || ids.size() <= max_tokens) {
        return true;
    }
    // Keep chat control tokens and other messages intact; select only within the longest text segment.
    size_t begin = 0, end = 0, start = 0;
    for (size_t i = 0; i <= ids.size(); ++i) {
        if (i != ids.size() && !tokenizer->is_special(ids[i])) {
            continue;
        }
        if (i - start > end - begin) {
            begin = start;
            end = i;
        }
        start = i + 1;
    }
    size_t immutable = ids.size() - (end - begin);
    if (immutable >= max_tokens) {
        return false;
    }
    Compressor compressor(sentenceDelimiters(tokenizer), tokenizer->encode("\n[…]\n", false));
    Compressor::Options options;
    options.maxTokens = max_tokens - immutable;
    Compressor::Result result;
    if (!compressor.compress(std::vector<int>(ids.begin() + begin, ids.begin() + end), options, result)) {
        return false;
    }
    std::vector<int> output;
    output.reserve(immutable + result.tokens.size());
    output.insert(output.end(), ids.begin(), ids.begin() + begin);
    output.insert(output.end(), result.tokens.begin(), result.tokens.end());
    output.insert(output.end(), ids.begin() + end, ids.end());
    ids.swap(output);
    return true;
}

std::vector<int> encodePrompt(Tokenizer* tokenizer, const std::string& text, int maxTokens) {
    auto ids = tokenizer->encode(text);
    if (maxTokens < 0) {
        MNN_ERROR("token_compress must be a non-negative integer token limit\n");
        return {};
    }
    if (!compressPrompt(tokenizer, ids, static_cast<size_t>(maxTokens))) {
        MNN_ERROR("token_compress cannot fit immutable tokens and boundary sentences\n");
        return {};
    }
    return ids;
}
} // namespace TokenCompression
} // namespace Transformer
} // namespace MNN
#endif
