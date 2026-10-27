#ifndef PROMPT_CACHE_UTILS_HPP
#define PROMPT_CACHE_UTILS_HPP

#include <fstream>
#include <string>
#include <vector>
#include "MNN_generated.h"
#include "core/MNNFileUtils.h"

namespace MNN {
namespace Transformer {

inline const Net* loadVerifiedNet(const std::string& modelPath, std::vector<uint8_t>& modelBuffer) {
    using flatbuffers::soffset_t;
    std::ifstream modelFile(modelPath, std::ios::binary | std::ios::ate);
    const auto modelSize = modelFile.tellg();
    if (modelSize <= 0 || modelSize >= FLATBUFFERS_MAX_BUFFER_SIZE) {
        return nullptr;
    }
    modelBuffer.resize(static_cast<size_t>(modelSize));
    modelFile.seekg(0);
    if (!modelFile.read(reinterpret_cast<char*>(modelBuffer.data()), modelBuffer.size())) {
        return nullptr;
    }
    flatbuffers::Verifier verifier(modelBuffer.data(), modelBuffer.size());
    if (!VerifyNetBuffer(verifier)) {
        return nullptr;
    }
    return GetNet(modelBuffer.data());
}

inline bool modelHasLinearAttention(const std::string& modelPath) {
    std::vector<uint8_t> modelBuffer;
    const auto net = loadVerifiedNet(modelPath, modelBuffer);
    if (net == nullptr || net->oplists() == nullptr) {
        return false;
    }
    for (const auto op : *net->oplists()) {
        if (op->type() == OpType_LinearAttention) {
            return true;
        }
    }
    return false;
}

inline bool validatePrefixCacheFiles(const std::string& modelPath, const std::string& prefixPath, int layerCount) {
    std::vector<bool> needsValue;
    {
        std::vector<uint8_t> modelBuffer;
        const auto net = loadVerifiedNet(modelPath, modelBuffer);
        if (net == nullptr || net->oplists() == nullptr ||
            (net->subgraphs() != nullptr && net->subgraphs()->size() != 0)) {
            return false;
        }
        // Static modules preserve serialized op order, which assigns prefix-cache layer indices.
        for (const auto op : *net->oplists()) {
            if (op->type() == OpType_Attention) {
                const auto param = op->main_as_AttentionParam();
                if (param == nullptr) {
                    return false;
                }
                // A KV-shared layer reads its owner's cache file and takes no index of its own.
                if (param->kv_cache() && param->kv_shared_layer_index() < 0) {
                    needsValue.push_back(true);
                }
            } else if (op->type() == OpType_LinearAttention) {
                const auto param = op->main_as_LinearAttentionParam();
                if (param == nullptr || param->attn_type() == nullptr) {
                    return false;
                }
                const auto type = param->attn_type()->str();
                if (type != "short_conv" && type != "gated_delta_rule") {
                    return false;
                }
                needsValue.push_back(type != "short_conv");
            }
        }
    }
    if (layerCount <= 0 || needsValue.size() != static_cast<size_t>(layerCount)) {
        MNN_PRINT("Prefix cache layer layout unavailable or mismatched: %s\n", modelPath.c_str());
        return false;
    }
    for (int layer = 0; layer < layerCount; ++layer) {
        const auto base = prefixPath + "_" + std::to_string(layer);
        const bool hasValue = MNNFileExist((base + ".v").c_str());
        const bool hasValueSync = MNNFileExist((base + "_sync.v").c_str());
        if (!MNNFileExist((base + "_sync.k").c_str()) ||
            hasValue != needsValue[layer] || hasValueSync != needsValue[layer]) {
            MNN_PRINT("Prefix cache required files missing or mismatched at layer %d\n", layer);
            return false;
        }
        for (const char* suffix : {".k", ".v"}) {
            if (suffix[1] == 'v' && !needsValue[layer]) {
                continue;
            }
            const auto path = base + suffix;
            auto file = MNNOpenFile(path.c_str(), MNN_FILE_READ);
            const size_t size = file == INVALID_FILE ? INVALID_SIZE : MNNGetFileSize(file);
            if (file != INVALID_FILE) {
                MNNCloseFile(file);
            }
            if (size == INVALID_SIZE || size == 0) {
                MNN_PRINT("Prefix cache data file missing or empty: %s\n", path.c_str());
                return false;
            }
        }
    }
    return true;
}

// Strip <think>...</think> blocks from cached prompt text.
// When enable_thinking is active, the template appends <think>...</think>
// to the last assistant message only. Since it won't be last on the next
// turn, stripping ensures the cached text matches the next turn's rendering.
inline void stripThinkBlocks(std::string& text) {
    while (true) {
        auto start = text.find("<think>");
        if (start == std::string::npos)
            break;
        auto end = text.find("</think>", start);
        if (end == std::string::npos)
            break;
        end += 8;
        while (end < text.size() && (text[end] == '\n' || text[end] == '\r'))
            end++;
        text.erase(start, end - start);
    }
}

}  // namespace Transformer
}  // namespace MNN

#endif  // PROMPT_CACHE_UTILS_HPP
