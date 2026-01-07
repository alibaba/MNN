//
// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#include "model_sources.hpp"
#include <algorithm>

namespace mnn::downloader {

ModelSource ModelSources::FromString(const std::string& source_str) {
    std::string lower = source_str;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    
    if (lower == "huggingface" || lower == "hf") return ModelSource::HUGGING_FACE;
    if (lower == "modelscope" || lower == "ms") return ModelSource::MODEL_SCOPE;
    if (lower == "modelers" || lower == "ml") return ModelSource::MODELERS;
    
    return ModelSource::UNKNOWN;
}

std::string ModelSources::GetDefaultOrgName(const std::string& provider) {
    std::string lower = provider;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    
    if (lower == "huggingface" || lower == "hf") return "taobao-mnn";
    if (lower == "modelscope" || lower == "ms") return "MNN";
    if (lower == "modelers" || lower == "ml") return "MNN";
    
    return "MNN";  // Default to MNN for unknown providers
}

std::string ModelSources::ToString(ModelSource source) {
    switch (source) {
        case ModelSource::HUGGING_FACE: return SOURCE_HUGGING_FACE;
        case ModelSource::MODEL_SCOPE: return SOURCE_MODEL_SCOPE;
        case ModelSource::MODELERS: return SOURCE_MODELERS;
        default: return "Unknown";
    }
}

ModelSource ModelSources::GetSource(const std::string& model_id) {
    auto splits = SplitSource(model_id);
    return FromString(splits.first);
}

std::pair<std::string, std::string> ModelSources::SplitSource(const std::string& model_id) {
    size_t colon_pos = model_id.find('/');
    if (colon_pos == std::string::npos) {
        return {"", model_id};
    }
    
    std::string source = model_id.substr(0, colon_pos);
    std::string path = model_id.substr(colon_pos + 1);
    return {source, path};
}

std::string ModelSources::GetModelName(const std::string& model_id) {
    auto splits = SplitSource(model_id);
    if (splits.second.empty()) {
        return model_id;
    }
    
    // Extract the last part of the path as model name
    size_t last_slash = splits.second.find_last_of('/');
    if (last_slash != std::string::npos) {
        return splits.second.substr(last_slash + 1);
    }
    
    return splits.second;
}

} // namespace mnn::downloader
