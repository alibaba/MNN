#include "model_name_utils.hpp"
#include <regex>
#include "log_utils.hpp"
#include "model_sources.hpp"

#include <string>
#include <vector>

namespace mnn::downloader::ModelNameUtils {

namespace {
std::string ToLowerCopy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}
} // namespace

std::string GetDisplayModelName(const std::string& full_name, const mnn::downloader::Config& config) {
    const std::string provider_lower = ToLowerCopy(config.download_provider);
    std::vector<std::string> prefixes;

    if (provider_lower == "modelscope" || provider_lower == "ms") {
        prefixes = {
            "modelscope/MNN/",
            "ModelScope/MNN/",
            "modelscope/mnn/",
            "ModelScope/mnn/",
            "MNN/",
            "mnn/"
        };
    } else if (provider_lower == "modelers") {
        prefixes = {
            "modelers/MNN/",
            "Modelers/MNN/",
            "modelers/mnn/",
            "Modelers/mnn/",
            "MNN/",
            "mnn/"
        };
    } else if (provider_lower == "huggingface" || provider_lower == "hf") {
        prefixes = {
            "huggingface/taobao-mnn/",
            "HuggingFace/taobao-mnn/",
            "huggingface/Taobao-MNN/",
            "HuggingFace/Taobao-MNN/",
            "taobao-mnn/",
            "Taobao-MNN/"
        };
    }

    // Keep CLI display consistent with the web UI's extractModelSuffix helper by
    // also accounting for the currently configured default model path.
    if (!config.default_model.empty()) {
        std::string normalized_default = config.default_model;
        std::replace(normalized_default.begin(), normalized_default.end(), '\\', '/');
        auto last_slash = normalized_default.find_last_of('/');
        if (last_slash != std::string::npos) {
            std::string dynamic_prefix = normalized_default.substr(0, last_slash + 1);
            if (!dynamic_prefix.empty()) {
                prefixes.push_back(dynamic_prefix);
                prefixes.push_back(ToLowerCopy(dynamic_prefix));
            }
        }
    }

    for (const auto& prefix : prefixes) {
        if (!prefix.empty() && full_name.rfind(prefix, 0) == 0) {
            return full_name.substr(prefix.size());
        }
    }

    return full_name;
}

// Example input/output:
//   - model_name: "qwen-7b", config.download_provider: "modelscope"
//       => "ModelScope/MNN/qwen-7b"
//   - model_name: "huggingface/taobao-mnn/qwen-7b-chat", config.download_provider: "ms"
//       => "huggingface/taobao-mnn/qwen-7b-chat"   (unchanged, already fully qualified)
//   - model_name: "qwen-7b", config.download_provider: "huggingface"
//       => "huggingface/taobao-mnn/qwen-7b"
//   - model_name: "qwen-7b", provider: "" (fallback to ModelScope)
//       => "ModelScope/MNN/qwen-7b"
// Core implementation that takes provider directly
std::string GetFullModelId(const std::string& model_name, const std::string& provider) {
    if (model_name.empty()) {
        return "";
    }

    const std::string provider_lower = ToLowerCopy(provider);
    
    // Check if model_name already contains a provider prefix (e.g., "ModelScope/MNN/qwen-7b")
    std::vector<std::string> known_providers = {
        mnn::downloader::ModelSources::SOURCE_MODEL_SCOPE,
        mnn::downloader::ModelSources::SOURCE_MODELERS,
        mnn::downloader::ModelSources::SOURCE_HUGGING_FACE,
    };
    
    std::string model_lower = ToLowerCopy(model_name);
    for (const auto& prov : known_providers) {
        std::string prov_lower = ToLowerCopy(prov);
        if (model_lower.find(prov_lower + "/") == 0) {
            // Already has provider prefix, return as is
            return model_name;
        }
    }
    
    // Find the provider in known_providers and get the full provider name
    std::string full_provider_name;
    for (const auto& prov : known_providers) {
        std::string prov_lower = ToLowerCopy(prov);
        if (provider_lower == prov_lower) {
            full_provider_name = prov;
            break;
        }
    }
    
    // Handle abbreviations if no exact match
    if (full_provider_name.empty()) {
        if (provider_lower == "ms") {
            full_provider_name = ModelSources::SOURCE_MODEL_SCOPE;
        } else if (provider_lower == "hf") {
            full_provider_name = ModelSources::SOURCE_HUGGING_FACE;
        } else if (provider_lower == "ml") {
            full_provider_name = ModelSources::SOURCE_MODELERS;
        } else {
            // Default to ModelScope if not found
            full_provider_name = ModelSources::SOURCE_MODEL_SCOPE;
        }
    }
    
    // Check if model_name contains a slash (org_name/repo_name format)
    bool has_slash = (model_name.find('/') != std::string::npos);
    
    // Build repo_id: has_slash ? model_name : default_org_name + "/" + model_name
    std::string repo_id;
    if (has_slash) {
        repo_id = model_name;
    } else {
        std::string default_org_name = ModelSources::GetDefaultOrgName(provider);
        repo_id = default_org_name + "/" + model_name;
    }
    
    // full_model_id = provider + "/" + repo_id
    return full_provider_name + "/" + repo_id;
}

// Convenience overload that takes Config and delegates to the provider string version
std::string GetFullModelId(const std::string& model_name, const mnn::downloader::Config& config) {
    // Delegate to the provider string overload to reduce code redundancy
    return GetFullModelId(model_name, config.download_provider);
}

} // namespace mnn::downloader::ModelNameUtils
