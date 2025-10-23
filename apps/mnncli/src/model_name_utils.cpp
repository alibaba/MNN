#include "model_name_utils.hpp"

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

namespace mnncli::ModelNameUtils {

namespace {
std::string ToLowerCopy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}
} // namespace

std::string GetDisplayModelName(const std::string& full_name, const ConfigManager::Config& config) {
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

} // namespace mnncli::ModelNameUtils
