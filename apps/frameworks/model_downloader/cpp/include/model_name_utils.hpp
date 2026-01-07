#pragma once

#include "dl_config.hpp"
#include <string>

namespace mnn::downloader::ModelNameUtils {

// Normalize local model identifiers for display in CLI output.
std::string GetDisplayModelName(const std::string& full_name, const mnn::downloader::Config& config);

// Get full model ID in the format provider/org_name/repo_name, e.g., ModelScope/MNN/qwen-7b
std::string GetFullModelId(const std::string& model_name, const mnn::downloader::Config& config);

// Overloaded version that takes provider directly instead of config
std::string GetFullModelId(const std::string& model_name, const std::string& provider);

} // namespace mnn::downloader::ModelNameUtils

