#pragma once

#include "mnncli_config.hpp"
#include <string>

namespace mnncli::ModelNameUtils {

// Normalize local model identifiers for display in CLI output.
std::string GetDisplayModelName(const std::string& full_name, const mnncli::Config& config);

// Get full model ID in the format provider/org_name/repo_name, e.g., ModelScope/MNN/qwen-7b
std::string GetFullModelId(const std::string& model_name, const mnncli::Config& config);

// Overloaded version that takes provider directly instead of config
std::string GetFullModelId(const std::string& model_name, const std::string& provider);

} // namespace mnncli::ModelNameUtils

