#pragma once

#include "mnncli_config.hpp"
#include <string>

namespace mnncli::ModelNameUtils {

// Normalize local model identifiers for display in CLI output.
std::string GetDisplayModelName(const std::string& full_name, const ConfigManager::Config& config);

} // namespace mnncli::ModelNameUtils

