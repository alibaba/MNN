//
//  local_model_utils.cpp
//
//  Utility functions for local model operations (listing, scanning)
//

#include "local_model_utils.hpp"
#include "cli_config_manager.hpp"
#include "log_utils.hpp"
#include "model_name_utils.hpp"
#include <algorithm>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

namespace mnncli {

// Check if a model directory is fully downloaded (has .mnncli/.complete marker)
bool LocalModelUtils::CheckIsDownloadedModel(const std::string& model_entry_path) {
  std::error_code ec;
  fs::path model_path(model_entry_path);
  std::string model_name = model_path.filename().string();

  // Skip hidden directories
  if (model_name.empty() || model_name[0] == '.') {
    return false;
  }

  // Check for .mnncli/.complete marker
  fs::path complete_marker = model_path / ".mnncli" / ".complete";
  if (fs::exists(complete_marker, ec)) {
    return true;
  }

  std::string owner_name = model_path.parent_path().filename().string();
  std::string identifier = owner_name.empty() ? model_name : owner_name + "/" + model_name;
  LOG_DEBUG_TAG("Skipping incomplete model: " + identifier, "LocalModelUtils");
  return false;
}

// Get local models from a specific provider's directory
// Only checks the provider directory for two-level directory structure:
//   <cache_dir>/<provider>/<owner>/<model_name>
// Example: /home/cache/ModelScope/Qwen/Qwen3_VXX-ssf -> "Qwen/Qwen3_VXX-ssf"
//
// Only returns models that have .mnncli/.complete marker (fully downloaded)
//
// Args:
//   provider: Provider name (e.g., "HuggingFace", "ModelScope", "Modelers")
//   cache_dir: Optional cache directory (for testing). If empty, uses ConfigManager
//
// Returns:
//   Vector of model names in "owner/model_name" format
std::vector<std::string> LocalModelUtils::ListLocalModelsInner(const std::string& provider,
                                                                const std::string& cache_dir) {
  std::vector<std::string> model_names;
  
  // Get cache directory - use provided one for testing, or ConfigManager for production
  std::string expanded_cache_dir;
  std::string local_cache_dir = cache_dir; 

  if (local_cache_dir.length() > 0) {
    if (local_cache_dir.back() == '/' || local_cache_dir.back() == '\\') {
      local_cache_dir.pop_back();
    }
    expanded_cache_dir = local_cache_dir;
  } else {
    auto& config_mgr = ConfigManager::GetInstance();
    expanded_cache_dir = config_mgr.GetBaseCacheDir();
  }

  // Build provider directory path
  std::string provider_dir = expanded_cache_dir + "/" + provider;

  // Debug output
  LOG_DEBUG_TAG("Scanning provider directory: " + provider_dir, "LocalModelUtils");

  // Check if provider directory exists
  std::error_code ec;
  if (!fs::exists(provider_dir, ec) || !fs::is_directory(provider_dir, ec)) {
    LOG_DEBUG_TAG("Provider directory does not exist: " + provider_dir, "LocalModelUtils");
    return model_names; // Return empty vector
  }

  // Scan for two-level directory structure: provider/owner/model_name
  for (const auto &owner_entry : fs::directory_iterator(provider_dir, ec)) {
    if (ec) {
      continue;
    }

    // Check if it's a directory (owner level)
    if (!fs::is_directory(owner_entry, ec)) {
      continue;
    }

    std::string owner_name = owner_entry.path().filename().string();
    // Skip hidden directories
    if (owner_name.empty() || owner_name[0] == '.') {
      continue;
    }

    // Scan model directories under owner
    std::string owner_path = owner_entry.path().string();
    for (const auto &model_entry : fs::directory_iterator(owner_path, ec)) {
      if (ec) {
        continue;
      }

      // Check if it's a directory (model level)
      if (fs::is_directory(model_entry, ec)) {
        std::string model_path = model_entry.path().string();
        if (CheckIsDownloadedModel(model_path)) {
          std::string model_name = model_entry.path().filename().string();
          model_names.emplace_back(provider + "/" + owner_name + "/" + model_name);
        }
      }
    }
  }

  std::sort(model_names.begin(), model_names.end());
  LOG_DEBUG_TAG("Found " + std::to_string(model_names.size()) + " complete models in " + provider_dir, "LocalModelUtils");
  return model_names;
}

int LocalModelUtils::ListLocalModels() {
  // Iterate through all known providers
  static const std::vector<std::string> kProviders = {
    "HuggingFace", "ModelScope", "Modelers"
  };
  
  std::vector<std::string> all_models;
  auto &config_mgr = ConfigManager::GetInstance();
  auto config = config_mgr.LoadConfig();

  mnn::downloader::Config downloader_config;
  downloader_config.cache_dir = config.cache_dir;
  downloader_config.download_provider = config.download_provider;

  for (const auto& provider : kProviders) {
    auto models = ListLocalModelsInner(provider, config_mgr.GetBaseCacheDir());
    all_models.insert(all_models.end(), models.begin(), models.end());
  }
  if (!all_models.empty()) {
    std::cout << "Local models:\n";
    for (const auto &name : all_models) {
      std::cout << mnn::downloader::ModelNameUtils::GetDisplayModelName(name, downloader_config)
                << "\n";
    }
  } else {
    std::cout << "No local models found.\n";
    std::cout << "Use 'mnncli search <keyword>' to search remote models\n";
    std::cout << "Use 'mnncli download <name>' to download models\n";
  }
  return 0;
}

} // namespace mnncli
