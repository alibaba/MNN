//
//  model_manager.cpp
//
//  Model lifecycle management implementation
//

#include "model_manager.hpp"
#include "cli_config_manager.hpp"
#include "cli_download_listener.hpp"
#include "file_utils.hpp"
#include "log_utils.hpp"
#include "model_download_manager.hpp"
#include "model_name_utils.hpp"
#include "model_repository.hpp"
#include "user_interface.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <thread>

namespace fs = std::filesystem;

namespace mnncli {

using namespace mnn::downloader;

int ModelManager::SearchRemoteModels(const std::string &keyword, bool verbose,
                                     const std::string &cache_dir_override) {
  try {
    // Get cache directory from override, config, or use default
    auto &config_mgr = ConfigManager::GetInstance();
    auto config = config_mgr.LoadConfig();
    std::string cache_dir;
    if (!cache_dir_override.empty()) {
      cache_dir = cache_dir_override;
    } else {
      cache_dir = config.cache_dir;
      if (cache_dir.empty()) {
        cache_dir = config_mgr.GetBaseCacheDir();
      }
    }

    // Get current download provider from config
    std::string current_provider = config.download_provider;
    if (current_provider.empty()) {
      current_provider = mnn::downloader::ModelSources::SOURCE_HUGGING_FACE; // Default provider
    }

    // Create ModelRepository instance
    auto &model_repo = mnncli::ModelRepository::GetInstance(cache_dir);
    model_repo.SetDownloadProvider(current_provider);

    // Show search information
    std::cout << "🔍 Searching for LLM models with keyword: '" << keyword
              << "'\n";
    std::cout << "   Current provider: " << current_provider << "\n";
    std::cout << "   Cache directory: " << cache_dir << "\n\n";

    LOG_DEBUG_TAG("Starting model search", "ModelSearch");
    LOG_DEBUG_TAG("Keyword: " + keyword, "ModelSearch");
    LOG_DEBUG_TAG("Provider: " + current_provider, "ModelSearch");
    LOG_DEBUG_TAG("Cache directory: " + cache_dir, "ModelSearch");

    // Perform search
    auto searchResults = model_repo.SearchModels(keyword);

    if (searchResults.empty()) {
      if (keyword.empty()) {
        std::cout << "No LLM models found for provider: " << current_provider
                  << "\n";
      } else {
        std::cout << "No LLM models found matching keyword: '" << keyword
                  << "' for provider: " << current_provider << "\n";
      }
      std::cout << "\n💡 Try:\n";
      std::cout << "  • Use a different keyword\n";
      std::cout << "  • Change download provider: export "
                   "MNN_DOWNLOAD_PROVIDER=<provider>\n";
      std::cout
          << "  • Available providers: " << mnn::downloader::ModelSources::SOURCE_HUGGING_FACE << ", " << mnn::downloader::ModelSources::SOURCE_MODEL_SCOPE << ", " << mnn::downloader::ModelSources::SOURCE_MODELERS << "\n";
    } else {
      std::cout << "Found " << searchResults.size()
                << " matching LLM model(s):\n\n";

      // Display results in a table format
      std::cout << std::left << std::setw(50) << "Model Name" << std::setw(15)
                << "Vendor" << std::setw(12) << "Size" << std::setw(15)
                << "Tags" << "\n";
      std::cout << std::string(92, '-') << "\n";

      for (const auto &model : searchResults) {
        // Format tags for display with TAB prefix
        std::string tags_str;
        if (!model.tags.empty()) {
          tags_str = "\t" + model.tags[0]; // Show first tag with TAB prefix
          if (model.tags.size() > 1) {
            tags_str += " (+" + std::to_string(model.tags.size() - 1) + ")";
          }
        }

        // Format file size using existing LogUtils::FormatFileSize function
        std::string size_str;
        if (model.file_size > 0) {
          size_str = mnn::downloader::LogUtils::FormatFileSize(model.file_size);
        } else if (model.size_gb > 0) {
          // Fallback to size_gb if file_size is not available
          size_str = mnn::downloader::LogUtils::FormatFileSize(
              static_cast<int64_t>(model.size_gb * 1024 * 1024 * 1024));
        } else {
          size_str = "N/A";
        }

        std::cout << std::left << std::setw(50) << model.modelName
                  << std::setw(15) << model.vendor << std::setw(12) << size_str
                  << std::setw(15) << tags_str << "\n";
      }

      std::cout << "\n💡 To download a model, use:\n";
      std::cout << "  mnncli download <model_name>\n";
      std::cout << "  Example: mnncli download " << searchResults[0].modelName
                << "\n";
    }

  } catch (const std::exception &e) {
    mnncli::UserInterface::ShowError("Failed to search models: " +
                                     std::string(e.what()));
    return 1;
  }
  return 0;
}

int ModelManager::DownloadModel(const std::string &model_name, bool verbose,
                                const std::string &cache_dir_override) {
  if (model_name.empty()) {
    mnncli::UserInterface::ShowError("Model name is required",
                                     "Usage: mnncli model download <name>");
    return 1;
  }

  LOG_INFO("Downloading model: " + model_name);

  // Get current configuration
  auto &config_mgr = ConfigManager::GetInstance();
  auto config = config_mgr.LoadConfig();

  // Show which download provider will be used
  LOG_INFO("Using download provider: " + config.download_provider);

  // Early validation for obviously invalid model names
  if (!IsValidModelName(model_name)) {
    mnncli::UserInterface::ShowError("Invalid model name format: '" +
                                     model_name + "'");
    std::cout << "\n💡 Valid model name formats:\n";
    std::cout
        << "  • Simple name (e.g., 'qwen-7b') - will search in repository\n";
    std::cout << "  • Full ID (e.g., 'Qwen/Qwen-7B-Chat') - direct download\n";
    std::cout << "  • Prefixed ID (e.g., 'hf:Qwen/Qwen-7B-Chat') - specify "
                 "provider\n";
    std::cout << "\n💡 Try:\n";
    std::cout << "  • Use 'mnncli search " << model_name
              << "' to find available models\n";
    std::cout << "  • Use 'mnncli list' to see downloaded models\n";
    return 1;
  }

  try {
    // Get cache directory from override, config, or use default
    std::string cache_dir;
    if (!cache_dir_override.empty()) {
      cache_dir = cache_dir_override;
    } else {
      cache_dir = config.cache_dir;
      if (cache_dir.empty()) {
        cache_dir = config_mgr.GetBaseCacheDir();
      }
    }

    // Create download manager instance
    auto &download_manager =
        mnncli::ModelDownloadManager::GetInstance(cache_dir);

    // Create CLI download listener for user feedback
    CLIDownloadListener cli_listener;
    download_manager.AddListener(&cli_listener);

    // Use ModelRepository to get the correct model ID for download
    std::string model_id;
    std::string source = config.download_provider;

    // Log download information
    LOG_DEBUG_TAG("Starting model download", "ModelManager");
    LOG_DEBUG_TAG("Model name: " + model_name, "ModelManager");
    LOG_DEBUG_TAG("Source: " + source, "ModelManager");
    LOG_DEBUG_TAG("Cache directory: " + cache_dir, "ModelManager");

    // If no source specified in config, try to detect from model name
    // If no source specified in config, try to detect from model name
    if (source.empty()) {
      if (model_name.find("hf:") == 0 || model_name.find("huggingface:") == 0) {
        source = mnn::downloader::ModelSources::SOURCE_HUGGING_FACE;
        model_id = model_name.substr(model_name.find(":") + 1);
      } else if (model_name.find("ms:") == 0 ||
                 model_name.find("modelscope:") == 0) {
        source = mnn::downloader::ModelSources::SOURCE_MODEL_SCOPE;
        model_id = model_name.substr(model_name.find(":") + 1);
      } else if (model_name.find("ml:") == 0 ||
                 model_name.find("modelers:") == 0) {
        source = mnn::downloader::ModelSources::SOURCE_MODELERS;
        model_id = model_name.substr(model_name.find(":") + 1);
      } else {
        // Try to use ModelRepository to find the model
        try {
          auto &model_repo = mnncli::ModelRepository::GetInstance(cache_dir);
          model_repo.SetDownloadProvider(
              mnn::downloader::ModelSources::SOURCE_HUGGING_FACE); // Default to HuggingFace

          auto model_id_opt =
              model_repo.GetModelIdForDownload(model_name, mnn::downloader::ModelSources::SOURCE_HUGGING_FACE);
          if (model_id_opt) {
            model_id = *model_id_opt;
            source = ModelSources::SOURCE_HUGGING_FACE;
            LOG_INFO("✓ Found model in repository: " + model_id);
          } else {
            // Fallback to default behavior
            source = ModelSources::SOURCE_HUGGING_FACE;
            model_id = model_name;
          }
        } catch (const std::exception &e) {
          LOG_DEBUG_TAG("Failed to use ModelRepository: " +
                            std::string(e.what()),
                        "ModelManager");
          // Fallback to default behavior
          source = ModelSources::SOURCE_HUGGING_FACE;
          model_id = model_name;
        }
      }
    } else {
      // Source is specified, try to use ModelRepository
      try {
        auto &model_repo = mnncli::ModelRepository::GetInstance(cache_dir);
        model_repo.SetDownloadProvider(source);

        auto model_id_opt =
            model_repo.GetModelIdForDownload(model_name, source);
        if (model_id_opt) {
          model_id = *model_id_opt;
          LOG_INFO("✓ Found model in repository: " + model_id);

          // Get model type for additional info
          std::string model_type = model_repo.GetModelType(model_id);
          LOG_INFO("  Model type: " + model_type);
        } else {
          // Fallback to direct model name
          model_id = model_name;
          LOG_WARNING("⚠ Model not found in repository, using direct name: " +
                      model_id);
        }
      } catch (const std::exception &e) {
        LOG_DEBUG_TAG("Failed to use ModelRepository: " + std::string(e.what()),
                      "ModelManager");
        // Fallback to direct model name
        model_id = model_name;
      }
    }

    // Show download information
    LOG_INFO("🌐 Downloading from " + source);
    LOG_INFO("   Target model: " + model_id);
    LOG_INFO("   Cache directory: " + cache_dir);

    // Start the download
    download_manager.StartDownload(model_id, source, model_name);

    // Wait for download to complete or fail
    LOG_DEBUG("[DEBUG] Waiting for download to complete, model_id: " +
              model_id);
    while (download_manager.IsDownloading(model_id)) {
      LOG_DEBUG("[DEBUG] Still downloading, model_id: " + model_id);
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    LOG_DEBUG("[DEBUG] Download loop exited, model_id: " + model_id);

    // Check final status
    auto download_info = download_manager.GetDownloadInfo(model_id);
    if (download_info.state == mnn::downloader::DownloadState::COMPLETED) {
      auto downloaded_file = download_manager.GetDownloadedFile(model_id);
      if (!downloaded_file.empty() &&
          std::filesystem::exists(downloaded_file)) {
        mnncli::UserInterface::ShowSuccess("Model downloaded successfully!");
        mnncli::UserInterface::ShowInfo("Model saved to: " +
                                        downloaded_file.string());
        return 0;
      }
    } else if (download_info.state == mnn::downloader::DownloadState::FAILED) {
      mnncli::UserInterface::ShowError(
          "Download failed. Check the error messages above.");
      return 1;
    }

    // Remove listener before returning
    download_manager.RemoveListener(&cli_listener);

  } catch (const std::exception &e) {
    mnncli::UserInterface::ShowError("Failed to download model: " +
                                     std::string(e.what()));
    return 1;
  }
  return 0;
}

int ModelManager::DeleteModel(const std::string &model_name) {
  if (model_name.empty()) {
    mnncli::UserInterface::ShowError("Model name is required",
                                     "Usage: mnncli delete <name>");
    return 1;
  }

  LOG_INFO("Deleting model: " + model_name);
  try {
    // Get configuration
    auto &config_mgr = ConfigManager::GetInstance();
    auto config = config_mgr.LoadConfig();
    std::string cache_dir = config.cache_dir;
    if (cache_dir.empty()) {
      cache_dir = mnncli::kCachePath;
    }

    LOG_DEBUG_TAG("ModelManager::DeleteModel: cache_dir = " + cache_dir, "ModelManager");
    
    // Get ModelDownloadManager instance
    auto &download_manager =
        mnn::downloader::ModelDownloadManager::GetInstance(cache_dir);

    // Get full model_id using ModelNameUtils
    mnn::downloader::Config downloader_config;
    downloader_config.cache_dir = config.cache_dir;
    downloader_config.download_provider = config.download_provider;
    std::string model_id = mnn::downloader::ModelNameUtils::GetFullModelId(model_name, downloader_config);
    LOG_DEBUG_TAG("Model ID: " + model_id, "ModelManager");
    LOG_DEBUG_TAG("Calling ModelDownloadManager::DeleteRepo with model_id = " + model_id, "ModelManager");
    
    // Use DeleteRepo to properly delete the model
    bool deleted = download_manager.DeleteRepo(model_id);
    
    LOG_DEBUG_TAG("ModelDownloadManager::DeleteRepo returned: " + std::string(deleted ? "true" : "false"), "ModelManager");
    
    if (deleted) {
      mnncli::UserInterface::ShowSuccess("Model deleted successfully: " +
                                         model_name);
    } else {
      mnncli::UserInterface::ShowError("Model not found: " + model_name);
      return 1;
    }
  } catch (const std::exception &e) {
    mnncli::UserInterface::ShowError("Failed to delete model: " +
                                     std::string(e.what()));
    return 1;
  }
  return 0;
}

int ModelManager::ShowModelInfo(const std::string &model_name, bool verbose) {
  if (model_name.empty()) {
    mnncli::UserInterface::ShowError("Model name is required",
                                     "Usage: mnncli model info <name>");
    return 1;
  }

  LOG_INFO("Showing model info: " + model_name);

  try {
    auto &config_mgr = ConfigManager::GetInstance();
    auto config = config_mgr.LoadConfig();
    mnn::downloader::Config downloader_config;
    downloader_config.cache_dir = config.cache_dir;
    downloader_config.download_provider = config.download_provider;
    std::string model_path = mnn::downloader::FileUtils::GetModelPath(model_name, downloader_config);

    if (model_path.empty()) {
      mnncli::UserInterface::ShowError("Model not found: " + model_name);
      std::cout << "\n💡 Try:\n";
      std::cout << "  • Use 'mnncli list' to see available models\n";
      std::cout << "  • Use 'mnncli search <keyword>' to find models\n";
      std::cout << "  • Use 'mnncli download <name>' to download models\n";
      return 1;
    }

    // Display model information
    std::cout << "📁 Model Information: " << model_name << "\n";
    std::cout << "====================\n";
    std::cout << "📍 Download Location: " << model_path << "\n";

    // Check if it's a symlink
    std::error_code ec;
    if (fs::is_symlink(model_path, ec)) {
      auto target = fs::read_symlink(model_path, ec);
      if (!ec) {
        std::cout << "🔗 Symlink Target: " << target.string() << "\n";
      }
    }

    // Get directory size
    try {
      int64_t total_size = 0;
      for (const auto &entry : fs::recursive_directory_iterator(model_path)) {
        if (entry.is_regular_file()) {
          total_size += entry.file_size();
        }
      }
      std::cout << "💾 Total Size: "
                << mnn::downloader::LogUtils::FormatFileSize(total_size) << "\n";
    } catch (const std::exception &e) {
      LOG_DEBUG_TAG("Failed to calculate directory size: " +
                        std::string(e.what()),
                    "ModelManager");
    }

    // Look for config.json file
    std::string config_path = model_path + "/config.json";
    if (fs::exists(config_path)) {
      std::cout << "\n📋 Configuration File: " << config_path << "\n";
      std::cout << "====================\n";

      try {
        std::ifstream config_file(config_path);
        if (config_file.is_open()) {
          std::string line;
          int line_count = 0;
          const int max_lines =
              verbose ? 1000 : 50; // Limit output unless verbose

          while (std::getline(config_file, line) && line_count < max_lines) {
            std::cout << line << "\n";
            line_count++;
          }

          if (line_count >= max_lines) {
            std::cout << "... (truncated, use -v for full output)\n";
          }

          config_file.close();
        } else {
          std::cout << "❌ Unable to read config file\n";
        }
      } catch (const std::exception &e) {
        std::cout << "❌ Error reading config file: " << e.what() << "\n";
      }
    } else {
      std::cout << "\n⚠️  No config.json found in model directory\n";

      // List available files
      std::cout << "\n📂 Available Files:\n";
      try {
        for (const auto &entry : fs::directory_iterator(model_path)) {
          if (entry.is_regular_file()) {
            std::cout << "  📄 " << entry.path().filename().string() << "\n";
          } else if (entry.is_directory()) {
            std::cout << "  📁 " << entry.path().filename().string() << "/\n";
          }
        }
      } catch (const std::exception &e) {
        std::cout << "❌ Error listing files: " << e.what() << "\n";
      }
    }

    // Show additional info if verbose
    if (verbose) {
      std::cout << "\n🔍 Additional Information:\n";
      std::cout << "====================\n";
      auto &config_mgr = ConfigManager::GetInstance();
      auto config = config_mgr.LoadConfig();
      std::string cache_dir = config.cache_dir;
      if (cache_dir.empty()) {
        cache_dir = config_mgr.GetBaseCacheDir();
      }
      std::cout << "Cache Directory: " << cache_dir << "\n";
      std::cout << "Download Provider: " << config.download_provider << "\n";

      // Check for other common model files
      std::vector<std::string> common_files = {
          "tokenizer.json",    "tokenizer_config.json",   "vocab.txt",
          "merges.txt",        "special_tokens_map.json", "model.mnn",
          "pytorch_model.bin", "model.safetensors",       "model.bin"};

      std::cout << "\n📋 Model Files Status:\n";
      for (const auto &file : common_files) {
        std::string file_path = model_path + "/" + file;
        if (fs::exists(file_path)) {
          try {
            auto file_size = fs::file_size(file_path);
            std::cout << "  ✅ " << file << " ("
                      << mnn::downloader::LogUtils::FormatFileSize(file_size) << ")\n";
          } catch (...) {
            std::cout << "  ✅ " << file << " (size unknown)\n";
          }
        } else {
          std::cout << "  ❌ " << file << "\n";
        }
      }
    }

  } catch (const std::exception &e) {
    mnncli::UserInterface::ShowError("Failed to show model info: " +
                                     std::string(e.what()));
    return 1;
  }

  return 0;
}

bool ModelManager::IsValidModelName(const std::string &model_name) {
  if (model_name.empty()) {
    return false;
  }

  // Check for invalid characters
  for (char c : model_name) {
    if (!std::isalnum(c) && c != '-' && c != '_' && c != '/' && c != ':' &&
        c != '.') {
      return false;
    }
  }

  // Check for minimum length (at least 2 characters)
  if (model_name.length() < 2) {
    return false;
  }

  // Check for obviously invalid patterns (too generic, likely a mistake)
  // Only reject very generic single words that are clearly mistakes
  if (model_name == "qwen" || model_name == "chatgpt" || model_name == "gpt" ||
      model_name == "llm" || model_name == "ai" || model_name == "model" ||
      model_name == "test" || model_name == "demo" || model_name == "example") {
    return false;
  }

  return true;
}

// Methods moved to LocalModelUtils class

} // namespace mnncli
