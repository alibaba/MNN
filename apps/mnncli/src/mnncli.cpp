//
//  mnncli.cpp
//
//  Created by MNN on 2023/03/24.
//  Jinde.Song
//  LLM command line tool, based on llm_demo.cpp
//
#include "../../../transformers/llm/engine/include/llm/llm.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include <fstream>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <functional>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <filesystem>
#include <thread>
#include <chrono>
#if defined(_WIN32)
#include <windows.h>
#include <shellapi.h>
#endif
#include "file_utils.hpp"
#include "model_download_manager.hpp"
#include "model_repository.hpp"
#include "model_file_downloader.hpp"
#include "hf_api_client.hpp"
#include "ms_api_client.hpp"
#include "ms_model_downloader.hpp"
#include "ml_model_downloader.hpp"
#include "ml_api_client.hpp"
#include "llm_benchmark.hpp"
#include "mnncli_config.hpp"
#include "mnncli_server.hpp"
#include "nlohmann/json.hpp"
#include "log_utils.hpp"
#include "model_runner.hpp"
#include "user_interface.hpp"

using namespace MNN::Transformer;
namespace fs = std::filesystem;

// Forward declarations
class CommandLineInterface;
class LLMManager;
class ModelManager;

// CLI-specific download listener for user interface feedback
class CLIDownloadListener : public mnncli::DownloadListener {
public:
    CLIDownloadListener() = default;
    ~CLIDownloadListener() override = default;
    
    void OnDownloadStart(const std::string& model_id) override {
        // Only show start message if we're actually starting a valid download
        if (!model_id.empty() && model_id.find('/') != std::string::npos) {
            mnncli::UserInterface::ShowInfo("Starting download: " + model_id);
        }
    }
    
    void OnDownloadProgress(const std::string& model_id, const mnncli::DownloadProgress& progress) override {
        // Only show progress if we have a valid model ID and meaningful progress
        if (model_id.empty() || progress.progress < 0) {
            return;
        }
        
        // Skip progress display if total_size is 0 (likely a state change notification)
        if (progress.total_size == 0) {
            return;
        }
        
        // Use ModelScope style: show filename with progress bar
        std::string file_name = progress.current_file.empty() ? "file" : progress.current_file;
        
        // Use parent class utility methods for formatting
        file_name = mnncli::ModelRepoDownloader::ExtractFileName(file_name);
        std::string size_info = mnncli::ModelRepoDownloader::FormatFileSizeInfo(
            progress.saved_size, progress.total_size);
        
        std::string message = file_name + size_info;
        mnncli::UserInterface::ShowProgress(message, progress.progress);
    }
    
    void OnDownloadFinished(const std::string& model_id, const std::string& path) override {
        mnncli::UserInterface::ShowSuccess("Download completed: " + model_id);
        mnncli::UserInterface::ShowInfo("Model saved to: " + path);
    }
    
    void OnDownloadFailed(const std::string& model_id, const std::string& error) override {
        mnncli::UserInterface::ShowError("Download failed: " + model_id + " - " + error);
    }
    
    void OnDownloadPaused(const std::string& model_id) override {
        mnncli::UserInterface::ShowInfo("Download paused: " + model_id);
    }
    
    void OnDownloadTaskAdded() override {
        mnncli::UserInterface::ShowInfo("Download task added");
    }
    
    void OnDownloadTaskRemoved() override {
        mnncli::UserInterface::ShowInfo("Download task removed");
    }
    
    void OnRepoInfo(const std::string& model_id, int64_t last_modified, int64_t repo_size) override {
        if (repo_size > 0) {
            mnncli::UserInterface::ShowInfo("Repository info for " + model_id + ": " + mnncli::LogUtils::FormatFileSize(repo_size));
        }
    }
    
    std::string GetClassTypeName() const override {
        return "CLIDownloadListener";
    }
    

};

// Configuration management
namespace ConfigManager {
    static std::string GetConfigFilePath() {
        std::string config_dir = mnncli::FileUtils::GetBaseCacheDir();
        return (fs::path(config_dir) / "mnncli_config.json").string();
    }
    
    bool SaveConfig(const Config& config) {
        try {
            std::string config_path = GetConfigFilePath();
            std::string config_dir = fs::path(config_path).parent_path().string();
            
            // Create config directory if it doesn't exist
            if (!fs::exists(config_dir)) {
                fs::create_directories(config_dir);
            }
            
            nlohmann::json j;
            j["default_model"] = config.default_model;
            j["cache_dir"] = config.cache_dir;
            j["log_level"] = config.log_level;
            j["default_max_tokens"] = config.default_max_tokens;
            j["default_temperature"] = config.default_temperature;
            j["api_host"] = config.api_host;
            j["api_port"] = config.api_port;
            j["download_provider"] = config.download_provider;
            
            std::ofstream file(config_path);
            if (file.is_open()) {
                file << j.dump(2);
                file.close();
                return true;
            }
        } catch (...) {
            // Silently fail if we can't save config
        }
        return false;
    }
    
    Config LoadDefaultConfig() {
        Config config;
        
        // Try to load from config file first
        std::string config_path = GetConfigFilePath();
        if (fs::exists(config_path)) {
            try {
                std::ifstream file(config_path);
                if (file.is_open()) {
                    nlohmann::json j;
                    file >> j;
                    
                    config.default_model = j.value("default_model", "");
                    config.cache_dir = j.value("cache_dir", "~/.cache/mnncli/mnnmodels");
                    config.log_level = j.value("log_level", "info");
                    config.default_max_tokens = j.value("default_max_tokens", 1000);
                    config.default_temperature = j.value("default_temperature", 0.7f);
                    config.api_host = j.value("api_host", "127.0.0.1");
                    config.api_port = j.value("api_port", 8000);
                    config.download_provider = j.value("download_provider", "huggingface");
                    
                    file.close();
                    
                    // Check environment variables (they take precedence over file)
                    if (const char* env_model = std::getenv("MNN_DEFAULT_MODEL")) {
                        config.default_model = env_model;
                    }
                    if (const char* env_provider = std::getenv("MNN_DOWNLOAD_PROVIDER")) {
                        config.download_provider = env_provider;
                    }
                    if (const char* env_cache = std::getenv("MNN_CACHE_DIR")) {
                        config.cache_dir = env_cache;
                    }
                    if (const char* env_host = std::getenv("MNN_API_HOST")) {
                        config.api_host = env_host;
                    }
                    if (const char* env_port = std::getenv("MNN_API_PORT")) {
                        try {
                            config.api_port = std::stoi(env_port);
                        } catch (...) {
                            // Keep file value if invalid
                        }
                    }
                    
                    return config;
                }
            } catch (...) {
                // If file loading fails, fall back to defaults
            }
        }
        
        // Fall back to defaults and environment variables
        std::string default_model = ""; // default
        if (const char* env_model = std::getenv("MNN_DEFAULT_MODEL")) {
            default_model = env_model;
        }

        std::string download_provider = "huggingface"; // default
        if (const char* env_provider = std::getenv("MNN_DOWNLOAD_PROVIDER")) {
            download_provider = env_provider;
        }

        std::string cache_dir = "~/.cache/mnncli/mnnmodels";
        if (const char* env_cache = std::getenv("MNN_CACHE_DIR")) {
            cache_dir = env_cache;
        }

        std::string api_host = "127.0.0.1";
        if (const char* env_host = std::getenv("MNN_API_HOST")) {
            api_host = env_host;
        }

        int api_port = 8000;
        if (const char* env_port = std::getenv("MNN_API_PORT")) {
            try {
                api_port = std::stoi(env_port);
            } catch (...) {
                // Keep default if invalid
            }
        }

        return {
            .default_model = default_model,
            .cache_dir = cache_dir,
            .log_level = "info",
            .default_max_tokens = 1000,
            .default_temperature = 0.7f,
            .api_host = api_host,
            .api_port = api_port,
            .download_provider = download_provider
        };
    }
    
    void ShowConfig(const Config& config) {
        std::cout << "Configuration:\n";
        std::cout << "  Default Model(default_model): " << (config.default_model.empty() ? "Not set" : config.default_model) << "\n";
        std::cout << "  Cache Directory(cache_dir): " << config.cache_dir << "\n";
        std::cout << "  Log Level(log_level): " << config.log_level << "\n";
        std::cout << "  Default Max Tokens(default_max_tokens): " << config.default_max_tokens << "\n";
        std::cout << "  Default Temperature(default_temperature): " << config.default_temperature << "\n";
        std::cout << "  API Host(api_host): " << config.api_host << "\n";
        std::cout << "  API Port(api_port): " << config.api_port << "\n";
        std::cout << "  Download Provider(download_provider): " << config.download_provider << "\n";
    }
    
    bool SetConfigValue(Config& config, const std::string& key, const std::string& value) {
        if (key == "default_model") {
            config.default_model = value;
            return true;
        } else if (key == "download_provider") {
            std::string lower_value = value;
            std::transform(lower_value.begin(), lower_value.end(), lower_value.begin(), ::tolower);
            if (lower_value == "huggingface" || lower_value == "hf" ||
                lower_value == "modelscope" || lower_value == "ms" ||
                lower_value == "modelers") {
                config.download_provider = lower_value;
                return true;
            } else {
                return false;
            }
        } else if (key == "cache_dir") {
            config.cache_dir = value;
            return true;
        } else if (key == "log_level") {
            config.log_level = value;
            return true;
        } else if (key == "api_host") {
            config.api_host = value;
            return true;
        } else if (key == "default_max_tokens") {
            try {
                config.default_max_tokens = std::stoi(value);
                return true;
            } catch (...) {
                return false;
            }
        } else if (key == "default_temperature") {
            try {
                config.default_temperature = std::stof(value);
                return true;
            } catch (...) {
                return false;
            }
        } else if (key == "api_port") {
            try {
                config.api_port = std::stoi(value);
                return true;
            } catch (...) {
                return false;
            }
        }
        return false;
    }
    
    std::string GetConfigHelp() {
        return R"(
Available configuration keys:
  default_model      - Set default model name
  download_provider  - Set default download provider (huggingface, modelscope, modelers)
  cache_dir         - Set cache directory path
  log_level         - Set log level (debug, info, warn, error)
  api_host          - Set API server host
  api_port          - Set API server port
  default_max_tokens - Set default maximum tokens for generation
  default_temperature - Set default temperature for generation

Environment Variables (take precedence over config):
  MNN_DOWNLOAD_PROVIDER - Set default download provider
  MNN_CACHE_DIR        - Set cache directory path
  MNN_API_HOST         - Set API server host
  MNN_API_PORT         - Set API server port

Examples:
  mnncli config set default_model Qwen3-VL-4B-Instruct-MNN
  mnncli config set download_provider modelscope
  mnncli config set cache_dir ~/.cache/mnncli/mnnmodels
  mnncli config set api_port 8080

  # Using environment variables
  export MNN_DEFAULT_MODEL=Qwen3-VL-4B-Instruct-MNN
  export MNN_DOWNLOAD_PROVIDER=modelscope
  export MNN_CACHE_DIR=~/.cache/mnncli/mnnmodels
  mnncli config show
)";
    }
};

// LLM management
class LLMManager {
public:
    static std::unique_ptr<Llm> CreateLLM(const std::string& config_path, bool use_template) {
        std::unique_ptr<Llm> llm(Llm::createLLM(config_path));
        if (use_template) {
            llm->set_config("{\"tmp_path\":\"tmp\"}");
        } else {
            llm->set_config("{\"tmp_path\":\"tmp\",\"use_template\":false}");
        }
        {
            AUTOTIME;
            llm->load();
        }
        if (true) {
            AUTOTIME;
            TuningPrepare(llm.get());
        }
        return llm;
    }
    
private:
    static void TuningPrepare(Llm* llm) {
        llm->tuning(OP_ENCODER_NUMBER, {1, 5, 10, 20, 30, 50, 100});
    }
};

// Model management
class ModelManager {
public:
    // Helper function to get all local models (shared by list and info commands)
    static int GetLocalModelNames(std::vector<std::string>& model_names, bool verbose = false) {
        // Use configured cache directory instead of GetBaseCacheDir()
        auto config = ConfigManager::LoadDefaultConfig();
        std::string base_cache_dir = config.cache_dir;
        if (base_cache_dir.empty()) {
            base_cache_dir = mnncli::kCachePath;
        }
        
        // Expand tilde in path
        std::string expanded_cache_dir = base_cache_dir;
        if (base_cache_dir[0] == '~') {
            const char* home_dir = getenv("HOME");
            if (home_dir) {
                expanded_cache_dir = std::string(home_dir) + base_cache_dir.substr(1);
            }
        }
        
        // Debug output
        LOG_DEBUG_TAG("Scanning cache directory: " + base_cache_dir, "ModelManager");
        LOG_DEBUG_TAG("Expanded path: " + expanded_cache_dir, "ModelManager");
        
        // List Hugging Face models (direct storage: owner/model)
        int result = ListLocalModelsDirectories(expanded_cache_dir, model_names, verbose);
        
        // List ModelScope models (direct storage: modelscope/owner/model)
        std::string modelscope_dir = expanded_cache_dir + "/modelscope";
        std::vector<std::string> modelscope_models;
        if (ListLocalModelsDirectories(modelscope_dir, modelscope_models, verbose) == 0) {
            for (auto& name : modelscope_models) {
                model_names.emplace_back("modelscope/" + name);
            }
        }
        
        // List Modelers models (symlinks in base directory)
        std::string modelers_dir = expanded_cache_dir + "/modelers";
        std::vector<std::string> modelers_models;
        if (ListLocalModels(modelers_dir, modelers_models) == 0) {
            for (auto& name : modelers_models) {
                model_names.emplace_back("modelers/" + name);
            }
        }
        
        return 0;
    }
    
    static int ListLocalModels(bool verbose = false) {
        std::vector<std::string> model_names;
        int result = GetLocalModelNames(model_names, verbose);
        
        if (!model_names.empty()) {
            std::cout << "Local models:\n";
            for (auto& name : model_names) {
                std::cout << "  📁 " << name << "\n";
            }
        } else {
            std::cout << "No local models found.\n";
            std::cout << "Use 'mnncli search <keyword>' to search remote models\n";
            std::cout << "Use 'mnncli download <name>' to download models\n";
        }
        return result;
    }
    
    static int SearchRemoteModels(const std::string& keyword, bool verbose = false, const std::string& cache_dir_override = "") {
        try {
            // Get cache directory from override, config, or use default
            auto config = ConfigManager::LoadDefaultConfig();
            std::string cache_dir;
            if (!cache_dir_override.empty()) {
                cache_dir = cache_dir_override;
            } else {
                cache_dir = config.cache_dir;
                if (cache_dir.empty()) {
                    cache_dir = mnncli::FileUtils::GetBaseCacheDir();
                }
            }
            
            // Get current download provider from config
            std::string current_provider = config.download_provider;
            if (current_provider.empty()) {
                current_provider = "HuggingFace"; // Default provider
            }
            
            // Create ModelRepository instance
            auto& model_repo = mnncli::ModelRepository::GetInstance(cache_dir);
            model_repo.SetDownloadProvider(current_provider);
            
            // Show search information
            std::cout << "🔍 Searching for LLM models with keyword: '" << keyword << "'\n";
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
                    std::cout << "No LLM models found for provider: " << current_provider << "\n";
                } else {
                    std::cout << "No LLM models found matching keyword: '" << keyword << "' for provider: " << current_provider << "\n";
                }
                std::cout << "\n💡 Try:\n";
                std::cout << "  • Use a different keyword\n";
                std::cout << "  • Change download provider: export MNN_DOWNLOAD_PROVIDER=<provider>\n";
                std::cout << "  • Available providers: HuggingFace, ModelScope, Modelers\n";
            } else {
                std::cout << "Found " << searchResults.size() << " matching LLM model(s):\n\n";
                
                // Display results in a table format
                std::cout << std::left << std::setw(50) << "Model Name"
                          << std::setw(15) << "Vendor"
                          << std::setw(12) << "Size"
                          << std::setw(15) << "Tags" << "\n";
                std::cout << std::string(92, '-') << "\n";
                
                for (const auto& model : searchResults) {
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
                        size_str = mnncli::LogUtils::FormatFileSize(model.file_size);
                    } else if (model.size_gb > 0) {
                        // Fallback to size_gb if file_size is not available
                        size_str = mnncli::LogUtils::FormatFileSize(static_cast<int64_t>(model.size_gb * 1024 * 1024 * 1024));
                    } else {
                        size_str = "N/A";
                    }

                    std::cout << std::left << std::setw(50) << model.modelName
                              << std::setw(15) << model.vendor
                              << std::setw(12) << size_str
                              << std::setw(15) << tags_str << "\n";
                }
                
                std::cout << "\n💡 To download a model, use:\n";
                std::cout << "  mnncli download <model_name>\n";
                std::cout << "  Example: mnncli download " << searchResults[0].modelName << "\n";
            }
            
        } catch (const std::exception& e) {
            mnncli::UserInterface::ShowError("Failed to search models: " + std::string(e.what()));
            return 1;
        }
        return 0;
    }
    
    static int DownloadModel(const std::string& model_name, bool verbose = false, const std::string& cache_dir_override = "") {
        if (model_name.empty()) {
            mnncli::UserInterface::ShowError("Model name is required", "Usage: mnncli model download <name>");
            return 1;
        }
        
        LOG_INFO("Downloading model: " + model_name);
        
        // Get current configuration
        auto config = ConfigManager::LoadDefaultConfig();
        
        // Show which download provider will be used
        LOG_INFO("Using download provider: " + config.download_provider);
        
        // Early validation for obviously invalid model names
        if (!isValidModelName(model_name)) {
            mnncli::UserInterface::ShowError("Invalid model name format: '" + model_name + "'");
            std::cout << "\n💡 Valid model name formats:\n";
            std::cout << "  • Simple name (e.g., 'qwen-7b') - will search in repository\n";
            std::cout << "  • Full ID (e.g., 'Qwen/Qwen-7B-Chat') - direct download\n";
            std::cout << "  • Prefixed ID (e.g., 'hf:Qwen/Qwen-7B-Chat') - specify provider\n";
            std::cout << "\n💡 Try:\n";
            std::cout << "  • Use 'mnncli search " << model_name << "' to find available models\n";
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
                    cache_dir = mnncli::FileUtils::GetBaseCacheDir();
                }
            }
            
            // Create download manager instance
            auto& download_manager = mnncli::ModelDownloadManager::GetInstance(cache_dir);
            
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
            if (source.empty()) {
                if (model_name.find("hf:") == 0 || model_name.find("huggingface:") == 0) {
                    source = "HuggingFace";
                    model_id = model_name.substr(model_name.find(":") + 1);
                } else if (model_name.find("ms:") == 0 || model_name.find("modelscope:") == 0) {
                    source = "ModelScope";
                    model_id = model_name.substr(model_name.find(":") + 1);
                } else if (model_name.find("ml:") == 0 || model_name.find("modelers:") == 0) {
                    source = "Modelers";
                    model_id = model_name.substr(model_name.find(":") + 1);
                } else {
                    // Try to use ModelRepository to find the model
                    try {
                        auto& model_repo = mnncli::ModelRepository::GetInstance(cache_dir);
                        model_repo.SetDownloadProvider("HuggingFace"); // Default to HuggingFace
                        
                        auto model_id_opt = model_repo.GetModelIdForDownload(model_name, "HuggingFace");
                        if (model_id_opt) {
                            model_id = *model_id_opt;
                            source = "HuggingFace";
                            LOG_INFO("✓ Found model in repository: " + model_id);
                        } else {
                            // Fallback to default behavior
                            source = "HuggingFace";
                            model_id = model_name;
                        }
                    } catch (const std::exception& e) {
                        LOG_DEBUG_TAG("Failed to use ModelRepository: " + std::string(e.what()), "ModelManager");
                        // Fallback to default behavior
                        source = "HuggingFace";
                        model_id = model_name;
                    }
                }
            } else {
                // Source is specified, try to use ModelRepository
                try {
                    auto& model_repo = mnncli::ModelRepository::GetInstance(cache_dir);
                    model_repo.SetDownloadProvider(source);
                    
                    auto model_id_opt = model_repo.GetModelIdForDownload(model_name, source);
                    if (model_id_opt) {
                        model_id = *model_id_opt;
                        LOG_INFO("✓ Found model in repository: " + model_id);
                        
                        // Get model type for additional info
                        std::string model_type = model_repo.GetModelType(model_id);
                        LOG_INFO("  Model type: " + model_type);
                    } else {
                        // Fallback to direct model name
                        model_id = model_name;
                        LOG_WARNING("⚠ Model not found in repository, using direct name: " + model_id);
                    }
                } catch (const std::exception& e) {
                    LOG_DEBUG_TAG("Failed to use ModelRepository: " + std::string(e.what()), "ModelManager");
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
            LOG_DEBUG("[DEBUG] Waiting for download to complete, model_id: " + model_id);
            while (download_manager.IsDownloading(model_id)) {
                LOG_DEBUG("[DEBUG] Still downloading, model_id: " + model_id);
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            LOG_DEBUG("[DEBUG] Download loop exited, model_id: " + model_id);
            
            // Check final status
            auto download_info = download_manager.GetDownloadInfo(model_id);
            if (download_info.state == mnncli::DownloadState::COMPLETED) {
                auto downloaded_file = download_manager.GetDownloadedFile(model_id);
                if (!downloaded_file.empty() && std::filesystem::exists(downloaded_file)) {
                    mnncli::UserInterface::ShowSuccess("Model downloaded successfully!");
                    mnncli::UserInterface::ShowInfo("Model saved to: " + downloaded_file.string());
                    return 0;
                }
            } else if (download_info.state == mnncli::DownloadState::FAILED) {
                mnncli::UserInterface::ShowError("Download failed. Check the error messages above.");
                return 1;
            }
            
            // Remove listener before returning
            download_manager.RemoveListener(&cli_listener);
            
        } catch (const std::exception& e) {
            mnncli::UserInterface::ShowError("Failed to download model: " + std::string(e.what()));
            return 1;
        }
        return 0;
    }
    
    static int DeleteModel(const std::string& model_name) {
        if (model_name.empty()) {
            mnncli::UserInterface::ShowError("Model name is required", "Usage: mnncli delete <name>");
            return 1;
        }
        
        LOG_INFO("Deleting model: " + model_name);
        try {
            // Use the same logic as ListLocalModels to find models
            auto config = ConfigManager::LoadDefaultConfig();
            std::string base_cache_dir = config.cache_dir;
            if (base_cache_dir.empty()) {
                base_cache_dir = "~/.cache/mnncli";
            }
            
            // Expand tilde in path
            std::string expanded_cache_dir = base_cache_dir;
            if (base_cache_dir[0] == '~') {
                const char* home_dir = getenv("HOME");
                if (home_dir) {
                    expanded_cache_dir = std::string(home_dir) + base_cache_dir.substr(1);
                }
            }
            
            // If config points to mnnmodels subdirectory, use parent directory for actual models
            if (expanded_cache_dir.find("/mnnmodels") != std::string::npos) {
                expanded_cache_dir = expanded_cache_dir.substr(0, expanded_cache_dir.find("/mnnmodels"));
            }
            
            bool deleted = false;
            
            // Use the same logic as ListLocalModels to find and delete models
            if (model_name.find("modelscope/") == 0) {
                // Handle modelscope models
                std::string actual_model_name = model_name.substr(11); // Remove "modelscope/" prefix
                std::string modelscope_dir = expanded_cache_dir + "/modelscope";
                
                // Find the symlink
                std::string symlink_path = modelscope_dir + "/" + actual_model_name;
                if (mnncli::FileUtils::RemoveFileIfExists(symlink_path)) {
                    LOG_INFO("Deleted symlink: " + symlink_path);
                    deleted = true;
                }
                
                // Find and delete the storage folder
                std::string storage_folder_pattern = "models--MNN--" + actual_model_name;
                std::string storage_path = modelscope_dir + "/" + storage_folder_pattern;
                if (mnncli::FileUtils::RemoveFileIfExists(storage_path)) {
                    LOG_INFO("Deleted storage folder: " + storage_path);
                    deleted = true;
                }
            } else if (model_name.find("modelers/") == 0) {
                // Handle modelers models
                std::string actual_model_name = model_name.substr(9); // Remove "modelers/" prefix
                std::string modelers_dir = expanded_cache_dir + "/modelers";
                
                // Find the symlink
                std::string symlink_path = modelers_dir + "/" + actual_model_name;
                if (mnncli::FileUtils::RemoveFileIfExists(symlink_path)) {
                    LOG_INFO("Deleted symlink: " + symlink_path);
                    deleted = true;
                }
                
                // Find and delete the storage folder
                std::string storage_folder_pattern = "models--" + actual_model_name;
                std::string storage_path = modelers_dir + "/" + storage_folder_pattern;
                if (mnncli::FileUtils::RemoveFileIfExists(storage_path)) {
                    LOG_INFO("Deleted storage folder: " + storage_path);
                    deleted = true;
                }
            } else {
                // Handle models in base cache directory (taobao-mnn models)
                std::string symlink_path = expanded_cache_dir + "/" + model_name;
                if (mnncli::FileUtils::RemoveFileIfExists(symlink_path)) {
                    LOG_INFO("Deleted symlink: " + symlink_path);
                    deleted = true;
                }
                
                // Find and delete the storage folder
                std::string storage_folder_pattern = "models--taobao-mnn--" + model_name;
                std::string storage_path = expanded_cache_dir + "/" + storage_folder_pattern;
                if (mnncli::FileUtils::RemoveFileIfExists(storage_path)) {
                    LOG_INFO("Deleted storage folder: " + storage_path);
                    deleted = true;
                }
            }
            
            if (deleted) {
                mnncli::UserInterface::ShowSuccess("Model deleted successfully: " + model_name);
            } else {
                mnncli::UserInterface::ShowError("Model not found: " + model_name);
                return 1;
            }
        } catch (const std::exception& e) {
            mnncli::UserInterface::ShowError("Failed to delete model: " + std::string(e.what()));
            return 1;
        }
        return 0;
    }
    
    static int ShowModelInfo(const std::string& model_name, bool verbose = false) {
        if (model_name.empty()) {
            mnncli::UserInterface::ShowError("Model name is required", "Usage: mnncli model info <name>");
            return 1;
        }
        
        LOG_INFO("Showing model info: " + model_name);
        
        try {
            std::string model_path = mnncli::FileUtils::GetModelPath(model_name);
            
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
                for (const auto& entry : fs::recursive_directory_iterator(model_path)) {
                    if (entry.is_regular_file()) {
                        total_size += entry.file_size();
                    }
                }
                std::cout << "💾 Total Size: " << mnncli::LogUtils::FormatFileSize(total_size) << "\n";
            } catch (const std::exception& e) {
                LOG_DEBUG_TAG("Failed to calculate directory size: " + std::string(e.what()), "ModelManager");
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
                        const int max_lines = verbose ? 1000 : 50; // Limit output unless verbose
                        
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
                } catch (const std::exception& e) {
                    std::cout << "❌ Error reading config file: " << e.what() << "\n";
                }
            } else {
                std::cout << "\n⚠️  No config.json found in model directory\n";
                
                // List available files
                std::cout << "\n📂 Available Files:\n";
                try {
                    for (const auto& entry : fs::directory_iterator(model_path)) {
                        if (entry.is_regular_file()) {
                            std::cout << "  📄 " << entry.path().filename().string() << "\n";
                        } else if (entry.is_directory()) {
                            std::cout << "  📁 " << entry.path().filename().string() << "/\n";
                        }
                    }
                } catch (const std::exception& e) {
                    std::cout << "❌ Error listing files: " << e.what() << "\n";
                }
            }
            
            // Show additional info if verbose
            if (verbose) {
                std::cout << "\n🔍 Additional Information:\n";
                std::cout << "====================\n";
                auto config = ConfigManager::LoadDefaultConfig();
                std::string cache_dir = config.cache_dir;
                if (cache_dir.empty()) {
                    cache_dir = mnncli::FileUtils::GetBaseCacheDir();
                }
                std::cout << "Cache Directory: " << cache_dir << "\n";
                std::cout << "Download Provider: " << config.download_provider << "\n";
                
                // Check for other common model files
                std::vector<std::string> common_files = {
                    "tokenizer.json", "tokenizer_config.json", "vocab.txt", 
                    "merges.txt", "special_tokens_map.json", "model.mnn",
                    "pytorch_model.bin", "model.safetensors", "model.bin"
                };
                
                std::cout << "\n📋 Model Files Status:\n";
                for (const auto& file : common_files) {
                    std::string file_path = model_path + "/" + file;
                    if (fs::exists(file_path)) {
                        try {
                            auto file_size = fs::file_size(file_path);
                            std::cout << "  ✅ " << file << " (" << mnncli::LogUtils::FormatFileSize(file_size) << ")\n";
                        } catch (...) {
                            std::cout << "  ✅ " << file << " (size unknown)\n";
                        }
                    } else {
                        std::cout << "  ❌ " << file << "\n";
                    }
                }
            }
            
        } catch (const std::exception& e) {
            mnncli::UserInterface::ShowError("Failed to show model info: " + std::string(e.what()));
            return 1;
        }
        
        return 0;
    }
    
private:
    static bool isValidModelName(const std::string& model_name) {
        if (model_name.empty()) {
            return false;
        }
        
        // Check for invalid characters
        for (char c : model_name) {
            if (!std::isalnum(c) && c != '-' && c != '_' && c != '/' && c != ':' && c != '.') {
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
    
public:
    static int ListLocalModels(const std::string& directory_path, std::vector<std::string>& model_names) {
        std::error_code ec;
        if (!fs::exists(directory_path, ec)) {
            return 1;
        }
        if (!fs::is_directory(directory_path, ec)) {
            return 1;
        }
        for (const auto& entry : fs::directory_iterator(directory_path, ec)) {
            if (ec) {
                return 1;
            }
            if (fs::is_symlink(entry, ec)) {
                if (ec) {
                    return 1;
                }
                std::string file_name = entry.path().filename().string();
                model_names.emplace_back(file_name);
            }
        }
        std::sort(model_names.begin(), model_names.end());
        return 0;
    }
    
    static int ListLocalModelsDirectories(const std::string& directory_path, std::vector<std::string>& model_names, bool verbose) {
        std::error_code ec;
        if (!fs::exists(directory_path, ec)) {
            LOG_DEBUG_TAG("Directory does not exist: " + directory_path, "ModelManager");
            return 1;
        }
        if (!fs::is_directory(directory_path, ec)) {
            LOG_DEBUG_TAG("Path is not a directory: " + directory_path, "ModelManager");
            return 1;
        }
        
        LOG_DEBUG_TAG("Scanning directory: " + directory_path, "ModelManager");
        
        for (const auto& entry : fs::directory_iterator(directory_path, ec)) {
            if (ec) {
                LOG_DEBUG_TAG("Error iterating directory: " + ec.message(), "ModelManager");
                return 1;
            }
            
            // Check if it's a directory (for direct storage models)
            if (fs::is_directory(entry, ec)) {
                if (ec) {
                    continue; // Skip entries we can't access
                }
                
                std::string dir_name = entry.path().filename().string();
                
                // Skip hidden directories and special directories
                if (dir_name.empty() || dir_name[0] == '.') {
                    continue;
                }
                
                // For ModelScope, we need to go one level deeper (modelscope/owner/model)
                if (directory_path.find("/modelscope") != std::string::npos) {
                    std::string owner_path = entry.path().string();
                    std::error_code owner_ec;
                    if (fs::is_directory(owner_path, owner_ec)) {
                        for (const auto& model_entry : fs::directory_iterator(owner_path, owner_ec)) {
                            if (owner_ec) {
                                continue;
                            }
                            if (fs::is_directory(model_entry, owner_ec)) {
                                std::string model_name = model_entry.path().filename().string();
                                if (!model_name.empty() && model_name[0] != '.') {
                                    model_names.emplace_back(dir_name + "/" + model_name);
                                }
                            }
                        }
                    }
                } else {
                    // For Hugging Face models, check if this directory contains model files
                    // or if it's a group directory containing nested models
                    bool has_model_files = false;
                    std::error_code model_ec;
                    
                    // First check if this directory directly contains model files
                    for (const auto& file_entry : fs::directory_iterator(entry.path(), model_ec)) {
                        if (model_ec) {
                            continue;
                        }
                        std::string file_name = file_entry.path().filename().string();
                        // Check for common model file extensions or config files
                        if (file_name.length() >= 4 && (
                            file_name.substr(file_name.length() - 4) == ".bin" ||
                            file_name.substr(file_name.length() - 4) == ".txt" ||
                            file_name.substr(file_name.length() - 4) == ".onnx" ||
                            file_name.substr(file_name.length() - 4) == ".mnn" ||
                            (file_name.length() >= 5 && file_name.substr(file_name.length() - 5) == ".json") ||
                            (file_name.length() >= 11 && file_name.substr(file_name.length() - 11) == ".safetensors"))) {
                            has_model_files = true;
                            break;
                        }
                    }
                    
                    if (has_model_files) {
                        model_names.emplace_back(dir_name);
                    } else {
                        // If no direct model files, check if this is a group directory with nested models
                        // (e.g., taobao-mnn/SmolVLM-256M-Instruct-MNN)
                        for (const auto& nested_entry : fs::directory_iterator(entry.path(), model_ec)) {
                            if (model_ec) {
                                continue;
                            }
                            if (fs::is_directory(nested_entry, model_ec)) {
                                std::string nested_name = nested_entry.path().filename().string();
                                if (!nested_name.empty() && nested_name[0] != '.') {
                                    // Check if nested directory contains model files
                                    bool nested_has_model_files = false;
                                    std::error_code nested_ec;
                                    for (const auto& nested_file_entry : fs::directory_iterator(nested_entry.path(), nested_ec)) {
                                        if (nested_ec) {
                                            continue;
                                        }
                                        std::string nested_file_name = nested_file_entry.path().filename().string();
                                        if (nested_file_name.length() >= 4 && (
                                            nested_file_name.substr(nested_file_name.length() - 4) == ".bin" ||
                                            nested_file_name.substr(nested_file_name.length() - 4) == ".txt" ||
                                            nested_file_name.substr(nested_file_name.length() - 4) == ".onnx" ||
                                            nested_file_name.substr(nested_file_name.length() - 4) == ".mnn" ||
                                            (nested_file_name.length() >= 5 && nested_file_name.substr(nested_file_name.length() - 5) == ".json") ||
                                            (nested_file_name.length() >= 11 && nested_file_name.substr(nested_file_name.length() - 11) == ".safetensors"))) {
                                            nested_has_model_files = true;
                                            break;
                                        }
                                    }
                                    
                                    if (nested_has_model_files) {
                                        model_names.emplace_back(dir_name + "/" + nested_name);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // Also check for symlinks (for backward compatibility)
            else if (fs::is_symlink(entry, ec)) {
                if (ec) {
                    continue;
                }
                std::string file_name = entry.path().filename().string();
                model_names.emplace_back(file_name);
            }
        }
        
        std::sort(model_names.begin(), model_names.end());
        LOG_DEBUG_TAG("Found " + std::to_string(model_names.size()) + " models in " + directory_path, "ModelManager");
        return 0;
    }
};

// Using ModelRunner for interactive chat and performance evaluation

// Performance evaluation using ModelRunner

// Main command line interface
class CommandLineInterface {
public:
    CommandLineInterface() : verbose_(false) {}
    
    int Run(int argc, const char* argv[]) {
        if (argc < 2) {
            PrintUsage();
            return 0;
        }
        
        // Parse global options first
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "-v" || arg == "--verbose") {
                verbose_ = true;
                // Set global verbose logging
                mnncli::LogUtils::SetVerbose(true);
                // Remove the verbose flag from argv by shifting remaining args
                for (int j = i; j < argc - 1; j++) {
                    argv[j] = argv[j + 1];
                }
                argc--;
                break;
            }
        }
        
        std::string cmd = argv[1];
        
        try {
            if (cmd == "list") {
                return ModelManager::ListLocalModels(verbose_);
            } else if (cmd == "search") {
                if (argc < 3) {
                    mnncli::UserInterface::ShowError("Search keyword required", "Usage: mnncli search <keyword>");
                    return 1;
                }
                return ModelManager::SearchRemoteModels(argv[2], verbose_);
            } else if (cmd == "download") {
                if (argc < 3) {
                    mnncli::UserInterface::ShowError("Model name required", "Usage: mnncli download <name>");
                    return 1;
                }
                return ModelManager::DownloadModel(argv[2], verbose_);
            } else if (cmd == "delete") {
                if (argc < 3) {
                    mnncli::UserInterface::ShowError("Model name required", "Usage: mnncli delete <name>");
                    return 1;
                }
                return ModelManager::DeleteModel(argv[2]);
            } else if (cmd == "model_info") {
                if (argc < 3) {
                    mnncli::UserInterface::ShowError("Model name required", "Usage: mnncli model_info <name>");
                    return 1;
                }
                return ModelManager::ShowModelInfo(argv[2], verbose_);
            } else if (cmd == "run") {
                return HandleRunCommand(argc, argv);
            } else if (cmd == "serve") {
                return HandleServeCommand(argc, argv);
            } else if (cmd == "benchmark") {
                return HandleBenchmarkCommand(argc, argv);
            } else if (cmd == "config") {
                return HandleConfigCommand(argc, argv);
            } else if (cmd == "info") {
                return HandleInfoCommand(argc, argv);
            } else if (cmd == "web") {
                return HandleWebSearchCommand(argc, argv);
            } else if (cmd == "email") {
                return HandleEmailCommand(argc, argv);
            } else if (cmd == "calendar") {
                return HandleCalendarCommand(argc, argv);
            } else if (cmd == "share") {
                return HandleShareCommand(argc, argv);
            } else if (cmd == "--help" || cmd == "-h") {
                PrintUsage();
                return 0;
            } else if (cmd == "--version" || cmd == "-v") {
                PrintVersion();
                return 0;
            } else {
                mnncli::UserInterface::ShowError("Unknown command: " + cmd, "Use 'ARIA --help' for usage information");
                return 1;
            }
        } catch (const std::exception& e) {
            mnncli::UserInterface::ShowError("Unexpected error: " + std::string(e.what()));
            return 1;
        }
        
        return 0;
    }
    
private:
    
    int HandleRunCommand(int argc, const char* argv[]) {
        std::string model_name;
        std::string config_path;
        std::string prompt;
        std::string prompt_file;

        // Parse options first to see if we have a config path or prompt
        for (int i = 2; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "-c" || arg == "--config") {
                if (++i >= argc) {
                    mnncli::UserInterface::ShowError("Missing config path", "Usage: -c <config_path>");
                    PrintRunUsage();
                    return 1;
                }
                config_path = mnncli::FileUtils::ExpandTilde(argv[i]);
            } else if (arg == "-p" || arg == "--prompt") {
                if (++i >= argc) {
                    mnncli::UserInterface::ShowError("Missing prompt text", "Usage: -p <prompt_text>");
                    PrintRunUsage();
                    return 1;
                }
                prompt = argv[i];
            } else if (arg == "-f" || arg == "--file") {
                if (++i >= argc) {
                    mnncli::UserInterface::ShowError("Missing prompt file", "Usage: -f <prompt_file>");
                    PrintRunUsage();
                    return 1;
                }
                prompt_file = argv[i];
            }
        }

        // If no config path specified, check if we have a model name or can use default model
        if (config_path.empty()) {
            // Check if argv[2] exists and is not an option (doesn't start with -)
            bool has_model_name = (argc >= 3) && (argv[2][0] != '-');

            if (!has_model_name) {
                // No model name provided, try to use default model
                auto config = ConfigManager::LoadDefaultConfig();
                if (!config.default_model.empty()) {
                    model_name = config.default_model;
                    LOG_INFO("Using default model: " + model_name);
                } else {
                    mnncli::UserInterface::ShowError("Model name required and no default model set",
                                                    "Set a default model with: mnncli config set default_model <model_name>");
                    PrintRunUsage();
                    return 1;
                }
            } else {
                // We have a model name
                model_name = argv[2];
            }
            config_path = mnncli::FileUtils::GetConfigPath(model_name);
        } else {
            // If config path is specified, extract model name from path or use a default
            std::string config_dir = fs::path(config_path).parent_path().string();
            std::string config_filename = fs::path(config_dir).filename().string();
            if (!config_filename.empty()) {
                model_name = config_filename;
            } else {
                model_name = "custom_model";
            }
        }
        
        // Parse remaining options (backward compatibility - in case some options were not processed in first loop)
        for (int i = 2; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "-p" || arg == "--prompt") {
                if (++i >= argc) {
                    mnncli::UserInterface::ShowError("Missing prompt text", "Usage: -p <prompt_text>");
                    PrintRunUsage();
                    return 1;
                }
                if (prompt.empty()) {  // Only set if not already set
                    prompt = argv[i];
                }
            } else if (arg == "-f" || arg == "--file") {
                if (++i >= argc) {
                    mnncli::UserInterface::ShowError("Missing prompt file", "Usage: -f <prompt_file>");
                    PrintRunUsage();
                    return 1;
                }
                if (prompt_file.empty()) {  // Only set if not already set
                    prompt_file = argv[i];
                }
            } else if (arg == "-c" || arg == "--config") {
                // Skip the config path that follows
                i++;
            }
        }
        
        if (config_path.empty()) {
            mnncli::UserInterface::ShowError("Config path is empty", "Unable to determine config path for model: " + model_name);
            PrintRunUsage();
            return 1;
        }
        
        LOG_INFO("Starting model: " + model_name);
        LOG_INFO("Config path: " + config_path);
        
        auto llm = LLMManager::CreateLLM(config_path, true);
        
        if (prompt.empty() && prompt_file.empty()) {
            ModelRunner runner(llm.get());
            runner.InteractiveChat();
        } else if (!prompt.empty()) {
            ModelRunner runner(llm.get());
            runner.EvalPrompts({prompt});
        } else {
            ModelRunner runner(llm.get());
            runner.EvalFile(prompt_file);
        }
        
        return 0;
    }
    
    int HandleServeCommand(int argc, const char* argv[]) {
        if (argc < 3) {
            mnncli::UserInterface::ShowError("Model name required", "Usage: mnncli serve <model_name> [options]");
            return 1;
        }
        
        std::string model_name = argv[2];
        std::string config_path = mnncli::FileUtils::GetConfigPath(model_name);
        std::string host = "127.0.0.1";
        int port = 8000;
        
        // Parse options
        for (int i = 3; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "-c" || arg == "--config") {
                if (++i >= argc) {
                    mnncli::UserInterface::ShowError("Missing config path", "Usage: -c <config_path>");
                    return 1;
                }
                config_path = mnncli::FileUtils::ExpandTilde(argv[i]);
            } else if (arg == "--host") {
                if (++i >= argc) {
                    mnncli::UserInterface::ShowError("Missing host", "Usage: --host <host>");
                    return 1;
                }
                host = argv[i];
            } else if (arg == "--port") {
                if (++i >= argc) {
                    mnncli::UserInterface::ShowError("Missing port", "Usage: --port <port>");
                    return 1;
                }
                port = std::stoi(argv[i]);
            }
        }
        
        std::cout << "🚀 Starting MNN CLI API Server\n";
        std::cout << "📦 Model: " << model_name << "\n";
        std::cout << "🌐 Server: http://" << host << ":" << port << "\n";
        std::cout << "📋 API Endpoints:\n";
        std::cout << "   - GET  /                    - Web interface\n";
        std::cout << "   - POST /chat/completions     - OpenAI-compatible chat API\n";
        std::cout << "   - POST /v1/chat/completions  - OpenAI-compatible chat API (v1)\n";
        std::cout << "   - POST /reset               - Reset conversation\n";
        std::cout << "⏳ Initializing model...\n";
        
        mnncli::MnncliServer server;
        bool is_r1 = IsR1(config_path);
        auto llm = LLMManager::CreateLLM(config_path, !is_r1);
        server.Start(llm.get(), is_r1, host, port);
        
        return 0;
    }
    
    int HandleBenchmarkCommand(int argc, const char* argv[]) {
        if (argc < 3) {
            mnncli::UserInterface::ShowError("Model name required", "Usage: mnncli benchmark <model_name> [options]");
            return 1;
        }
        
        std::string model_name = argv[2];
        std::string config_path = mnncli::FileUtils::GetConfigPath(model_name);
        
        // Parse options
        for (int i = 3; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "-c" || arg == "--config") {
                if (++i >= argc) {
                    mnncli::UserInterface::ShowError("Missing config path", "Usage: -c <config_path>");
                    return 1;
                }
                config_path = argv[i];
            }
        }
        
        if (config_path.empty()) {
            mnncli::UserInterface::ShowError("Config path is empty", "Use -c to specify config path");
            return 1;
        }
        
        LOG_INFO("Starting benchmark for model: " + model_name);
        
        auto llm = LLMManager::CreateLLM(config_path, true);
        mnncli::LLMBenchmark benchmark;
        benchmark.Start(llm.get(), {});
        
        return 0;
    }
    
    
    int HandleConfigCommand(int argc, const char* argv[]) {
        if (argc < 3) {
            ConfigManager::ShowConfig(ConfigManager::LoadDefaultConfig());
            return 0;
        }
        
        std::string subcmd = argv[2];
        if (subcmd == "show") {
            ConfigManager::ShowConfig(ConfigManager::LoadDefaultConfig());
        } else if (subcmd == "set") {
            if (argc < 5) {
                mnncli::UserInterface::ShowError("Missing key or value", "Usage: mnncli config set <key> <value>");
                return 1;
            }
            
            std::string key = argv[3];
            std::string value = argv[4];
            
            auto config = ConfigManager::LoadDefaultConfig();
            if (ConfigManager::SetConfigValue(config, key, value)) {
                if (ConfigManager::SaveConfig(config)) {
                    mnncli::UserInterface::ShowSuccess("Configuration updated and saved: " + key + " = " + value);
                } else {
                    mnncli::UserInterface::ShowSuccess("Configuration updated: " + key + " = " + value);
                    LOG_WARNING("Warning: Configuration could not be saved to file.");
                }
            } else {
                mnncli::UserInterface::ShowError("Invalid configuration key or value", 
                    "Use 'mnncli config help' to see available options");
                return 1;
            }
        } else if (subcmd == "reset") {
            mnncli::UserInterface::ShowInfo("Config reset command not implemented yet");
        } else if (subcmd == "help") {
            std::cout << ConfigManager::GetConfigHelp();
        } else {
            mnncli::UserInterface::ShowError("Unknown config subcommand", "Use: show, set, reset, or help");
            return 1;
        }
        
        return 0;
    }
    
    int HandleInfoCommand(int argc, const char* argv[]) {
        std::cout << "MNN CLI Information:\n";
        std::cout << "====================\n";
        std::cout << "Version: 1.0.0\n";
        
        auto config = ConfigManager::LoadDefaultConfig();
        std::cout << "Cache Directory: " << config.cache_dir << "\n";
        std::cout << "Download Provider: " << config.download_provider << "\n";
        std::cout << "API Server: " << config.api_host << ":" << config.api_port << "\n";
        
        std::cout << "Available Models: ";
        
        // Use the same logic as 'list' command to count models
        std::vector<std::string> model_names;
        ModelManager::GetLocalModelNames(model_names, false);
        std::cout << model_names.size() << "\n";
        
        std::cout << "\nEnvironment Variables:\n";
        if (const char* env_provider = std::getenv("MNN_DOWNLOAD_PROVIDER")) {
            std::cout << "  MNN_DOWNLOAD_PROVIDER: " << env_provider << "\n";
        }
        if (const char* env_cache = std::getenv("MNN_CACHE_DIR")) {
            std::cout << "  MNN_CACHE_DIR: " << env_cache << "\n";
        }
        if (const char* env_host = std::getenv("MNN_API_HOST")) {
            std::cout << "  MNN_API_HOST: " << env_host << "\n";
        }
        if (const char* env_port = std::getenv("MNN_API_PORT")) {
            std::cout << "  MNN_API_PORT: " << env_port << "\n";
        }
        
        return 0;
    }

    int HandleWebSearchCommand(int argc, const char* argv[]) {
#if defined(_WIN32)
        ShellExecute(0, 0, "https://www.google.com/search?q=ARIA", 0, 0, SW_SHOW);
#else
        system("open https://www.google.com/search?q=ARIA");
#endif
        return 0;
    }

    int HandleEmailCommand(int argc, const char* argv[]) {
#if defined(_WIN32)
        ShellExecute(0, 0, "mailto:test@example.com?subject=ARIA%20Test", 0, 0, SW_SHOW);
#else
        system("open mailto:test@example.com?subject=ARIA%20Test");
#endif
        return 0;
    }

    int HandleCalendarCommand(int argc, const char* argv[]) {
#if defined(_WIN32)
        ShellExecute(0, 0, "outlookcal:", 0, 0, SW_SHOW);
#else
        mnncli::UserInterface::ShowInfo("Calendar integration is not yet implemented for this platform.");
#endif
        return 0;
    }

    int HandleShareCommand(int argc, const char* argv[]) {
#if defined(_WIN32)
        // This is a placeholder for a more complex share implementation
        mnncli::UserInterface::ShowInfo("Share functionality is not yet implemented for this platform.");
#else
        mnncli::UserInterface::ShowInfo("Share functionality is not yet implemented for this platform.");
#endif
        return 0;
    }
    
    
    void PrintUsage() {
        std::cout << "ARIA - AI Model Command Line Interface\n\n";
        std::cout << "Usage: ARIA <command> [options]\n\n";
        std::cout << "Commands:\n";
        std::cout << "  list       List local models\n";
        std::cout << "  search     Search remote models\n";
        std::cout << "  download   Download model\n";
        std::cout << "  delete     Delete model\n";
        std::cout << "  model_info Show model information, download location, and config content\n";
        std::cout << "  run        Run model inference\n";
        std::cout << "  serve      Start API server\n";
        std::cout << "  benchmark  Run performance benchmarks\n";
        std::cout << "  config     Manage configuration (show, set, reset, help)\n";
        std::cout << "  info       Show system information\n";
        std::cout << "  web        Perform a web search for 'ARIA'\n";
        std::cout << "  email      Open the default email client with a sample message\n";
        std::cout << "  calendar   (Not implemented) Create a calendar event\n";
        std::cout << "  share      (Not implemented) Share a message\n";
        std::cout << "\nGlobal Options:\n";
        std::cout << "  -v, --verbose  Enable verbose output for detailed debugging\n";
        std::cout << "  --help    Show this help message\n";
        std::cout << "  --version Show version information\n";
        std::cout << "\nExamples:\n";
        std::cout << "  mnncli list                          # List local models\n";
        std::cout << "  mnncli search qwen                   # Search for Qwen models\n";
        std::cout << "  mnncli download qwen-7b             # Download Qwen-7B model\n";
        std::cout << "  mnncli download qwen-7b -v          # Download with verbose output\n";
        std::cout << "  mnncli model_info qwen-7b           # Show model information and config\n";
        std::cout << "  mnncli model_info qwen-7b -v        # Show detailed model information\n";
        std::cout << "  mnncli config set download_provider modelscope  # Set default provider\n";
        std::cout << "  mnncli config show                   # Show current configuration\n";
        std::cout << "  mnncli config help                   # Show configuration help\n";
        std::cout << "  mnncli run                           # Run default model in interactive mode\n";
        std::cout << "  mnncli run qwen-7b                  # Run Qwen-7B model\n";
        std::cout << "  mnncli run -p \"Hello world\"         # Run with prompt using default model\n";
        std::cout << "  mnncli serve qwen-7b --port 8000    # Start API server\n";
        std::cout << "  mnncli benchmark qwen-7b            # Run benchmark\n";
    }
    
    
    void PrintRunUsage() {
        std::cout << "Run Command Usage:\n";
        std::cout << "  mnncli run [model_name] [options]\n\n";
        std::cout << "Options:\n";
        std::cout << "  -c, --config <config_path>           # Specify custom config file path\n";
        std::cout << "  -p, --prompt <prompt_text>           # Provide prompt text directly\n";
        std::cout << "  -f, --file <prompt_file>             # Read prompts from file\n";
        std::cout << "  -v, --verbose                        # Enable verbose output\n\n";
        std::cout << "Examples:\n";
        std::cout << "  mnncli run                           # Run default model in interactive mode\n";
        std::cout << "  mnncli run qwen-7b                   # Run with default config\n";
        std::cout << "  mnncli run qwen-7b -p \"Hello\"        # Run with prompt\n";
        std::cout << "  mnncli run qwen-7b -f prompts.txt    # Run with prompt file\n";
        std::cout << "  mnncli run qwen-7b -c custom.json    # Run with custom config\n";
        std::cout << "  mnncli run qwen-7b -p \"Hello\" -v     # Run with prompt and verbose\n";
        std::cout << "  mnncli run -p \"Hello\"                # Run with prompt using default model\n";
        std::cout << "  mnncli run -f prompts.txt            # Run with prompt file using default model\n\n";
        std::cout << "Note: If no model_name is provided, the default model from configuration\n";
        std::cout << "      will be used (set with: mnncli config set default_model <name>)\n";
    }
    
    void PrintVersion() {
        std::cout << "MNN CLI version 1.0.0\n";
        std::cout << "Built with MNN framework\n";
    }
    
    static bool IsR1(const std::string& path) {
        std::string lowerModelName = path;
        std::transform(lowerModelName.begin(), lowerModelName.end(), lowerModelName.begin(), ::tolower);
        return lowerModelName.find("deepseek-r1") != std::string::npos;
    }
    
    bool verbose_;
};

int main(int argc, const char* argv[]) {
    mnncli::UserInterface::ShowWelcome();
    
    CommandLineInterface cli;
    return cli.Run(argc, argv);
}


