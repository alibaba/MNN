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
#include <algorithm>
#include <filesystem>
#include <thread>
#include <chrono>
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
#include "mnncli_server.hpp"
#include "nlohmann/json.hpp"
#include "log_utils.hpp"
#include "model_runner.hpp"

using namespace MNN::Transformer;
namespace fs = std::filesystem;

// Forward declarations
class CommandLineInterface;
class UserInterface;
class LLMManager;
class ModelManager;

// User interface utilities
class UserInterface {
public:
    static void ShowWelcome() {
        std::cout << "🚀 MNN CLI - MNN Command Line Interface\n";
        std::cout << "Type 'mnncli --help' for available commands\n\n";
    }
    
    static void ShowProgress(const std::string& message, float progress) {
        int bar_width = 50;
        int pos = bar_width * progress;
        
        std::cout << "\r" << message << " [";
        for (int i = 0; i < bar_width; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << int(progress * 100.0) << "%" << std::flush;
        
        if (progress >= 1.0) std::cout << std::endl;
    }
    
    static void ShowError(const std::string& error, const std::string& suggestion = "") {
        std::cerr << "❌ Error: " << error << std::endl;
        if (!suggestion.empty()) {
            std::cerr << "💡 Suggestion: " << suggestion << std::endl;
        }
    }
    
    static void ShowSuccess(const std::string& message) {
        std::cout << "✅ " << message << std::endl;
    }
    
    static void ShowInfo(const std::string& message) {
        std::cout << "ℹ️  " << message << std::endl;
    }
};

// CLI-specific download listener for user interface feedback
class CLIDownloadListener : public mnncli::DownloadListener {
public:
    CLIDownloadListener() = default;
    ~CLIDownloadListener() override = default;
    
    void onDownloadStart(const std::string& model_id) override {
        UserInterface::ShowInfo("Starting download: " + model_id);
    }
    
    void onDownloadProgress(const std::string& model_id, const mnncli::DownloadProgress& progress) override {
        std::string stage_info = progress.stage;
        if (!progress.current_file.empty()) {
            stage_info += " - " + progress.current_file;
        }
        
        std::string size_info;
        if (progress.total_size > 0) {
            size_info = " (" + mnncli::LogUtils::formatFileSize(progress.saved_size) + " / " + mnncli::LogUtils::formatFileSize(progress.total_size) + ")";
        } else {
            size_info = " (" + mnncli::LogUtils::formatFileSize(progress.saved_size) + ")";
        }
        
        std::string message = "Downloading " + model_id + " - " + stage_info + size_info;
        UserInterface::ShowProgress(message, progress.progress);
    }
    
    void onDownloadFinished(const std::string& model_id, const std::string& path) override {
        UserInterface::ShowSuccess("Download completed: " + model_id);
        UserInterface::ShowInfo("Model saved to: " + path);
    }
    
    void onDownloadFailed(const std::string& model_id, const std::string& error) override {
        UserInterface::ShowError("Download failed: " + model_id + " - " + error);
    }
    
    void onDownloadPaused(const std::string& model_id) override {
        UserInterface::ShowInfo("Download paused: " + model_id);
    }
    
    void onDownloadTaskAdded() override {
        UserInterface::ShowInfo("Download task added");
    }
    
    void onDownloadTaskRemoved() override {
        UserInterface::ShowInfo("Download task removed");
    }
    
    void onRepoInfo(const std::string& model_id, int64_t last_modified, int64_t repo_size) override {
        if (repo_size > 0) {
            UserInterface::ShowInfo("Repository info for " + model_id + ": " + mnncli::LogUtils::formatFileSize(repo_size));
        }
    }
    
    std::string getClassTypeName() const override {
        return "CLIDownloadListener";
    }
    

};

// Configuration management
namespace ConfigManager {
    struct Config {
        std::string default_model;
        std::string cache_dir;
        std::string log_level;
        int default_max_tokens;
        float default_temperature;
        std::string api_host;
        int api_port;
        int api_workers;
        std::string download_provider;  // "huggingface", "modelscope", or "modelers"
    };
    
    static std::string GetConfigFilePath() {
        std::string config_dir = mnncli::FileUtils::GetBaseCacheDir();
        return (fs::path(config_dir) / "mnncli_config.json").string();
    }
    
    static bool SaveConfig(const Config& config) {
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
            j["api_workers"] = config.api_workers;
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
    
    static Config LoadDefaultConfig() {
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
                    config.cache_dir = j.value("cache_dir", "~/.cache/mnncli");
                    config.log_level = j.value("log_level", "info");
                    config.default_max_tokens = j.value("default_max_tokens", 1000);
                    config.default_temperature = j.value("default_temperature", 0.7f);
                    config.api_host = j.value("api_host", "127.0.0.1");
                    config.api_port = j.value("api_port", 8000);
                    config.api_workers = j.value("api_workers", 4);
                    config.download_provider = j.value("download_provider", "huggingface");
                    
                    file.close();
                    
                    // Check environment variables (they take precedence over file)
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
        std::string download_provider = "huggingface"; // default
        if (const char* env_provider = std::getenv("MNN_DOWNLOAD_PROVIDER")) {
            download_provider = env_provider;
        }
        
        std::string cache_dir = "~/.cache/mnncli";
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
            .default_model = "",
            .cache_dir = cache_dir,
            .log_level = "info",
            .default_max_tokens = 1000,
            .default_temperature = 0.7f,
            .api_host = api_host,
            .api_port = api_port,
            .api_workers = 4,
            .download_provider = download_provider
        };
    }
    
    static void ShowConfig(const Config& config) {
        std::cout << "Configuration:\n";
        std::cout << "  Default Model(default_model): " << (config.default_model.empty() ? "Not set" : config.default_model) << "\n";
        std::cout << "  Cache Directory(cache_dir): " << config.cache_dir << "\n";
        std::cout << "  Log Level(log_level): " << config.log_level << "\n";
        std::cout << "  Default Max Tokens(default_max_tokens): " << config.default_max_tokens << "\n";
        std::cout << "  Default Temperature(default_temperature): " << config.default_temperature << "\n";
        std::cout << "  API Host(api_host): " << config.api_host << "\n";
        std::cout << "  API Port(api_port): " << config.api_port << "\n";
        std::cout << "  API Workers(api_workers): " << config.api_workers << "\n";
        std::cout << "  Download Provider(download_provider): " << config.download_provider << "\n";
    }
    
    static bool SetConfigValue(Config& config, const std::string& key, const std::string& value) {
        if (key == "download_provider") {
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
        } else if (key == "api_workers") {
            try {
                config.api_workers = std::stoi(value);
                return true;
            } catch (...) {
                return false;
            }
        }
        return false;
    }
    
    static std::string GetConfigHelp() {
        return R"(
Available configuration keys:
  download_provider  - Set default download provider (huggingface, modelscope, modelers)
  cache_dir         - Set cache directory path
  log_level         - Set log level (debug, info, warn, error)
  api_host          - Set API server host
  api_port          - Set API server port
  default_max_tokens - Set default maximum tokens for generation
  default_temperature - Set default temperature for generation
  api_workers       - Set number of API worker threads

Environment Variables (take precedence over config):
  MNN_DOWNLOAD_PROVIDER - Set default download provider
  MNN_CACHE_DIR        - Set cache directory path
  MNN_API_HOST         - Set API server host
  MNN_API_PORT         - Set API server port

Examples:
  mnncli config set download_provider modelscope
  mnncli config set cache_dir ~/.mnncli/cache
  mnncli config set api_port 8080
  
  # Using environment variables
  export MNN_DOWNLOAD_PROVIDER=modelscope
  export MNN_CACHE_DIR=~/.mnncli/cache
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
        MNN_PRINT("Prepare for tuning opt Begin\n");
        llm->tuning(OP_ENCODER_NUMBER, {1, 5, 10, 20, 30, 50, 100});
        MNN_PRINT("Prepare for tuning opt End\n");
    }
};

// Model management
class ModelManager {
public:
    static int ListLocalModels() {
        std::vector<std::string> model_names;
        std::string base_cache_dir = mnncli::FileUtils::GetBaseCacheDir();
        
        // List models from base cache directory
        int result = list_local_models(base_cache_dir, model_names);
        if (result != 0) {
            UserInterface::ShowError("Failed to list local models from base directory");
            return result;
        }
        
        // List models from modelscope subdirectory
        std::string modelscope_dir = base_cache_dir + "/modelscope";
        std::vector<std::string> modelscope_models;
        if (list_local_models(modelscope_dir, modelscope_models) == 0) {
            for (auto& name : modelscope_models) {
                model_names.emplace_back("modelscope/" + name);
            }
        }
        
        // List models from modelers subdirectory
        std::string modelers_dir = base_cache_dir + "/modelers";
        std::vector<std::string> modelers_models;
        if (list_local_models(modelers_dir, modelers_models) == 0) {
            for (auto& name : modelers_models) {
                model_names.emplace_back("modelers/" + name);
            }
        }
        
        if (!model_names.empty()) {
            std::cout << "Local models:\n";
            for (auto& name : model_names) {
                std::cout << "  📁 " << name << "\n";
            }
        } else {
            std::cout << "No local models found.\n";
            std::cout << "Use 'mnncli model search <keyword>' to search remote models\n";
            std::cout << "Use 'mnncli model download <name>' to download models\n";
        }
        return 0;
    }
    
    static int SearchRemoteModels(const std::string& keyword, bool verbose = false) {
        try {
            // Get cache directory from config or use default
            auto config = ConfigManager::LoadDefaultConfig();
            std::string cache_dir = config.cache_dir;
            if (cache_dir.empty()) {
                cache_dir = mnncli::FileUtils::GetBaseCacheDir() + "/.mnnmodels";
            }
            
            // Get current download provider from config
            std::string current_provider = config.download_provider;
            if (current_provider.empty()) {
                current_provider = "HuggingFace"; // Default provider
            }
            
            // Create ModelRepository instance
            auto& model_repo = mnncli::ModelRepository::getInstance(cache_dir);
            model_repo.setDownloadProvider(current_provider);
            
            // Show search information
            std::cout << "🔍 Searching for LLM models with keyword: '" << keyword << "'\n";
            std::cout << "   Current provider: " << current_provider << "\n";
            std::cout << "   Cache directory: " << cache_dir << "\n\n";
            
            LOG_DEBUG_TAG("Starting model search", "ModelSearch");
            LOG_DEBUG_TAG("Keyword: " + keyword, "ModelSearch");
            LOG_DEBUG_TAG("Provider: " + current_provider, "ModelSearch");
            LOG_DEBUG_TAG("Cache directory: " + cache_dir, "ModelSearch");
            
            // Perform search
            auto searchResults = model_repo.searchModels(keyword);
            
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
                std::cout << std::left << std::setw(30) << "Model Name" 
                          << std::setw(15) << "Vendor" 
                          << std::setw(10) << "Size (GB)" 
                          << std::setw(20) << "Tags" 
                          << std::setw(50) << "Model ID" << "\n";
                std::cout << std::string(125, '-') << "\n";
                
                for (const auto& model : searchResults) {
                    // Format tags for display
                    std::string tags_str;
                    if (!model.tags.empty()) {
                        tags_str = model.tags[0]; // Show first tag
                        if (model.tags.size() > 1) {
                            tags_str += " (+" + std::to_string(model.tags.size() - 1) + ")";
                        }
                    }
                    
                    // Format size
                    std::string size_str = model.size_gb > 0 ? std::to_string(model.size_gb) + " GB" : "N/A";
                    
                    std::cout << std::left << std::setw(30) << model.modelName.substr(0, 29)
                              << std::setw(15) << model.vendor.substr(0, 14)
                              << std::setw(10) << size_str
                              << std::setw(20) << tags_str.substr(0, 19)
                              << std::setw(50) << model.modelId.substr(0, 49) << "\n";
                }
                
                std::cout << "\n💡 To download a model, use:\n";
                std::cout << "  mnncli model download <model_name>\n";
                std::cout << "  Example: mnncli model download " << searchResults[0].modelName << "\n";
            }
            
        } catch (const std::exception& e) {
            UserInterface::ShowError("Failed to search models: " + std::string(e.what()));
            return 1;
        }
        return 0;
    }
    
    static int DownloadModel(const std::string& model_name, bool verbose = false) {
        if (model_name.empty()) {
            UserInterface::ShowError("Model name is required", "Usage: mnncli model download <name>");
            return 1;
        }
        
        std::cout << "Downloading model: " << model_name << "\n";
        
        // Get current configuration
        auto config = ConfigManager::LoadDefaultConfig();
        
        // Show which download provider will be used
        std::cout << "Using download provider: " << config.download_provider << "\n";
        
        try {
            // Get cache directory from config or use default
            std::string cache_dir = config.cache_dir;
            if (cache_dir.empty()) {
                cache_dir = mnncli::FileUtils::GetBaseCacheDir() + "/.mnnmodels";
            }
            
            // Create download manager instance
            auto& download_manager = mnncli::ModelDownloadManager::getInstance(cache_dir);
            
            // Create CLI download listener for user feedback
            CLIDownloadListener cli_listener;
            download_manager.addListener(&cli_listener);
            
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
                        auto& model_repo = mnncli::ModelRepository::getInstance(cache_dir);
                        model_repo.setDownloadProvider("HuggingFace"); // Default to HuggingFace
                        
                        auto model_id_opt = model_repo.getModelIdForDownload(model_name, "HuggingFace");
                        if (model_id_opt) {
                            model_id = *model_id_opt;
                            source = "HuggingFace";
                            std::cout << "✓ Found model in repository: " << model_id << "\n";
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
                    auto& model_repo = mnncli::ModelRepository::getInstance(cache_dir);
                    model_repo.setDownloadProvider(source);
                    
                    auto model_id_opt = model_repo.getModelIdForDownload(model_name, source);
                    if (model_id_opt) {
                        model_id = *model_id_opt;
                        std::cout << "✓ Found model in repository: " << model_id << "\n";
                        
                        // Get model type for additional info
                        std::string model_type = model_repo.getModelType(model_id);
                        std::cout << "  Model type: " << model_type << "\n";
                    } else {
                        // Fallback to direct model name
                        model_id = model_name;
                        std::cout << "⚠ Model not found in repository, using direct name: " << model_id << "\n";
                    }
                } catch (const std::exception& e) {
                    LOG_DEBUG_TAG("Failed to use ModelRepository: " + std::string(e.what()), "ModelManager");
                    // Fallback to direct model name
                    model_id = model_name;
                }
            }
            
            // Show download information
            std::cout << "🌐 Downloading from " << source << "\n";
            std::cout << "   Target model: " << model_id << "\n";
            std::cout << "   Cache directory: " << cache_dir << "\n";
            
            // Start the download
            download_manager.startDownload(model_id, source, model_name);
            
            // Wait for download to complete or fail
            while (download_manager.isDownloading(model_id)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            
            // Check final status
            auto download_info = download_manager.getDownloadInfo(model_id);
            if (download_info.state == mnncli::DownloadState::COMPLETED) {
                auto downloaded_file = download_manager.getDownloadedFile(model_id);
                if (!downloaded_file.empty() && std::filesystem::exists(downloaded_file)) {
                    UserInterface::ShowSuccess("Model downloaded successfully!");
                    UserInterface::ShowInfo("Model saved to: " + downloaded_file.string());
                    return 0;
                }
            } else if (download_info.state == mnncli::DownloadState::FAILED) {
                UserInterface::ShowError("Download failed. Check the error messages above.");
                return 1;
            }
            
            // Remove listener before returning
            download_manager.removeListener(&cli_listener);
            
        } catch (const std::exception& e) {
            UserInterface::ShowError("Failed to download model: " + std::string(e.what()));
            return 1;
        }
        return 0;
    }
    
    static int DeleteModel(const std::string& model_name) {
        if (model_name.empty()) {
            UserInterface::ShowError("Model name is required", "Usage: mnncli model delete <name>");
            return 1;
        }
        
        std::cout << "Deleting model: " << model_name << "\n";
        try {
            std::string linker_path = mnncli::FileUtils::GetFolderLinkerPath(model_name);
            mnncli::FileUtils::RemoveFileIfExists(linker_path);
            
            std::string full_name = model_name;
            if (model_name.find("taobao-mnn") != 0) {
                full_name = "taobao-mnn/" + model_name;
            }
            
            std::string storage_path = mnncli::FileUtils::GetStorageFolderPath(full_name);
            mnncli::FileUtils::RemoveFileIfExists(storage_path);
            
            UserInterface::ShowSuccess("Model deleted successfully: " + model_name);
        } catch (const std::exception& e) {
            UserInterface::ShowError("Failed to delete model: " + std::string(e.what()));
            return 1;
        }
        return 0;
    }
    
private:
    static int list_local_models(const std::string& directory_path, std::vector<std::string>& model_names) {
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
                mnncli::LogUtils::setVerbose(true);
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
            if (cmd == "model") {
                return HandleModelCommand(argc, argv);
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
            } else if (cmd == "--help" || cmd == "-h") {
                PrintUsage();
                return 0;
            } else if (cmd == "--version" || cmd == "-v") {
                PrintVersion();
                return 0;
            } else {
                // Legacy command support for backward compatibility
                return HandleLegacyCommand(argc, argv);
            }
        } catch (const std::exception& e) {
            UserInterface::ShowError("Unexpected error: " + std::string(e.what()));
            return 1;
        }
        
        return 0;
    }
    
private:
    int HandleModelCommand(int argc, const char* argv[]) {
        if (argc < 3) {
            PrintModelUsage();
            return 1;
        }
        
        std::string subcmd = argv[2];
        
        if (subcmd == "list") {
            return ModelManager::ListLocalModels();
        } else if (subcmd == "search") {
            if (argc < 4) {
                UserInterface::ShowError("Search keyword required", "Usage: mnncli model search <keyword>");
                return 1;
            }
            return ModelManager::SearchRemoteModels(argv[3], verbose_);
        } else if (subcmd == "download") {
            if (argc < 4) {
                UserInterface::ShowError("Model name required", "Usage: mnncli model download <name>");
                return 1;
            }
            return ModelManager::DownloadModel(argv[3], verbose_);
        } else if (subcmd == "delete") {
            if (argc < 4) {
                UserInterface::ShowError("Model name required", "Usage: mnncli model delete <name>");
                return 1;
            }
            return ModelManager::DeleteModel(argv[3]);
        } else {
            PrintModelUsage();
            return 1;
        }
    }
    
    int HandleRunCommand(int argc, const char* argv[]) {
        if (argc < 3) {
            UserInterface::ShowError("Model name required", "Usage: mnncli run <model_name> [options]");
            PrintRunUsage();
            return 1;
        }
        
        std::string model_name;
        std::string config_path;
        std::string prompt;
        std::string prompt_file;
        
        // Parse options first to see if we have a config path
        for (int i = 2; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "-c" || arg == "--config") {
                if (++i >= argc) {
                    UserInterface::ShowError("Missing config path", "Usage: -c <config_path>");
                    PrintRunUsage();
                    return 1;
                }
                config_path = mnncli::FileUtils::ExpandTilde(argv[i]);
            }
        }
        
        // If no config path specified, require model name
        if (config_path.empty()) {
            if (argc < 3) {
                UserInterface::ShowError("Model name required", "Usage: mnncli run <model_name> [options]");
                PrintRunUsage();
                return 1;
            }
            model_name = argv[2];
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
        
        // Parse remaining options
        for (int i = 2; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "-p" || arg == "--prompt") {
                if (++i >= argc) {
                    UserInterface::ShowError("Missing prompt text", "Usage: -p <prompt_text>");
                    PrintRunUsage();
                    return 1;
                }
                prompt = argv[i];
            } else if (arg == "-f" || arg == "--file") {
                if (++i >= argc) {
                    UserInterface::ShowError("Missing prompt file", "Usage: -f <prompt_file>");
                    PrintRunUsage();
                    return 1;
                }
                prompt_file = argv[i];
            } else if (arg == "-c" || arg == "--config") {
                // Skip the config path that follows
                i++;
            }
        }
        
        if (config_path.empty()) {
            UserInterface::ShowError("Config path is empty", "Use -c to specify config path");
            PrintRunUsage();
            return 1;
        }
        
        std::cout << "Starting model: " << model_name << "\n";
        std::cout << "Config path: " << config_path << "\n";
        
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
            UserInterface::ShowError("Model name required", "Usage: mnncli serve <model_name> [options]");
            return 1;
        }
        
        std::string model_name = argv[2];
        std::string config_path = (fs::path(mnncli::FileUtils::GetBaseCacheDir()) / model_name / "config.json").string();
        std::string host = "127.0.0.1";
        int port = 8000;
        
        // Parse options
        for (int i = 3; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "-c" || arg == "--config") {
                if (++i >= argc) {
                    UserInterface::ShowError("Missing config path", "Usage: -c <config_path>");
                    return 1;
                }
                config_path = mnncli::FileUtils::ExpandTilde(argv[i]);
            } else if (arg == "--host") {
                if (++i >= argc) {
                    UserInterface::ShowError("Missing host", "Usage: --host <host>");
                    return 1;
                }
                host = argv[i];
            } else if (arg == "--port") {
                if (++i >= argc) {
                    UserInterface::ShowError("Missing port", "Usage: --port <port>");
                    return 1;
                }
                port = std::stoi(argv[i]);
            }
        }
        
        std::cout << "Starting API server for model: " << model_name << "\n";
        std::cout << "Host: " << host << ":" << port << "\n";
        
        mnncli::MnncliServer server;
        bool is_r1 = IsR1(config_path);
        auto llm = LLMManager::CreateLLM(config_path, !is_r1);
        server.Start(llm.get(), is_r1);
        
        return 0;
    }
    
    int HandleBenchmarkCommand(int argc, const char* argv[]) {
        if (argc < 3) {
            UserInterface::ShowError("Model name required", "Usage: mnncli benchmark <model_name> [options]");
            return 1;
        }
        
        std::string model_name = argv[2];
        std::string config_path = mnncli::FileUtils::GetConfigPath(model_name);
        
        // Parse options
        for (int i = 3; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "-c" || arg == "--config") {
                if (++i >= argc) {
                    UserInterface::ShowError("Missing config path", "Usage: -c <config_path>");
                    return 1;
                }
                config_path = argv[i];
            }
        }
        
        if (config_path.empty()) {
            UserInterface::ShowError("Config path is empty", "Use -c to specify config path");
            return 1;
        }
        
        std::cout << "Starting benchmark for model: " << model_name << "\n";
        
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
                UserInterface::ShowError("Missing key or value", "Usage: mnncli config set <key> <value>");
                return 1;
            }
            
            std::string key = argv[3];
            std::string value = argv[4];
            
            auto config = ConfigManager::LoadDefaultConfig();
            if (ConfigManager::SetConfigValue(config, key, value)) {
                if (ConfigManager::SaveConfig(config)) {
                    UserInterface::ShowSuccess("Configuration updated and saved: " + key + " = " + value);
                } else {
                    UserInterface::ShowSuccess("Configuration updated: " + key + " = " + value);
                    std::cout << "Warning: Configuration could not be saved to file.\n";
                }
            } else {
                UserInterface::ShowError("Invalid configuration key or value", 
                    "Use 'mnncli config help' to see available options");
                return 1;
            }
        } else if (subcmd == "reset") {
            UserInterface::ShowInfo("Config reset command not implemented yet");
        } else if (subcmd == "help") {
            std::cout << ConfigManager::GetConfigHelp();
        } else {
            UserInterface::ShowError("Unknown config subcommand", "Use: show, set, reset, or help");
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
        
        std::vector<std::string> model_names;
        if (list_local_models(config.cache_dir, model_names) == 0) {
            std::cout << model_names.size() << "\n";
        } else {
            std::cout << "Unknown\n";
        }
        
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
    
    int HandleLegacyCommand(int argc, const char* argv[]) {
        std::string cmd = argv[1];
        
        if (cmd == "list") {
            return ModelManager::ListLocalModels();
        } else if (cmd == "search") {
            if (argc < 3) {
                UserInterface::ShowError("Search keyword required", "Usage: mnncli search <keyword>");
                return 1;
            }
            return ModelManager::SearchRemoteModels(argv[2]);
        } else if (cmd == "download") {
            if (argc < 3) {
                UserInterface::ShowError("Model name required", "Usage: mnncli download <name>");
                return 1;
            }
            return ModelManager::DownloadModel(argv[2], verbose_);
        } else if (cmd == "delete") {
            if (argc < 3) {
                UserInterface::ShowError("Model name required", "Usage: mnncli delete <name>");
                return 1;
            }
            return ModelManager::DeleteModel(argv[2]);
        } else if (cmd == "run") {
            return HandleRunCommand(argc, argv);
        } else if (cmd == "serve") {
            return HandleServeCommand(argc, argv);
        } else if (cmd == "benchmark") {
            return HandleBenchmarkCommand(argc, argv);
        } else {
            PrintUsage();
            return 1;
        }
    }
    
    void PrintUsage() {
        std::cout << "MNN CLI - AI Model Command Line Interface\n\n";
        std::cout << "Usage: mnncli <command> [options]\n\n";
        std::cout << "Commands:\n";
        std::cout << "  model     Manage models (list, search, download, delete)\n";
        std::cout << "  run       Run model inference\n";
        std::cout << "  serve     Start API server\n";
        std::cout << "  benchmark Run performance benchmarks\n";
        std::cout << "  config    Manage configuration (show, set, reset, help)\n";
        std::cout << "  info      Show system information\n";
        std::cout << "\nGlobal Options:\n";
        std::cout << "  -v, --verbose  Enable verbose output for detailed debugging\n";
        std::cout << "  --help    Show this help message\n";
        std::cout << "  --version Show version information\n";
        std::cout << "\nExamples:\n";
        std::cout << "  mnncli model list                    # List local models\n";
        std::cout << "  mnncli model search qwen             # Search for Qwen models\n";
        std::cout << "  mnncli model download qwen-7b        # Download Qwen-7B model\n";
        std::cout << "  mnncli download qwen-7b -v           # Download with verbose output\n";
        std::cout << "  mnncli config set download_provider modelscope  # Set default provider\n";
        std::cout << "  mnncli config show                   # Show current configuration\n";
        std::cout << "  mnncli config help                   # Show configuration help\n";
        std::cout << "  mnncli run qwen-7b                  # Run Qwen-7B model\n";
        std::cout << "  mnncli serve qwen-7b --port 8000    # Start API server\n";
        std::cout << "  mnncli benchmark qwen-7b            # Run benchmark\n";
    }
    
    void PrintModelUsage() {
        std::cout << "Model Management Commands:\n";
        std::cout << "  mnncli model list                    # List local models\n";
        std::cout << "  mnncli model search <keyword>        # Search remote models\n";
        std::cout << "  mnncli model download <name>         # Download model\n";
        std::cout << "  mnncli model delete <name>           # Delete model\n";
    }
    
    void PrintRunUsage() {
        std::cout << "Run Command Usage:\n";
        std::cout << "  mnncli run <model_name> [options]\n\n";
        std::cout << "Options:\n";
        std::cout << "  -c, --config <config_path>           # Specify custom config file path\n";
        std::cout << "  -p, --prompt <prompt_text>           # Provide prompt text directly\n";
        std::cout << "  -f, --file <prompt_file>             # Read prompts from file\n";
        std::cout << "  -v, --verbose                        # Enable verbose output\n\n";
        std::cout << "Examples:\n";
        std::cout << "  mnncli run qwen-7b                   # Run with default config\n";
        std::cout << "  mnncli run qwen-7b -p \"Hello\"        # Run with prompt\n";
        std::cout << "  mnncli run qwen-7b -f prompts.txt    # Run with prompt file\n";
        std::cout << "  mnncli run qwen-7b -c custom.json    # Run with custom config\n";
        std::cout << "  mnncli run qwen-7b -p \"Hello\" -v     # Run with prompt and verbose\n";
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
    
    static int list_local_models(const std::string& directory_path, std::vector<std::string>& model_names) {
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
    
    bool verbose_;
};

int main(int argc, const char* argv[]) {
    UserInterface::ShowWelcome();
    
    CommandLineInterface cli;
    return cli.Run(argc, argv);
}

