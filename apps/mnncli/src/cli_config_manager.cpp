//
//  cli_config_manager.cpp
//
//  Configuration management implementation
//

#include "cli_config_manager.hpp"
#include "file_utils.hpp"
#include "model_sources.hpp"
#include <filesystem>
#include <fstream>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include "json.hpp"

namespace fs = std::filesystem;

namespace mnncli {

using namespace mnn::downloader;

ConfigManager& ConfigManager::GetInstance() {
    static ConfigManager instance;
    return instance;
}

Config ConfigManager::LoadConfig() {
    // 1. Get default config as base
    current_config_ = GetDefaultConfig();

    // 2. If environment has valid values, override with environment
    current_config_ = LoadFromEnvironment();
    
    // 3. If file has valid values, override with file
    current_config_ = LoadFromFile();
    
    return current_config_;
}

bool ConfigManager::SaveConfig(const Config& config) {
    try {
        std::string config_path = GetConfigFilePath();
        std::string config_dir = fs::path(config_path).parent_path().string();
        
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
            current_config_ = config;
            return true;
        }
    } catch (...) {
        std::cerr << "Failed to save configuration file." << std::endl;
    }
    return false;
}

void ConfigManager::ShowConfig(const Config& config) {
    std::cout << "Configuration:\n";
    std::cout << "  Default Model(default_model): " 
              << (config.default_model.empty() ? "Not set" : config.default_model) << "\n";
    std::cout << "  Cache Directory(cache_dir): " << config.cache_dir << "\n";
    std::cout << "  Log Level(log_level): " << config.log_level << "\n";
    std::cout << "  Default Max Tokens(default_max_tokens): " << config.default_max_tokens << "\n";
    std::cout << "  Default Temperature(default_temperature): " << config.default_temperature << "\n";
    std::cout << "  API Host(api_host): " << config.api_host << "\n";
    std::cout << "  API Port(api_port): " << config.api_port << "\n";
    std::cout << "  Download Provider(download_provider): " << config.download_provider << "\n";
}

bool ConfigManager::SetConfigValue(const std::string& key, const std::string& value) {
    Config& config = current_config_;
    
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
        }
        return false;
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

std::string ConfigManager::GetConfigHelp() const {
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

std::string ConfigManager::GetCacheDir() const {
    return current_config_.cache_dir.empty() ? mnncli::kCachePath : current_config_.cache_dir;
}

std::string ConfigManager::GetBaseCacheDir() const {
    std::string cache_dir = GetCacheDir();
    
#ifdef __ANDROID__
    // On Android, use current working directory as base for relative path
    std::string expanded_cache_dir = FileUtils::ExpandTilde(cache_dir);
    
    if (expanded_cache_dir.empty()) {
        fprintf(stderr, "Unable to get cache directory path.");
        return ""; // Handle error appropriately in your application
    }
    
    // Get current working directory and append the cache path
    std::filesystem::path cwd = std::filesystem::current_path();
    std::filesystem::path cache_path = cwd / expanded_cache_dir;
#else
    // On other platforms, use the home directory
    std::string expanded_cache_dir = FileUtils::ExpandTilde(cache_dir);

    if (expanded_cache_dir.empty()) {
        fprintf(stderr, "Unable to get home directory.");
        return ""; // Handle error appropriately in your application
    }

    std::filesystem::path cache_path(expanded_cache_dir);
#endif

    if (!fs::exists(cache_path)) {
        fs::create_directories(cache_path);
    }

    return cache_path.string();
}

std::string ConfigManager::GetDownloadProvider() const {
    return current_config_.download_provider.empty() ? "huggingface" : current_config_.download_provider;
}

std::string ConfigManager::GetDefaultModel() const {
    return current_config_.default_model;
}

int ConfigManager::GetApiPort() const {
    return current_config_.api_port;
}

std::string ConfigManager::GetApiHost() const {
    return current_config_.api_host;
}

const char* ConfigManager::GetModelSource() const {
    std::string provider = GetDownloadProvider();
    std::string lower_provider = provider;
    std::transform(lower_provider.begin(), lower_provider.end(), lower_provider.begin(), ::tolower);
    
    if (lower_provider == "huggingface" || lower_provider == "hf") {
        return ModelSources::SOURCE_HUGGING_FACE;
    } else if (lower_provider == "modelscope" || lower_provider == "ms") {
        return ModelSources::SOURCE_MODEL_SCOPE;
    } else if (lower_provider == "modelers") {
        return ModelSources::SOURCE_MODELERS;
    } else {
        // If provider is already in correct format (e.g., "ModelScope"), check against known sources
        if (provider == ModelSources::SOURCE_HUGGING_FACE) {
            return ModelSources::SOURCE_HUGGING_FACE;
        } else if (provider == ModelSources::SOURCE_MODEL_SCOPE) {
            return ModelSources::SOURCE_MODEL_SCOPE;
        } else if (provider == ModelSources::SOURCE_MODELERS) {
            return ModelSources::SOURCE_MODELERS;
        }
        // Default to HuggingFace if unknown
        return ModelSources::SOURCE_HUGGING_FACE;
    }
}

std::string ConfigManager::GetConfigFilePath() const {
    // Use GetCacheDir instead of GetBaseCacheDir to avoid recursion
    std::string config_dir_str = GetCacheDir();
    std::string expanded_config_dir = FileUtils::ExpandTilde(config_dir_str);
    fs::path config_dir(expanded_config_dir);
    return (config_dir / "mnncli_config.json").string();
}

Config ConfigManager::LoadFromFile() {
    std::string config_path = GetConfigFilePath();
    
    if (fs::exists(config_path)) {
        try {
            std::ifstream file(config_path);
            if (file.is_open()) {
                nlohmann::json j;
                file >> j;
                
                // Only override if value exists in file
                if (j.contains("default_model") && !j["default_model"].get<std::string>().empty()) {
                    current_config_.default_model = j["default_model"].get<std::string>();
                }
                if (j.contains("cache_dir") && !j["cache_dir"].get<std::string>().empty()) {
                    current_config_.cache_dir = j["cache_dir"].get<std::string>();
                }
                if (j.contains("log_level") && !j["log_level"].get<std::string>().empty()) {
                    current_config_.log_level = j["log_level"].get<std::string>();
                }
                if (j.contains("default_max_tokens")) {
                    current_config_.default_max_tokens = j["default_max_tokens"].get<int>();
                }
                if (j.contains("default_temperature")) {
                    current_config_.default_temperature = j["default_temperature"].get<float>();
                }
                if (j.contains("api_host") && !j["api_host"].get<std::string>().empty()) {
                    current_config_.api_host = j["api_host"].get<std::string>();
                }
                if (j.contains("api_port")) {
                    current_config_.api_port = j["api_port"].get<int>();
                }
                if (j.contains("download_provider") && !j["download_provider"].get<std::string>().empty()) {
                    current_config_.download_provider = j["download_provider"].get<std::string>();
                }
                file.close();
            }
        } catch (...) {
            // Fall back to current config if file parsing fails
        }
    }
    return current_config_;
}

Config ConfigManager::LoadFromEnvironment() {
    
    // Override with environment variables if they exist and are valid
    if (const char* env_model = std::getenv("MNN_DEFAULT_MODEL")) {
        current_config_.default_model = env_model;
    }
    if (const char* env_provider = std::getenv("MNN_DOWNLOAD_PROVIDER")) {
        current_config_.download_provider = env_provider;
    }
    if (const char* env_cache = std::getenv("MNN_CACHE_DIR")) {
        current_config_.cache_dir = env_cache;
    }
    if (const char* env_host = std::getenv("MNN_API_HOST")) {
        current_config_.api_host = env_host;
    }
    if (const char* env_port = std::getenv("MNN_API_PORT")) {
        try {
            current_config_.api_port = std::stoi(env_port);
        } catch (...) {
            // Keep current value if conversion fails
        }
    }
    return current_config_;
}

Config ConfigManager::GetDefaultConfig() {
    return {
        .default_model = "",
        .cache_dir = mnncli::kConfigPath,
        .log_level = "info",
        .default_max_tokens = 1000,
        .default_temperature = 0.7f,
        .api_host = "127.0.0.1",
        .api_port = 8000,
        .download_provider = "modelscope"
    };
}

bool ConfigManager::ResetConfig() {
    try {
        std::string config_file = GetConfigFilePath();
        
        // Delete the config file if it exists
        if (fs::exists(config_file)) {
            fs::remove(config_file);
        }
        
        // Reset in-memory config to default
        current_config_ = GetDefaultConfig();
        
        // Reload to apply environment variables if any
        LoadConfig();
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to reset configuration: " << e.what() << std::endl;
        return false;
    }
}

} // namespace mnncli

