//
//  cli_config_manager.hpp
//
//  Configuration management for mnncli
//

#pragma once

#include <string>
#include "mnncli_config.hpp"

namespace mnncli {

class ConfigManager {
public:
    // Singleton pattern
    static ConfigManager& GetInstance();
    
    // Configuration operations
    mnncli::Config LoadConfig();
    bool SaveConfig(const mnncli::Config& config);
    void ShowConfig(const mnncli::Config& config);
    bool SetConfigValue(const std::string& key, const std::string& value);
    bool ResetConfig();
    std::string GetConfigHelp() const;
    
    // Getters
    std::string GetCacheDir() const;
    std::string GetBaseCacheDir() const;
    std::string GetDownloadProvider() const;
    std::string GetDefaultModel() const;
    int GetApiPort() const;
    std::string GetApiHost() const;
    const char* GetModelSource() const;

private:
    ConfigManager() = default;
    ~ConfigManager() = default;
    ConfigManager(const ConfigManager&) = delete;
    ConfigManager& operator=(const ConfigManager&) = delete;
    
    std::string GetConfigFilePath() const;
    mnncli::Config LoadFromFile();
    mnncli::Config LoadFromEnvironment();
    mnncli::Config GetDefaultConfig();
    
    mnncli::Config current_config_;
};

} // namespace mnncli

