//
// Created by ruoyi.sjd on 2024/12/19.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#include "file_utils.hpp"
#include "cli_config_manager.hpp"

namespace mnn::downloader {

// This file contains FileUtils methods that depend on ConfigManager
// to avoid pulling json dependencies into the main file_utils.cpp

std::string FileUtils::GetConfigPath(const std::string& model_id) {
    // Get current configuration
    auto& config_mgr = mnncli::ConfigManager::GetInstance();
    auto config = config_mgr.LoadConfig();
    
    mnn::downloader::Config downloader_config; 
    // Manual mapping or just pass what's needed. 
    // Actually FileUtils::GetConfigPath(model_id, config) takes mnn::downloader::Config
    // We need to convert mnncli::Config to mnn::downloader::Config if explicit conversion is missing.
    // Or just use kCachePath directly? 
    // The library implementation uses GetConfigPath(model_id, config).
    // Let's implement it using kCachePath/default if possible, 
    // OR map the config.
    
    // Simplest: just map the relevant fields
    downloader_config.cache_dir = config.cache_dir; // Assuming compatible
    downloader_config.download_provider = config.download_provider;
    
    // Use the overloaded version with config parameter
    return GetConfigPath(model_id, downloader_config);
}

} // namespace mnn::downloader
