//
// Created by ruoyi.sjd on 2024/12/19.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#include "file_utils.hpp"
#include "cli_config_manager.hpp"

namespace mnncli {

// This file contains FileUtils methods that depend on ConfigManager
// to avoid pulling json dependencies into the main file_utils.cpp

std::string FileUtils::GetConfigPath(const std::string& model_id) {
    // Get current configuration
    auto& config_mgr = mnncli::ConfigManager::GetInstance();
    auto config = config_mgr.LoadConfig();
    
    // Use the overloaded version with config parameter
    return GetConfigPath(model_id, config);
}

} // namespace mnncli

