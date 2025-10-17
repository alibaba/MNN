//
// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#pragma once

namespace mnncli {
#ifdef __ANDROID__
   // On Android, use a relative path in the mnncli directory to avoid filesystem permission issues
   const char* const kCachePath = ".mnnmodels";
#else
   // On other platforms, use the cache directory following XDG Base Directory Specification
   const char* const kCachePath = "~/.cache/mnncli/mnnmodels";
#endif
}

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
        std::string download_provider;  // "huggingface", "modelscope", or "modelers"
    };
    
    Config LoadDefaultConfig();
    bool SaveConfig(const Config& config);
    void ShowConfig(const Config& config);
    bool SetConfigValue(Config& config, const std::string& key, const std::string& value);
    std::string GetConfigHelp();
}