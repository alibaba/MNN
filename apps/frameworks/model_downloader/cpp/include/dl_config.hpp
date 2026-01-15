#pragma once

#include <string>

namespace mnn {
namespace downloader {

#ifdef __ANDROID__
    // On Android, use a relative path in the mnncli directory to avoid filesystem permission issues
    constexpr char kCachePath[] = ".mnnmodels";
    constexpr char kConfigPath[] = ".mnnmodels";
#else
    constexpr char kCachePath[] = "~/.cache/mnncli/mnnmodels";
    constexpr char kConfigPath[] = "~/.cache/mnncli/models";
#endif

    // Centralized configuration structure
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

} // namespace downloader
} // namespace mnn
