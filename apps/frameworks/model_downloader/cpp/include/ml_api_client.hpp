//
// Created by ruoyi.sjd on 2024/12/18.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#pragma once

#include <string>
#include <vector>
#include <functional>
#include <thread>
#include <chrono>
#include <cmath>
#include "file_utils.hpp"

namespace mnn::downloader {

// Modelers file information structure
struct MlFileInfo {
    std::string path;      // File path within the repository
    std::string type;      // File type (file, tree, etc.)
    int64_t size;          // File size in bytes
    std::string sha256;    // File SHA256 hash
};

// Modelers repository information structure
struct MlRepoInfo {
    std::string model_id;  // Model identifier
    std::string revision;  // Repository revision
    std::vector<MlFileInfo> files;  // List of files in the repository
};

// Modelers API client
class MlApiClient {
public:
    MlApiClient();
    
    // Get repository information from Modelers
    MlRepoInfo GetRepoInfo(const std::string& repo_name, const std::string& revision, std::string& error_info);
    
    // Download the entire repository
    void DownloadRepo(const MlRepoInfo& repo_info);

private:
    // Perform HTTP request with retry logic
    bool PerformRequestWithRetry(std::function<bool()> request_func, int max_attempts, int retry_delay_seconds);
    
    std::string host_;                    // API host (modelers.cn)
    std::string cache_path_;              // Cache directory path
    int max_attempts_;                    // Maximum retry attempts
    int retry_delay_seconds_;             // Delay between retries
    MlRepoInfo repo_info_;                // Current repository info
};

} // namespace mnn::downloader
