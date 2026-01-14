//
// Created by ruoyi.sjd on 2024/12/18.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <filesystem>
#include "ml_api_client.hpp"
#include "file_utils.hpp"
#include "model_repo_downloader.hpp"

namespace mnn::downloader {

// Modelers model downloader class
class MlModelDownloader : public ModelRepoDownloader {
public:
    explicit MlModelDownloader(const std::string& cache_root_path);
    
    // Implement virtual methods from ModelRepoDownloader
    void Download(const std::string& model_id) override;
    void Pause(const std::string& model_id) override;
    void Resume(const std::string& model_id) override;
    
    // Repository management
    std::filesystem::path GetDownloadPath(const std::string& model_id) override;
    bool DeleteRepo(const std::string& model_id) override;
    int64_t GetRepoSize(const std::string& model_id) override;
    bool CheckUpdate(const std::string& model_id) override;
    
    // Legacy methods for backward compatibility
    bool DownloadModel(const std::string& model_id, std::string& error_info, bool verbose = false);
    int64_t GetRepoSizeWithError(const std::string& model_id, std::string& error_info);
    bool CheckUpdateWithError(const std::string& model_id, std::string& error_info);

private:
    // Download the Modelers repository
    bool DownloadMlRepo(const std::string& model_id, std::string& error_info);
    
    // Inner download implementation
    bool DownloadMlRepoInner(const std::string& model_id, const std::string& modelers_id, 
                            const MlRepoInfo& ml_repo_info, std::string& error_info);
    
    // Collect download tasks for Modelers files
    std::vector<std::pair<std::string, std::filesystem::path>> CollectMlTaskList(
        const std::string& modelers_id,
        const std::filesystem::path& storage_folder,
        const std::filesystem::path& parent_pointer_path,
        const MlRepoInfo& ml_repo_info,
        int64_t& total_size,
        int64_t& downloaded_size);
    
    // Download a single file
    bool DownloadFile(const std::string& url, const std::filesystem::path& destination_path, 
                     int64_t expected_size, const std::string& file_name, const std::string& model_id, std::string& error_info);
    
    // Internal delete repo implementation
    bool DeleteRepoImpl(const std::string& model_id);
    
    // Get cache path root for Modelers
    static std::string GetCachePathRoot(const std::string& model_download_path_root);
    
    // Get model path
    static std::filesystem::path GetModelPath(const std::string& modelers_download_path_root, const std::string& model_id);

    // Show beautiful progress bar
    void ShowProgressBar(double progress, int64_t downloaded_size, int64_t total_size);
    
    // Show file download progress
    void ShowFileDownloadProgress(const std::string& file_name, double percentage, int64_t downloaded_size, int64_t total_size);

private:
    std::unique_ptr<MlApiClient> ml_api_client_;
    bool verbose_;
};

} // namespace mnn::downloader
