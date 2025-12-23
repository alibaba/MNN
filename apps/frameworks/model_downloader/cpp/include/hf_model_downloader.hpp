//
// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#pragma once

#include "model_repo_downloader.hpp"
#include "model_file_downloader.hpp"
#include "hf_api_client.hpp"
#include "hf_file_metadata.hpp"
#include "hf_file_metadata_utils.hpp"
#include "httplib.h"
#include <memory>
#include <vector>
#include <unordered_map>

namespace mnn::downloader {

// HuggingFace model downloader class
class HfModelDownloader : public ModelRepoDownloader {
public:
    explicit HfModelDownloader(const std::string& cache_root_path);
    ~HfModelDownloader() override = default;
    
    // Core download methods
    void Download(const std::string& model_id) override;
    void Pause(const std::string& model_id) override;
    void Resume(const std::string& model_id) override;
    
    // Repository management
    std::filesystem::path GetDownloadPath(const std::string& model_id) override;
    bool DeleteRepo(const std::string& model_id) override;
    int64_t GetRepoSize(const std::string& model_id) override;
    bool CheckUpdate(const std::string& model_id) override;
    
    // Set HuggingFace API client
    void setHfApiClient(std::shared_ptr<HfApiClient> client) { hf_api_client_ = client; }

private:
    // Download HuggingFace repository
    void downloadHfRepo(const RepoInfo& repo_info);
    
    // Inner download implementation
    void downloadHfRepoInner(const RepoInfo& repo_info);
    
    // Collect download tasks for all files in the repo (simplified for direct storage)
    std::vector<FileDownloadTask> collectTaskList(
        const std::filesystem::path& model_folder,
        const RepoInfo& repo_info,
        int64_t& total_size,
        int64_t& downloaded_size);
    
    // Request metadata for all files
    std::vector<HfFileMetadata> requestMetadataList(const RepoInfo& repo_info);
    
    // Get metadata for a single file
    HfFileMetadata getFileMetadata(const std::string& url, std::string& error_info);
    
    // Download a single file with metadata
    bool downloadFile(const std::string& url, const std::filesystem::path& destination_path,
                     const HfFileMetadata& metadata, const std::string& file_name,
                     std::string& error_info);
    
    // Get HuggingFace API client
    std::shared_ptr<HfApiClient> getHfApiClient();
    
    // Get HuggingFace model ID (remove prefix if present)
    std::string getHfModelId(const std::string& model_id);
    
    // Get cache path root for HuggingFace
    static std::string getCachePathRoot(const std::string& model_download_path_root);
    
    // Get model path
    static std::filesystem::path getModelPath(const std::string& cache_root_path, const std::string& model_id);

private:
    std::shared_ptr<HfApiClient> hf_api_client_;
    // HTTP client for metadata requests
    std::shared_ptr<httplib::SSLClient> metadata_client_;
    
    // Store original model_id for notifications
    std::string original_model_id_;
    
    // Constants
    static constexpr const char* HOST_DEFAULT = "huggingface.co";
    static constexpr int CONNECT_TIMEOUT_SECONDS = 30;
};

} // namespace mnn::downloader
