//
// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#pragma once

#include "model_repo_downloader.hpp"
#include "hf_model_downloader.hpp"
#include "ms_model_downloader.hpp"
#include "ml_model_downloader.hpp"
#include "model_sources.hpp"
#include <memory>
#include <unordered_map>
#include <vector>
#include <string>

namespace mnn::downloader {

// Main download manager class
class ModelDownloadManager : public DownloadListener {
public:
    explicit ModelDownloadManager(const std::string& cache_root_path);
    ~ModelDownloadManager() = default;
    
    // Singleton access
    static ModelDownloadManager& GetInstance(const std::string& cache_root_path = "");
    
    // Listener management
    void AddListener(DownloadListener* listener);
    void RemoveListener(DownloadListener* listener);
    
    // Download operations
    void StartDownload(const std::string& model_id);
    void StartDownload(const std::string& model_id, const std::string& source);
    void StartDownload(const std::string& model_id, const std::string& source, const std::string& model_name);
    
    // Download control
    void PauseDownload(const std::string& model_id);
    void ResumeDownload(const std::string& model_id);
    void CancelDownload(const std::string& model_id);
    
    // Repository management
    std::filesystem::path GetDownloadedFile(const std::string& model_id);
    bool DeleteRepo(const std::string& model_id);
    int64_t GetRepoSize(const std::string& model_id);
    bool CheckUpdate(const std::string& model_id);
    
    // Download information
    DownloadProgress GetDownloadInfo(const std::string& model_id);
    std::vector<std::string> GetActiveDownloads() const;
    bool IsDownloading(const std::string& model_id) const;
    
    // Utility methods
    std::string GetCacheRootPath() const { return cache_root_path_; }
    
    // DownloadListener implementation
    void OnDownloadStart(const std::string& model_id);
    void OnDownloadProgress(const std::string& model_id, const DownloadProgress& progress);
    void OnDownloadFinished(const std::string& model_id, const std::string& path);
    void OnDownloadFailed(const std::string& model_id, const std::string& error);
    void OnDownloadPaused(const std::string& model_id);

private:
    // Get appropriate downloader for source
    ModelRepoDownloader* GetDownloaderForSource(ModelSource source);
    ModelRepoDownloader* GetDownloaderForSource(const std::string& source_str);
    
    // Download state management
    void UpdateDownloadState(const std::string& model_id, DownloadState state);
    void UpdateDownloadProgress(const std::string& model_id, const std::string& stage,
                              const std::string& current_file, int64_t saved_size, int64_t total_size);
    
    // Active download tracking
    void AddActiveDownload(const std::string& model_id, const std::string& display_name);
    void RemoveActiveDownload(const std::string& model_id);
    
    // Calculate real download size
    int64_t GetRealDownloadSize(const std::string& model_id);
    
    // Calculate download speed
    void CalculateDownloadSpeed(const std::string& model_id, int64_t current_download_size);
    
    // Setup download callbacks
    void SetupDownloadCallbacks();

private:
    std::string cache_root_path_;
    bool verbose_;
    
    // Downloaders
    std::unique_ptr<HfModelDownloader> hf_downloader_;
    std::unique_ptr<MsModelDownloader> ms_downloader_;
    std::unique_ptr<MlModelDownloader> ml_downloader_;
    
    // Listeners
    std::vector<DownloadListener*> listeners_;
    
    // Download state tracking
    std::unordered_map<std::string, DownloadProgress> download_info_map_;
    std::unordered_map<std::string, std::string> active_download_names_;
    std::vector<std::string> active_downloads_;
    
    // Download speed tracking
    std::unordered_map<std::string, int64_t> last_download_sizes_;
    std::unordered_map<std::string, int64_t> last_log_times_;
    
    // Constants
    static constexpr const char* kTag = "ModelDownloadManager";
    static constexpr int64_t kSpeedUpdateIntervalMs = 1000;
};

} // namespace mnn::downloader
