//
// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#pragma once

#include "model_repo_downloader.hpp"
#include "hf_model_downloader.hpp"
#include "ms_model_downloader.hpp"
#include "ml_model_downloader.hpp"
#include <memory>
#include <unordered_map>
#include <vector>
#include <string>

namespace mnncli {

// Model source types
enum class ModelSource {
    HUGGING_FACE,
    MODEL_SCOPE,
    MODELERS,
    UNKNOWN
};

// Model source utilities
class ModelSources {
public:
    static constexpr const char* SOURCE_HUGGING_FACE = "HuggingFace";
    static constexpr const char* SOURCE_MODEL_SCOPE = "ModelScope";
    static constexpr const char* SOURCE_MODELERS = "Modelers";
    
    // Convert string to ModelSource
    static ModelSource fromString(const std::string& source_str);
    
    // Convert ModelSource to string
    static std::string toString(ModelSource source);
    
    // Extract source from model ID
    static ModelSource getSource(const std::string& model_id);
    
    // Split model ID into source and path
    static std::pair<std::string, std::string> splitSource(const std::string& model_id);
    
    // Get model name from model ID
    static std::string getModelName(const std::string& model_id);
};

// Main download manager class
class ModelDownloadManager : public DownloadListener {
public:
    explicit ModelDownloadManager(const std::string& cache_root_path);
    ~ModelDownloadManager() = default;
    
    // Singleton access
    static ModelDownloadManager& getInstance(const std::string& cache_root_path = "");
    
    // Listener management
    void addListener(DownloadListener* listener);
    void removeListener(DownloadListener* listener);
    
    // Download operations
    void startDownload(const std::string& model_id);
    void startDownload(const std::string& model_id, const std::string& source);
    void startDownload(const std::string& model_id, const std::string& source, const std::string& model_name);
    
    // Download control
    void pauseDownload(const std::string& model_id);
    void resumeDownload(const std::string& model_id);
    void cancelDownload(const std::string& model_id);
    
    // Repository management
    std::filesystem::path getDownloadedFile(const std::string& model_id);
    bool deleteRepo(const std::string& model_id);
    int64_t getRepoSize(const std::string& model_id);
    bool checkUpdate(const std::string& model_id);
    
    // Download information
    DownloadProgress getDownloadInfo(const std::string& model_id);
    std::vector<std::string> getActiveDownloads() const;
    bool isDownloading(const std::string& model_id) const;
    
    // Utility methods
    std::string getCacheRootPath() const { return cache_root_path_; }
    
    // DownloadListener implementation
    void onDownloadFinished(const std::string& model_id, const std::string& path) override;
    void onDownloadFailed(const std::string& model_id, const std::string& error) override;
    void onDownloadPaused(const std::string& model_id) override;

private:
    // Get appropriate downloader for source
    ModelRepoDownloader* getDownloaderForSource(ModelSource source);
    ModelRepoDownloader* getDownloaderForSource(const std::string& source_str);
    
    // Download state management
    void updateDownloadState(const std::string& model_id, DownloadState state);
    void updateDownloadProgress(const std::string& model_id, const std::string& stage,
                              const std::string& current_file, int64_t saved_size, int64_t total_size);
    
    // Active download tracking
    void addActiveDownload(const std::string& model_id, const std::string& display_name);
    void removeActiveDownload(const std::string& model_id);
    
    // Calculate real download size
    int64_t getRealDownloadSize(const std::string& model_id);
    
    // Calculate download speed
    void calculateDownloadSpeed(const std::string& model_id, int64_t current_download_size);
    
    // Setup download callbacks
    void setupDownloadCallbacks();

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
    static constexpr const char* TAG = "ModelDownloadManager";
    static constexpr int64_t SPEED_UPDATE_INTERVAL_MS = 1000;
};

} // namespace mnncli
