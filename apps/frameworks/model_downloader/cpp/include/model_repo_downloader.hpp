//
// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#pragma once

#include <string>
#include <filesystem>
#include <functional>
#include <unordered_map>
#include <vector>

namespace mnn::downloader {

// Download state enum
enum class DownloadState {
    NOT_START,
    DOWNLOADING,
    PAUSED,
    COMPLETED,
    FAILED
};

// Download progress callback
struct DownloadProgress {
    std::string model_id;
    std::string stage;
    std::string current_file;
    int64_t saved_size;
    int64_t total_size;
    double progress;
    DownloadState state;
};

// Download listener interface
class DownloadListener {
public:
    virtual ~DownloadListener() = default;
    
    virtual void OnDownloadStart(const std::string& model_id) {}
    virtual void OnDownloadProgress(const std::string& model_id, const DownloadProgress& progress) {}
    virtual void OnDownloadFinished(const std::string& model_id, const std::string& path) {}
    virtual void OnDownloadFailed(const std::string& model_id, const std::string& error) {}
    virtual void OnDownloadPaused(const std::string& model_id) {}
    virtual void OnDownloadTaskAdded() {}
    virtual void OnDownloadTaskRemoved() {}
    virtual void OnRepoInfo(const std::string& model_id, int64_t last_modified, int64_t repo_size) {}
    
    // Get class type name (replaces RTTI usage)
    virtual std::string GetClassTypeName() const { return "DownloadListener"; }
};

// Base class for model repository downloaders
class ModelRepoDownloader {
public:
    explicit ModelRepoDownloader(const std::string& cache_root_path);
    virtual ~ModelRepoDownloader() = default;
    
    // Set download listener
    void SetListener(DownloadListener* listener) { listener_ = listener; }
    
    // Core download methods
    virtual void Download(const std::string& model_id) = 0;
    virtual void Pause(const std::string& model_id) = 0;
    virtual void Resume(const std::string& model_id) = 0;
    
    // Repository management
    virtual std::filesystem::path GetDownloadPath(const std::string& model_id) = 0;
    virtual bool DeleteRepo(const std::string& model_id) = 0;
    virtual int64_t GetRepoSize(const std::string& model_id) = 0;
    virtual bool CheckUpdate(const std::string& model_id) = 0;
    
    // Utility methods
    std::string GetCacheRootPath() const { return cache_root_path_; }
    
    // Common utility methods for download progress display (public static)
    // Format file size with smart unit selection (KB for <1MB, MB otherwise)
    static std::string FormatFileSizeInfo(int64_t downloaded_bytes, int64_t total_bytes);
    
    // Extract filename from path for cleaner display
    static std::string ExtractFileName(const std::string& file_path);
    
protected:
    // Completion markers and manifest helpers
    // Markers and manifest are stored under model_folder/.mnncli/
    bool IsDownloadComplete(const std::filesystem::path& model_folder) const;
    void MarkDownloading(const std::filesystem::path& model_folder) const;
    bool MarkComplete(
        const std::filesystem::path& model_folder,
        const std::vector<std::pair<std::string, int64_t>>& manifest_entries
    ) const;
    void ClearMarkers(const std::filesystem::path& model_folder) const;
    bool ValidateFilesBySize(
        const std::filesystem::path& model_folder,
        const std::vector<std::pair<std::string, int64_t>>& manifest_entries
    ) const;

    // Helper methods for subclasses
    void NotifyDownloadStart(const std::string& model_id);
    void NotifyDownloadProgress(const std::string& model_id, const std::string& stage, 
                              const std::string& current_file, int64_t saved_size, int64_t total_size);
    void NotifyDownloadFinished(const std::string& model_id, const std::string& path);
    void NotifyDownloadFailed(const std::string& model_id, const std::string& error);
    void NotifyDownloadPaused(const std::string& model_id);
    void NotifyDownloadTaskAdded();
    void NotifyDownloadTaskRemoved();
    void NotifyRepoInfo(const std::string& model_id, int64_t last_modified, int64_t repo_size);
    
    // Calculate progress percentage
    double CalculateProgress(int64_t saved_size, int64_t total_size) const;
    
    // Update download state
    void UpdateDownloadState(const std::string& model_id, DownloadState state);
    
    // Paused models tracking
    bool IsPaused(const std::string& model_id) const;
    void AddPausedModel(const std::string& model_id);
    void RemovePausedModel(const std::string& model_id);

protected:
    std::string cache_root_path_;
    DownloadListener* listener_;
    std::vector<std::string> paused_models_;
    std::unordered_map<std::string, DownloadState> download_states_;
};

} // namespace mnn::downloader
