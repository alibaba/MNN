//
// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#pragma once

#include <string>
#include <filesystem>
#include <functional>
#include <unordered_map>

namespace mnncli {

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
    
    virtual void onDownloadStart(const std::string& model_id) {}
    virtual void onDownloadProgress(const std::string& model_id, const DownloadProgress& progress) {}
    virtual void onDownloadFinished(const std::string& model_id, const std::string& path) {}
    virtual void onDownloadFailed(const std::string& model_id, const std::string& error) {}
    virtual void onDownloadPaused(const std::string& model_id) {}
    virtual void onDownloadTaskAdded() {}
    virtual void onDownloadTaskRemoved() {}
    virtual void onRepoInfo(const std::string& model_id, int64_t last_modified, int64_t repo_size) {}
    
    // Get class type name (replaces RTTI usage)
    virtual std::string getClassTypeName() const { return "DownloadListener"; }
};

// Base class for model repository downloaders
class ModelRepoDownloader {
public:
    explicit ModelRepoDownloader(const std::string& cache_root_path);
    virtual ~ModelRepoDownloader() = default;
    
    // Set download listener
    void setListener(DownloadListener* listener) { listener_ = listener; }
    
    // Core download methods
    virtual void download(const std::string& model_id) = 0;
    virtual void pause(const std::string& model_id) = 0;
    virtual void resume(const std::string& model_id) = 0;
    
    // Repository management
    virtual std::filesystem::path getDownloadPath(const std::string& model_id) = 0;
    virtual bool deleteRepo(const std::string& model_id) = 0;
    virtual int64_t getRepoSize(const std::string& model_id) = 0;
    virtual bool checkUpdate(const std::string& model_id) = 0;
    
    // Utility methods
    std::string getCacheRootPath() const { return cache_root_path_; }
    
protected:
    // Helper methods for subclasses
    void notifyDownloadStart(const std::string& model_id);
    void notifyDownloadProgress(const std::string& model_id, const std::string& stage, 
                              const std::string& current_file, int64_t saved_size, int64_t total_size);
    void notifyDownloadFinished(const std::string& model_id, const std::string& path);
    void notifyDownloadFailed(const std::string& model_id, const std::string& error);
    void notifyDownloadPaused(const std::string& model_id);
    void notifyDownloadTaskAdded();
    void notifyDownloadTaskRemoved();
    void notifyRepoInfo(const std::string& model_id, int64_t last_modified, int64_t repo_size);
    
    // Calculate progress percentage
    double calculateProgress(int64_t saved_size, int64_t total_size) const;
    
    // Update download state
    void updateDownloadState(const std::string& model_id, DownloadState state);
    
    // Paused models tracking
    bool isPaused(const std::string& model_id) const;
    void addPausedModel(const std::string& model_id);
    void removePausedModel(const std::string& model_id);

protected:
    std::string cache_root_path_;
    DownloadListener* listener_;
    std::vector<std::string> paused_models_;
    std::unordered_map<std::string, DownloadState> download_states_;
};

} // namespace mnncli
