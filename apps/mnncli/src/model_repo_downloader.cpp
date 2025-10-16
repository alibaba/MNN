//
// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#include "model_repo_downloader.hpp"
#include <algorithm>
#include <iostream>

namespace mnncli {

ModelRepoDownloader::ModelRepoDownloader(const std::string& cache_root_path)
    : cache_root_path_(cache_root_path), listener_(nullptr) {
}

void ModelRepoDownloader::NotifyDownloadStart(const std::string& model_id) {
    if (listener_) {
        listener_->OnDownloadStart(model_id);
    }
    UpdateDownloadState(model_id, DownloadState::DOWNLOADING);
}

void ModelRepoDownloader::NotifyDownloadProgress(const std::string& model_id, const std::string& stage,
                                               const std::string& current_file, int64_t saved_size, int64_t total_size) {
    if (listener_) {
        DownloadProgress progress;
        progress.model_id = model_id;
        progress.stage = stage;
        progress.current_file = current_file;
        progress.saved_size = saved_size;
        progress.total_size = total_size;
        progress.progress = CalculateProgress(saved_size, total_size);
        progress.state = download_states_[model_id];
        
        listener_->OnDownloadProgress(model_id, progress);
    }
}

void ModelRepoDownloader::NotifyDownloadFinished(const std::string& model_id, const std::string& path) {
    if (listener_) {
        listener_->OnDownloadFinished(model_id, path);
    }
    UpdateDownloadState(model_id, DownloadState::COMPLETED);
}

void ModelRepoDownloader::NotifyDownloadFailed(const std::string& model_id, const std::string& error) {
    if (listener_) {
        listener_->OnDownloadFailed(model_id, error);
    }
    UpdateDownloadState(model_id, DownloadState::FAILED);
}

void ModelRepoDownloader::NotifyDownloadPaused(const std::string& model_id) {
    if (listener_) {
        listener_->OnDownloadPaused(model_id);
    }
    UpdateDownloadState(model_id, DownloadState::PAUSED);
}

void ModelRepoDownloader::NotifyDownloadTaskAdded() {
    if (listener_) {
        listener_->OnDownloadTaskAdded();
    }
}

void ModelRepoDownloader::NotifyDownloadTaskRemoved() {
    if (listener_) {
        listener_->OnDownloadTaskRemoved();
    }
}

void ModelRepoDownloader::NotifyRepoInfo(const std::string& model_id, int64_t last_modified, int64_t repo_size) {
    if (listener_) {
        listener_->OnRepoInfo(model_id, last_modified, repo_size);
    }
}

double ModelRepoDownloader::CalculateProgress(int64_t saved_size, int64_t total_size) const {
    if (total_size <= 0) return 0.0;
    return static_cast<double>(saved_size) / static_cast<double>(total_size);
}

void ModelRepoDownloader::UpdateDownloadState(const std::string& model_id, DownloadState state) {
    download_states_[model_id] = state;
}

bool ModelRepoDownloader::IsPaused(const std::string& model_id) const {
    return std::find(paused_models_.begin(), paused_models_.end(), model_id) != paused_models_.end();
}

void ModelRepoDownloader::AddPausedModel(const std::string& model_id) {
    if (!IsPaused(model_id)) {
        paused_models_.push_back(model_id);
    }
}

void ModelRepoDownloader::RemovePausedModel(const std::string& model_id) {
    auto it = std::find(paused_models_.begin(), paused_models_.end(), model_id);
    if (it != paused_models_.end()) {
        paused_models_.erase(it);
    }
}

// Static utility method: Format file size with smart unit selection
std::string ModelRepoDownloader::FormatFileSizeInfo(int64_t downloaded_bytes, int64_t total_bytes) {
    std::string size_info;
    
    // Smart unit selection: use KB for files < 1MB, MB otherwise
    if (total_bytes < 1024 * 1024) {
        // Use KB for small files
        int64_t downloaded_kb = downloaded_bytes / 1024;
        int64_t total_kb = total_bytes / 1024;
        if (total_kb == 0) total_kb = 1; // Avoid "0 KB / 0 KB"
        size_info = " (" + std::to_string(downloaded_kb) + " KB / " + 
                   std::to_string(total_kb) + " KB)";
    } else {
        // Use MB for larger files
        int64_t downloaded_mb = downloaded_bytes / (1024 * 1024);
        int64_t total_mb = total_bytes / (1024 * 1024);
        size_info = " (" + std::to_string(downloaded_mb) + " MB / " + 
                   std::to_string(total_mb) + " MB)";
    }
    
    return size_info;
}

// Static utility method: Extract filename from path for cleaner display
std::string ModelRepoDownloader::ExtractFileName(const std::string& file_path) {
    size_t last_slash = file_path.find_last_of('/');
    if (last_slash != std::string::npos) {
        return file_path.substr(last_slash + 1);
    }
    return file_path;
}

} // namespace mnncli
