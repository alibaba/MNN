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

void ModelRepoDownloader::notifyDownloadStart(const std::string& model_id) {
    if (listener_) {
        listener_->onDownloadStart(model_id);
    }
    updateDownloadState(model_id, DownloadState::DOWNLOADING);
}

void ModelRepoDownloader::notifyDownloadProgress(const std::string& model_id, const std::string& stage,
                                               const std::string& current_file, int64_t saved_size, int64_t total_size) {
    if (listener_) {
        DownloadProgress progress;
        progress.model_id = model_id;
        progress.stage = stage;
        progress.current_file = current_file;
        progress.saved_size = saved_size;
        progress.total_size = total_size;
        progress.progress = calculateProgress(saved_size, total_size);
        progress.state = download_states_[model_id];
        
        listener_->onDownloadProgress(model_id, progress);
    }
}

void ModelRepoDownloader::notifyDownloadFinished(const std::string& model_id, const std::string& path) {
    if (listener_) {
        listener_->onDownloadFinished(model_id, path);
    }
    updateDownloadState(model_id, DownloadState::COMPLETED);
}

void ModelRepoDownloader::notifyDownloadFailed(const std::string& model_id, const std::string& error) {
    if (listener_) {
        listener_->onDownloadFailed(model_id, error);
    }
    updateDownloadState(model_id, DownloadState::FAILED);
}

void ModelRepoDownloader::notifyDownloadPaused(const std::string& model_id) {
    if (listener_) {
        listener_->onDownloadPaused(model_id);
    }
    updateDownloadState(model_id, DownloadState::PAUSED);
}

void ModelRepoDownloader::notifyDownloadTaskAdded() {
    if (listener_) {
        listener_->onDownloadTaskAdded();
    }
}

void ModelRepoDownloader::notifyDownloadTaskRemoved() {
    if (listener_) {
        listener_->onDownloadTaskRemoved();
    }
}

void ModelRepoDownloader::notifyRepoInfo(const std::string& model_id, int64_t last_modified, int64_t repo_size) {
    if (listener_) {
        listener_->onRepoInfo(model_id, last_modified, repo_size);
    }
}

double ModelRepoDownloader::calculateProgress(int64_t saved_size, int64_t total_size) const {
    if (total_size <= 0) return 0.0;
    return static_cast<double>(saved_size) / static_cast<double>(total_size);
}

void ModelRepoDownloader::updateDownloadState(const std::string& model_id, DownloadState state) {
    download_states_[model_id] = state;
}

bool ModelRepoDownloader::isPaused(const std::string& model_id) const {
    return std::find(paused_models_.begin(), paused_models_.end(), model_id) != paused_models_.end();
}

void ModelRepoDownloader::addPausedModel(const std::string& model_id) {
    if (!isPaused(model_id)) {
        paused_models_.push_back(model_id);
    }
}

void ModelRepoDownloader::removePausedModel(const std::string& model_id) {
    auto it = std::find(paused_models_.begin(), paused_models_.end(), model_id);
    if (it != paused_models_.end()) {
        paused_models_.erase(it);
    }
}

} // namespace mnncli
