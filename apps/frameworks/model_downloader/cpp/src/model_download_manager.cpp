//
// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#include "model_download_manager.hpp"
#include "log_utils.hpp"
#include <algorithm>
#include <chrono>
#include <cctype>

namespace mnn::downloader {

// ModelDownloadManager implementation
ModelDownloadManager::ModelDownloadManager(const std::string& cache_root_path)
    : cache_root_path_(cache_root_path), verbose_(true) {
    
    // Initialize downloaders
    hf_downloader_ = std::make_unique<HfModelDownloader>(cache_root_path);
    ms_downloader_ = std::make_unique<MsModelDownloader>(cache_root_path);
    ml_downloader_ = std::make_unique<MlModelDownloader>(cache_root_path);
    
    // Set up download callbacks for each downloader
    SetupDownloadCallbacks();
}

ModelDownloadManager& ModelDownloadManager::GetInstance(const std::string& cache_root_path) {
    // Expand ~ to home directory before using the path
    static ModelDownloadManager instance(FileUtils::ExpandTilde(cache_root_path));
    return instance;
}

void ModelDownloadManager::AddListener(DownloadListener* listener) {
    if (listener && std::find(listeners_.begin(), listeners_.end(), listener) == listeners_.end()) {
        listeners_.push_back(listener);
        LOG_DEBUG("Added listener: " + listener->GetClassTypeName());
    }
}

void ModelDownloadManager::RemoveListener(DownloadListener* listener) {
    auto it = std::find(listeners_.begin(), listeners_.end(), listener);
    if (it != listeners_.end()) {
        listeners_.erase(it);
        LOG_DEBUG("Removed listener: " + listener->GetClassTypeName());
    }
}

void ModelDownloadManager::StartDownload(const std::string& model_id) {
    auto splits = ModelSources::SplitSource(model_id);
    if (splits.first.empty()) {
        // Default to HuggingFace if no source specified
        StartDownload(model_id, ModelSources::SOURCE_HUGGING_FACE);
        return;
    }
    StartDownload(model_id, splits.first);
}

void ModelDownloadManager::StartDownload(const std::string& model_id, const std::string& source) {
    StartDownload(model_id, source, ModelSources::GetModelName(model_id));
}

void ModelDownloadManager::StartDownload(const std::string& model_id, const std::string& source, const std::string& model_name) {
    if (verbose_) {
        LOG_INFO("Starting download: " + model_id + " source: " + source);
    }
    
    // Get appropriate downloader
    auto downloader = GetDownloaderForSource(source);
    if (!downloader) {
        std::string error = "Unknown source: " + source;
        if (verbose_) {
            LOG_ERROR(error);
        }
        return;
    }
    
    // Notify listeners
    for (auto listener : listeners_) {
        listener->OnDownloadStart(model_id);
    }
    
    // Track active download
    AddActiveDownload(model_id, model_name);
    
    // Start download
    downloader->Download(model_id);
}

void ModelDownloadManager::PauseDownload(const std::string& model_id) {
    if (verbose_) {
        LOG_INFO("Pausing download: " + model_id);
    }
    
    auto source = ModelSources::GetSource(model_id);
    auto downloader = GetDownloaderForSource(source);
    if (downloader) {
        downloader->Pause(model_id);
    }
}

void ModelDownloadManager::ResumeDownload(const std::string& model_id) {
    if (verbose_) {
        LOG_INFO("Resuming download: " + model_id);
    }
    
    auto source = ModelSources::GetSource(model_id);
    auto downloader = GetDownloaderForSource(source);
    if (downloader) {
        downloader->Resume(model_id);
    }
}

void ModelDownloadManager::CancelDownload(const std::string& model_id) {
    if (verbose_) {
        LOG_INFO("Canceling download: " + model_id);
    }
    
    // Remove from active downloads
    RemoveActiveDownload(model_id);
    
    // Update state
    UpdateDownloadState(model_id, DownloadState::NOT_START);
}

std::filesystem::path ModelDownloadManager::GetDownloadedFile(const std::string& model_id) {
    auto source = ModelSources::GetSource(model_id);
    auto downloader = GetDownloaderForSource(source);
    if (downloader) {
        return downloader->GetDownloadPath(model_id);
    }
    return std::filesystem::path();
}

bool ModelDownloadManager::DeleteRepo(const std::string& model_id) {
    LOG_DEBUG_TAG("ModelDownloadManager::DeleteRepo called with model_id = " + model_id, "ModelDownloadManager");
    
    auto source = ModelSources::GetSource(model_id);
    std::string source_str = ModelSources::ToString(source);
    LOG_DEBUG_TAG("Extracted source: " + source_str, "ModelDownloadManager");
    
    auto downloader = GetDownloaderForSource(source);
    if (downloader) {
        LOG_DEBUG_TAG("Found downloader for source: " + source_str, "ModelDownloadManager");
        bool result = downloader->DeleteRepo(model_id);
        LOG_DEBUG_TAG("Downloader->DeleteRepo returned: " + std::string(result ? "true" : "false"), "ModelDownloadManager");
        return result;
    }
    LOG_DEBUG_TAG("No downloader found for source: " + source_str, "ModelDownloadManager");
    return false;
}

int64_t ModelDownloadManager::GetRepoSize(const std::string& model_id) {
    auto source = ModelSources::GetSource(model_id);
    auto downloader = GetDownloaderForSource(source);
    if (downloader) {
        return downloader->GetRepoSize(model_id);
    }
    return 0;
}

bool ModelDownloadManager::CheckUpdate(const std::string& model_id) {
    auto source = ModelSources::GetSource(model_id);
    auto downloader = GetDownloaderForSource(source);
    if (downloader) {
        return downloader->CheckUpdate(model_id);
    }
    return false;
}

DownloadProgress ModelDownloadManager::GetDownloadInfo(const std::string& model_id) {
    auto it = download_info_map_.find(model_id);
    if (it != download_info_map_.end()) {
        return it->second;
    }
    
    // Create default download info
    DownloadProgress info;
    info.model_id = model_id;
    info.state = DownloadState::NOT_START;
    info.progress = 0.0;
    info.saved_size = 0;
    info.total_size = 0;
    
    // Check if file exists
    auto downloaded_file = GetDownloadedFile(model_id);
    if (std::filesystem::exists(downloaded_file)) {
        info.state = DownloadState::COMPLETED;
        info.progress = 1.0;
        info.saved_size = GetRealDownloadSize(model_id);
        info.total_size = GetRepoSize(model_id);
    }
    
    download_info_map_[model_id] = info;
    return info;
}

std::vector<std::string> ModelDownloadManager::GetActiveDownloads() const {
    return active_downloads_;
}

bool ModelDownloadManager::IsDownloading(const std::string& model_id) const {
    bool found = std::find(active_downloads_.begin(), active_downloads_.end(), model_id) != active_downloads_.end();
    if (verbose_) {
        LOG_DEBUG("isDownloading(" + model_id + "): " + (found ? "true" : "false") + 
                 " (active_downloads_.size()=" + std::to_string(active_downloads_.size()) + ")");
        if (!active_downloads_.empty()) {
            std::string active_list = "Active downloads: ";
            for (const auto& id : active_downloads_) {
                active_list += id + " ";
            }
            LOG_DEBUG(active_list);
        }
    }
    return found;
}

ModelRepoDownloader* ModelDownloadManager::GetDownloaderForSource(ModelSource source) {
    switch (source) {
        case ModelSource::HUGGING_FACE: return hf_downloader_.get();
        case ModelSource::MODEL_SCOPE: return ms_downloader_.get();
        case ModelSource::MODELERS: return ml_downloader_.get();
        default: return nullptr;
    }
}

ModelRepoDownloader* ModelDownloadManager::GetDownloaderForSource(const std::string& source_str) {
    return GetDownloaderForSource(ModelSources::FromString(source_str));
}

void ModelDownloadManager::UpdateDownloadState(const std::string& model_id, DownloadState state) {
    auto& info = download_info_map_[model_id];
    info.model_id = model_id;
    info.state = state;
    
    // Notify listeners
    for (auto listener : listeners_) {
        // Create progress object for state change
        DownloadProgress progress = info;
        listener->OnDownloadProgress(model_id, progress);
    }
}

void ModelDownloadManager::UpdateDownloadProgress(const std::string& model_id, const std::string& stage,
                                                const std::string& current_file, int64_t saved_size, int64_t total_size) {
    auto& info = download_info_map_[model_id];
    info.model_id = model_id;
    info.stage = stage;
    info.current_file = current_file;
    info.saved_size = saved_size;
    info.total_size = total_size;
    info.progress = (total_size > 0) ? static_cast<double>(saved_size) / total_size : 0.0;
    
    // Calculate download speed
    CalculateDownloadSpeed(model_id, saved_size);
    
    // Notify listeners
    for (auto listener : listeners_) {
        listener->OnDownloadProgress(model_id, info);
    }
}

void ModelDownloadManager::AddActiveDownload(const std::string& model_id, const std::string& display_name) {
    if (std::find(active_downloads_.begin(), active_downloads_.end(), model_id) == active_downloads_.end()) {
        active_downloads_.push_back(model_id);
        active_download_names_[model_id] = display_name;
        if (verbose_) {
            LOG_DEBUG("Added to active downloads: " + model_id + " (total: " + std::to_string(active_downloads_.size()) + ")");
        }
    } else {
        if (verbose_) {
            LOG_DEBUG("Model already in active downloads: " + model_id);
        }
    }
}

void ModelDownloadManager::RemoveActiveDownload(const std::string& model_id) {
    auto it = std::find(active_downloads_.begin(), active_downloads_.end(), model_id);
    if (it != active_downloads_.end()) {
        active_downloads_.erase(it);
        active_download_names_.erase(model_id);
        if (verbose_) {
            LOG_DEBUG("Removed from active downloads: " + model_id + " (remaining: " + std::to_string(active_downloads_.size()) + ")");
        }
    } else {
        if (verbose_) {
            LOG_DEBUG("Model not found in active downloads: " + model_id);
        }
    }
}

int64_t ModelDownloadManager::GetRealDownloadSize(const std::string& model_id) {
    // This would implement actual file size calculation
    // For now, return 0 as placeholder
    return 0;
}

void ModelDownloadManager::CalculateDownloadSpeed(const std::string& model_id, int64_t current_download_size) {
    auto current_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    auto last_time_it = last_log_times_.find(model_id);
    auto last_size_it = last_download_sizes_.find(model_id);
    
    if (last_time_it != last_log_times_.end() && last_size_it != last_download_sizes_.end()) {
        int64_t time_diff = current_time - last_time_it->second;
        int64_t size_diff = current_download_size - last_size_it->second;
        
        if (time_diff >= kSpeedUpdateIntervalMs && size_diff >= 0) {
            // Calculate speed (bytes per second)
            double speed_bps = (size_diff * 1000.0) / time_diff;
            
            if (verbose_) {
                std::string speed_str;
                if (speed_bps >= 1024 * 1024) {
                    speed_str = std::to_string(speed_bps / (1024 * 1024)) + " MB/s";
                } else if (speed_bps >= 1024) {
                    speed_str = std::to_string(speed_bps / 1024) + " KB/s";
                } else {
                    speed_str = std::to_string(speed_bps) + " B/s";
                }
                
                LOG_DEBUG("Download speed for " + model_id + ": " + speed_str);
            }
        }
    }
    
    // Update tracking
    last_log_times_[model_id] = current_time;
    last_download_sizes_[model_id] = current_download_size;
}

void ModelDownloadManager::SetupDownloadCallbacks() {
    // Set up callbacks for each downloader to forward events to listeners
    // Set this manager as the listener for each downloader
    hf_downloader_->SetListener(this);
    ms_downloader_->SetListener(this);
    ml_downloader_->SetListener(this);
}

// DownloadListener implementation
void ModelDownloadManager::OnDownloadStart(const std::string& model_id) {
    if (verbose_) {
        LOG_INFO("Download started: " + model_id);
    }
    
    // Add to active downloads
    AddActiveDownload(model_id, model_id);
    
    // Forward to other listeners
    for (auto listener : listeners_) {
        listener->OnDownloadStart(model_id);
    }
}

void ModelDownloadManager::OnDownloadProgress(const std::string& model_id, const DownloadProgress& progress) {
    // Update internal state
    download_info_map_[model_id] = progress;
    
    // Calculate speed if we have saved size
    if (progress.saved_size > 0) {
        CalculateDownloadSpeed(model_id, progress.saved_size);
    }
    
    // Forward to other listeners
    for (auto listener : listeners_) {
        listener->OnDownloadProgress(model_id, progress);
    }
}

void ModelDownloadManager::OnDownloadFinished(const std::string& model_id, const std::string& path) {
    if (verbose_) {
        LOG_INFO("Download finished: " + model_id + " -> " + path);
    }
    
    // Remove from active downloads
    RemoveActiveDownload(model_id);
    
    // Update download state
    UpdateDownloadState(model_id, DownloadState::COMPLETED);
    
    // Forward to other listeners
    for (auto listener : listeners_) {
        listener->OnDownloadFinished(model_id, path);
    }
}

void ModelDownloadManager::OnDownloadFailed(const std::string& model_id, const std::string& error) {
    if (verbose_) {
        LOG_ERROR("Download failed: " + model_id + " - " + error);
    }
    
    // Remove from active downloads
    RemoveActiveDownload(model_id);
    
    // Update download state
    UpdateDownloadState(model_id, DownloadState::FAILED);
    
    // Forward to other listeners
    for (auto listener : listeners_) {
        listener->OnDownloadFailed(model_id, error);
    }
}

void ModelDownloadManager::OnDownloadPaused(const std::string& model_id) {
    if (verbose_) {
        LOG_INFO("Download paused: " + model_id);
    }
    
    // Remove from active downloads
    RemoveActiveDownload(model_id);
    
    // Update download state
    UpdateDownloadState(model_id, DownloadState::PAUSED);
    
    // Forward to other listeners
    for (auto listener : listeners_) {
        listener->OnDownloadPaused(model_id);
    }
}

} // namespace mnn::downloader
