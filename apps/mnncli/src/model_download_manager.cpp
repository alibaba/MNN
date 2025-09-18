//
// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#include "model_download_manager.hpp"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <thread>
#include <cctype>

namespace mnncli {

// ModelSources implementation
ModelSource ModelSources::fromString(const std::string& source_str) {
    std::string lower = source_str;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    
    if (lower == "huggingface" || lower == "hf") return ModelSource::HUGGING_FACE;
    if (lower == "modelscope" || lower == "ms") return ModelSource::MODEL_SCOPE;
    if (lower == "modelers" || lower == "ml") return ModelSource::MODELERS;
    
    return ModelSource::UNKNOWN;
}

std::string ModelSources::toString(ModelSource source) {
    switch (source) {
        case ModelSource::HUGGING_FACE: return SOURCE_HUGGING_FACE;
        case ModelSource::MODEL_SCOPE: return SOURCE_MODEL_SCOPE;
        case ModelSource::MODELERS: return SOURCE_MODELERS;
        default: return "Unknown";
    }
}

ModelSource ModelSources::getSource(const std::string& model_id) {
    auto splits = splitSource(model_id);
    return fromString(splits.first);
}

std::pair<std::string, std::string> ModelSources::splitSource(const std::string& model_id) {
    size_t colon_pos = model_id.find(':');
    if (colon_pos == std::string::npos) {
        return {"", model_id};
    }
    
    std::string source = model_id.substr(0, colon_pos);
    std::string path = model_id.substr(colon_pos + 1);
    return {source, path};
}

std::string ModelSources::getModelName(const std::string& model_id) {
    auto splits = splitSource(model_id);
    if (splits.second.empty()) {
        return model_id;
    }
    
    // Extract the last part of the path as model name
    size_t last_slash = splits.second.find_last_of('/');
    if (last_slash != std::string::npos) {
        return splits.second.substr(last_slash + 1);
    }
    
    return splits.second;
}

// ModelDownloadManager implementation
ModelDownloadManager::ModelDownloadManager(const std::string& cache_root_path)
    : cache_root_path_(cache_root_path), verbose_(false) {
    
    // Initialize downloaders
    hf_downloader_ = std::make_unique<HfModelDownloader>(cache_root_path);
    ms_downloader_ = std::make_unique<MsModelDownloader>(cache_root_path);
    ml_downloader_ = std::make_unique<MlModelDownloader>(cache_root_path);
    
    // Set up download callbacks for each downloader
    setupDownloadCallbacks();
}

ModelDownloadManager& ModelDownloadManager::getInstance(const std::string& cache_root_path) {
    static ModelDownloadManager instance(cache_root_path);
    return instance;
}

void ModelDownloadManager::addListener(DownloadListener* listener) {
    if (listener && std::find(listeners_.begin(), listeners_.end(), listener) == listeners_.end()) {
        listeners_.push_back(listener);
        std::cout << "[" << TAG << "] Added listener: " << listener->getClassTypeName() << std::endl;
    }
}

void ModelDownloadManager::removeListener(DownloadListener* listener) {
    auto it = std::find(listeners_.begin(), listeners_.end(), listener);
    if (it != listeners_.end()) {
        listeners_.erase(it);
        std::cout << "[" << TAG << "] Removed listener: " << listener->getClassTypeName() << std::endl;
    }
}

void ModelDownloadManager::startDownload(const std::string& model_id) {
    auto splits = ModelSources::splitSource(model_id);
    if (splits.first.empty()) {
        // Default to HuggingFace if no source specified
        startDownload(model_id, "HuggingFace");
        return;
    }
    startDownload(model_id, splits.first);
}

void ModelDownloadManager::startDownload(const std::string& model_id, const std::string& source) {
    startDownload(model_id, source, ModelSources::getModelName(model_id));
}

void ModelDownloadManager::startDownload(const std::string& model_id, const std::string& source, const std::string& model_name) {
    if (verbose_) {
        std::cout << "[" << TAG << "] Starting download: " << model_id << " source: " << source << std::endl;
    }
    
    // Get appropriate downloader
    auto downloader = getDownloaderForSource(source);
    if (!downloader) {
        std::string error = "Unknown source: " + source;
        if (verbose_) {
            std::cerr << "[" << TAG << "] " << error << std::endl;
        }
        return;
    }
    
    // Notify listeners
    for (auto listener : listeners_) {
        listener->onDownloadStart(model_id);
    }
    
    // Track active download
    addActiveDownload(model_id, model_name);
    
    // Start download
    downloader->download(model_id);
}

void ModelDownloadManager::pauseDownload(const std::string& model_id) {
    if (verbose_) {
        std::cout << "[" << TAG << "] Pausing download: " << model_id << std::endl;
    }
    
    auto source = ModelSources::getSource(model_id);
    auto downloader = getDownloaderForSource(source);
    if (downloader) {
        downloader->pause(model_id);
    }
}

void ModelDownloadManager::resumeDownload(const std::string& model_id) {
    if (verbose_) {
        std::cout << "[" << TAG << "] Resuming download: " << model_id << std::endl;
    }
    
    auto source = ModelSources::getSource(model_id);
    auto downloader = getDownloaderForSource(source);
    if (downloader) {
        downloader->resume(model_id);
    }
}

void ModelDownloadManager::cancelDownload(const std::string& model_id) {
    if (verbose_) {
        std::cout << "[" << TAG << "] Canceling download: " << model_id << std::endl;
    }
    
    // Remove from active downloads
    removeActiveDownload(model_id);
    
    // Update state
    updateDownloadState(model_id, DownloadState::NOT_START);
}

std::filesystem::path ModelDownloadManager::getDownloadedFile(const std::string& model_id) {
    auto source = ModelSources::getSource(model_id);
    auto downloader = getDownloaderForSource(source);
    if (downloader) {
        return downloader->getDownloadPath(model_id);
    }
    return std::filesystem::path();
}

bool ModelDownloadManager::deleteRepo(const std::string& model_id) {
    auto source = ModelSources::getSource(model_id);
    auto downloader = getDownloaderForSource(source);
    if (downloader) {
        return downloader->deleteRepo(model_id);
    }
    return false;
}

int64_t ModelDownloadManager::getRepoSize(const std::string& model_id) {
    auto source = ModelSources::getSource(model_id);
    auto downloader = getDownloaderForSource(source);
    if (downloader) {
        return downloader->getRepoSize(model_id);
    }
    return 0;
}

bool ModelDownloadManager::checkUpdate(const std::string& model_id) {
    auto source = ModelSources::getSource(model_id);
    auto downloader = getDownloaderForSource(source);
    if (downloader) {
        return downloader->checkUpdate(model_id);
    }
    return false;
}

DownloadProgress ModelDownloadManager::getDownloadInfo(const std::string& model_id) {
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
    auto downloaded_file = getDownloadedFile(model_id);
    if (std::filesystem::exists(downloaded_file)) {
        info.state = DownloadState::COMPLETED;
        info.progress = 1.0;
        info.saved_size = getRealDownloadSize(model_id);
        info.total_size = getRepoSize(model_id);
    }
    
    download_info_map_[model_id] = info;
    return info;
}

std::vector<std::string> ModelDownloadManager::getActiveDownloads() const {
    return active_downloads_;
}

bool ModelDownloadManager::isDownloading(const std::string& model_id) const {
    return std::find(active_downloads_.begin(), active_downloads_.end(), model_id) != active_downloads_.end();
}

ModelRepoDownloader* ModelDownloadManager::getDownloaderForSource(ModelSource source) {
    switch (source) {
        case ModelSource::HUGGING_FACE: return hf_downloader_.get();
        case ModelSource::MODEL_SCOPE: return ms_downloader_.get();
        case ModelSource::MODELERS: return ml_downloader_.get();
        default: return nullptr;
    }
}

ModelRepoDownloader* ModelDownloadManager::getDownloaderForSource(const std::string& source_str) {
    return getDownloaderForSource(ModelSources::fromString(source_str));
}

void ModelDownloadManager::updateDownloadState(const std::string& model_id, DownloadState state) {
    auto& info = download_info_map_[model_id];
    info.model_id = model_id;
    info.state = state;
    
    // Notify listeners
    for (auto listener : listeners_) {
        // Create progress object for state change
        DownloadProgress progress = info;
        listener->onDownloadProgress(model_id, progress);
    }
}

void ModelDownloadManager::updateDownloadProgress(const std::string& model_id, const std::string& stage,
                                                const std::string& current_file, int64_t saved_size, int64_t total_size) {
    auto& info = download_info_map_[model_id];
    info.model_id = model_id;
    info.stage = stage;
    info.current_file = current_file;
    info.saved_size = saved_size;
    info.total_size = total_size;
    info.progress = (total_size > 0) ? static_cast<double>(saved_size) / total_size : 0.0;
    
    // Calculate download speed
    calculateDownloadSpeed(model_id, saved_size);
    
    // Notify listeners
    for (auto listener : listeners_) {
        listener->onDownloadProgress(model_id, info);
    }
}

void ModelDownloadManager::addActiveDownload(const std::string& model_id, const std::string& display_name) {
    if (std::find(active_downloads_.begin(), active_downloads_.end(), model_id) == active_downloads_.end()) {
        active_downloads_.push_back(model_id);
        active_download_names_[model_id] = display_name;
    }
}

void ModelDownloadManager::removeActiveDownload(const std::string& model_id) {
    auto it = std::find(active_downloads_.begin(), active_downloads_.end(), model_id);
    if (it != active_downloads_.end()) {
        active_downloads_.erase(it);
        active_download_names_.erase(model_id);
    }
}

int64_t ModelDownloadManager::getRealDownloadSize(const std::string& model_id) {
    // This would implement actual file size calculation
    // For now, return 0 as placeholder
    return 0;
}

void ModelDownloadManager::calculateDownloadSpeed(const std::string& model_id, int64_t current_download_size) {
    auto current_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    auto last_time_it = last_log_times_.find(model_id);
    auto last_size_it = last_download_sizes_.find(model_id);
    
    if (last_time_it != last_log_times_.end() && last_size_it != last_download_sizes_.end()) {
        int64_t time_diff = current_time - last_time_it->second;
        int64_t size_diff = current_download_size - last_size_it->second;
        
        if (time_diff >= SPEED_UPDATE_INTERVAL_MS && size_diff >= 0) {
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
                
                std::cout << "[" << TAG << "] Download speed for " << model_id << ": " << speed_str << std::endl;
            }
        }
    }
    
    // Update tracking
    last_log_times_[model_id] = current_time;
    last_download_sizes_[model_id] = current_download_size;
}

void ModelDownloadManager::setupDownloadCallbacks() {
    // Set up callbacks for each downloader to forward events to listeners
    // Set this manager as the listener for each downloader
    hf_downloader_->setListener(this);
    ms_downloader_->setListener(this);
    ml_downloader_->setListener(this);
}

// DownloadListener implementation
void ModelDownloadManager::onDownloadFinished(const std::string& model_id, const std::string& path) {
    if (verbose_) {
        std::cout << "[" << TAG << "] Download finished: " << model_id << " -> " << path << std::endl;
    }
    
    // Remove from active downloads
    removeActiveDownload(model_id);
    
    // Update download state
    updateDownloadState(model_id, DownloadState::COMPLETED);
    
    // Forward to other listeners
    for (auto listener : listeners_) {
        listener->onDownloadFinished(model_id, path);
    }
}

void ModelDownloadManager::onDownloadFailed(const std::string& model_id, const std::string& error) {
    if (verbose_) {
        std::cout << "[" << TAG << "] Download failed: " << model_id << " - " << error << std::endl;
    }
    
    // Remove from active downloads
    removeActiveDownload(model_id);
    
    // Update download state
    updateDownloadState(model_id, DownloadState::FAILED);
    
    // Forward to other listeners
    for (auto listener : listeners_) {
        listener->onDownloadFailed(model_id, error);
    }
}

void ModelDownloadManager::onDownloadPaused(const std::string& model_id) {
    if (verbose_) {
        std::cout << "[" << TAG << "] Download paused: " << model_id << std::endl;
    }
    
    // Remove from active downloads
    removeActiveDownload(model_id);
    
    // Update download state
    updateDownloadState(model_id, DownloadState::PAUSED);
    
    // Forward to other listeners
    for (auto listener : listeners_) {
        listener->onDownloadPaused(model_id);
    }
}

} // namespace mnncli
