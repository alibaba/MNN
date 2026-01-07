//
// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#include "model_repo_downloader.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>

namespace mnn::downloader {

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

// --- Completion markers and manifest helpers ---
static inline std::filesystem::path markersDir(const std::filesystem::path& model_folder) {
    return model_folder / ".mnncli";
}

static inline std::filesystem::path downloadingMarker(const std::filesystem::path& model_folder) {
    return markersDir(model_folder) / ".downloading";
}

static inline std::filesystem::path completeMarker(const std::filesystem::path& model_folder) {
    return markersDir(model_folder) / ".complete";
}

static inline std::filesystem::path manifestPath(const std::filesystem::path& model_folder) {
    return markersDir(model_folder) / "manifest.json";
}

bool ModelRepoDownloader::ValidateFilesBySize(
    const std::filesystem::path& model_folder,
    const std::vector<std::pair<std::string, int64_t>>& manifest_entries
) const {
    for (const auto& entry : manifest_entries) {
        const auto& relative_path = entry.first;
        int64_t expected_size = entry.second;
        std::filesystem::path full_path = model_folder / relative_path;
        if (!std::filesystem::exists(full_path)) {
            return false;
        }
        std::error_code ec;
        auto size = std::filesystem::file_size(full_path, ec);
        if (ec || static_cast<int64_t>(size) != expected_size) {
            return false;
        }
    }
    return true;
}

bool ModelRepoDownloader::IsDownloadComplete(const std::filesystem::path& model_folder) const {
    auto cm = completeMarker(model_folder);
    if (!std::filesystem::exists(cm)) {
        return false;
    }
    // If manifest exists, validate sizes
    auto mp = manifestPath(model_folder);
    if (std::filesystem::exists(mp)) {
        std::ifstream in(mp);
        if (!in.is_open()) return false;
        // Minimal JSON parsing without dependency: expect lines of "path\t size"
        // But since we have header-only json might not be present, use a simple TSV fallback format.
        std::vector<std::pair<std::string, int64_t>> entries;
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            // format: path\t<tab>size
            auto tab = line.find('\t');
            if (tab == std::string::npos) continue;
            std::string rel = line.substr(0, tab);
            int64_t size = std::stoll(line.substr(tab + 1));
            entries.emplace_back(rel, size);
        }
        in.close();
        if (!entries.empty()) {
            return ValidateFilesBySize(model_folder, entries);
        }
    }
    return true;
}

void ModelRepoDownloader::MarkDownloading(const std::filesystem::path& model_folder) const {
    std::error_code ec;
    std::filesystem::create_directories(markersDir(model_folder), ec);
    std::ofstream out(downloadingMarker(model_folder));
    out << "1";
}

bool ModelRepoDownloader::MarkComplete(
    const std::filesystem::path& model_folder,
    const std::vector<std::pair<std::string, int64_t>>& manifest_entries
) const {
    std::error_code ec;
    std::filesystem::create_directories(markersDir(model_folder), ec);
    // Write a simple TSV manifest to avoid introducing a JSON dependency here
    if (!manifest_entries.empty()) {
        std::ofstream mf(manifestPath(model_folder));
        if (!mf.is_open()) return false;
        for (const auto& e : manifest_entries) {
            mf << e.first << '\t' << e.second << '\n';
        }
        mf.close();
    }
    // Atomically create complete marker after manifest
    std::ofstream cm(completeMarker(model_folder));
    if (!cm.is_open()) return false;
    cm << "1";
    // Remove downloading marker if present
    std::filesystem::remove(downloadingMarker(model_folder), ec);
    return true;
}

void ModelRepoDownloader::ClearMarkers(const std::filesystem::path& model_folder) const {
    std::error_code ec;
    std::filesystem::remove(completeMarker(model_folder), ec);
    std::filesystem::remove(downloadingMarker(model_folder), ec);
    std::filesystem::remove(manifestPath(model_folder), ec);
}

} // namespace mnn::downloader
