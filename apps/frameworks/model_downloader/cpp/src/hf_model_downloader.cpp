//
// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#include "hf_model_downloader.hpp"
#include "model_file_downloader.hpp"
#include "log_utils.hpp"

#include <iostream>
#include <functional>

namespace mnn::downloader {

HfModelDownloader::HfModelDownloader(const std::string& cache_root_path)
    : ModelRepoDownloader(cache_root_path), hf_api_client_(nullptr) {
    
    // Create HTTP client for metadata requests
    metadata_client_ = std::make_shared<httplib::SSLClient>("huggingface.co");
    // Note: httplib doesn't have direct timeout settings like OkHttp, 
    // but we can handle this in the request logic
}

void HfModelDownloader::Download(const std::string& model_id) {
    try {
        LOG_INFO("Starting download for model: " + model_id);
        NotifyDownloadTaskAdded();
        
        // Store original model_id for notifications
        original_model_id_ = model_id;
        
        // Get repository info
        auto client = getHfApiClient();
        std::string error_info;
        LOG_DEBUG_TAG("Fetching repository info for: " + model_id, "HfModelDownloader");
        auto repo_info = client->GetRepoInfo(getHfModelId(model_id), "main", error_info);
        
        if (!error_info.empty()) {
            LOG_ERROR("Failed to fetch repo info: " + error_info);
            NotifyDownloadFailed(original_model_id_, "Failed to fetch repo info: " + error_info);
            NotifyDownloadTaskRemoved();
            return;
        }
        
        LOG_DEBUG_TAG("Repository info fetched successfully", "HfModelDownloader");
        LOG_DEBUG_TAG("Siblings count: " + std::to_string(repo_info.siblings.size()), "HfModelDownloader");
        
        // Notify repo info
        NotifyRepoInfo(model_id, 0, 0); // TODO: Add proper timestamp and size calculation
        
        // Start download
        downloadHfRepo(repo_info);
        
        NotifyDownloadTaskRemoved();
        LOG_INFO("Download completed for model: " + model_id);
        
    } catch (const std::exception& e) {
        LOG_ERROR("Download failed with exception: " + std::string(e.what()));
        NotifyDownloadFailed(model_id, "Download failed: " + std::string(e.what()));
        NotifyDownloadTaskRemoved();
    }
}

void HfModelDownloader::Pause(const std::string& model_id) {
    AddPausedModel(model_id);
    NotifyDownloadPaused(model_id);
}

void HfModelDownloader::Resume(const std::string& model_id) {
    RemovePausedModel(model_id);
    // Restart download
    Download(model_id);
}

std::filesystem::path HfModelDownloader::GetDownloadPath(const std::string& model_id) {
    LOG_DEBUG_TAG("GetDownloadPath called with model_id = " + model_id, "HfModelDownloader");
    
    // Return cache_root/HuggingFace/owner/model
    auto hf_model_id = getHfModelId(model_id);  // Remove "hf:" prefix if present
    
    LOG_DEBUG_TAG("After getHfModelId, hf_model_id = " + hf_model_id, "HfModelDownloader");
    LOG_DEBUG_TAG("cache_root_path_ = " + cache_root_path_, "HfModelDownloader");
    
    auto path = std::filesystem::path(cache_root_path_) / "HuggingFace" / hf_model_id;
    
    LOG_DEBUG_TAG("Final path: " + path.string(), "HfModelDownloader");
    
    return path;
}

bool HfModelDownloader::DeleteRepo(const std::string& model_id) {
    LOG_DEBUG_TAG("HfModelDownloader::DeleteRepo called with model_id = " + model_id, "HfModelDownloader");
    
    // Simplified: delete model folder directly (ModelScope style)
    auto model_folder = GetDownloadPath(model_id);
    
    LOG_DEBUG_TAG("Got download path: " + model_folder.string(), "HfModelDownloader");
    LOG_INFO("Removing model folder: " + model_folder.string());
    
    if (std::filesystem::exists(model_folder)) {
        LOG_DEBUG_TAG("Model folder exists, attempting to delete: " + model_folder.string(), "HfModelDownloader");
        std::error_code ec;
        std::filesystem::remove_all(model_folder, ec);
        if (ec) {
            LOG_ERROR("Failed to remove model folder " + model_folder.string() + ": " + ec.message());
            return false;
        }
        LOG_DEBUG_TAG("Successfully deleted model folder: " + model_folder.string(), "HfModelDownloader");
        return true;
    } else {
        LOG_DEBUG_TAG("Model folder does not exist: " + model_folder.string(), "HfModelDownloader");
        return false;
    }
}

int64_t HfModelDownloader::GetRepoSize(const std::string& model_id) {
    try {
        auto client = getHfApiClient();
        std::string error_info;
        auto repo_info = client->GetRepoInfo(getHfModelId(model_id), "main", error_info);
        
        if (!error_info.empty()) {
            return 0;
        }
        
        // Calculate total size from metadata
        auto metadata_list = requestMetadataList(repo_info);
        int64_t total_size = 0;
        for (const auto& metadata : metadata_list) {
            total_size += metadata.size;
        }
        
        return total_size;
        
    } catch (const std::exception&) {
        return 0;
    }
}

bool HfModelDownloader::CheckUpdate(const std::string& model_id) {
    try {
        auto client = getHfApiClient();
        std::string error_info;
        auto repo_info = client->GetRepoInfo(getHfModelId(model_id), "main", error_info);
        
        return error_info.empty();
        
    } catch (const std::exception&) {
        return false;
    }
}

void HfModelDownloader::downloadHfRepo(const RepoInfo& repo_info) {
    LOG_INFO("DownloadStart " + repo_info.model_id + " host: " + getHfApiClient()->GetHost());
    
    // Debug logging to see what we're working with
    LOG_DEBUG_TAG("Repository info:", "HfModelDownloader");
    LOG_DEBUG_TAG("  Model ID: " + repo_info.model_id, "HfModelDownloader");
    LOG_DEBUG_TAG("  SHA: " + repo_info.sha, "HfModelDownloader");
    LOG_DEBUG_TAG("  Revision: " + repo_info.revision, "HfModelDownloader");
    LOG_DEBUG_TAG("  OriginalModelId: " + original_model_id_, "HfModelDownloader");
    LOG_DEBUG_TAG("  Siblings count: " + std::to_string(repo_info.siblings.size()), "HfModelDownloader");
    
    if (repo_info.siblings.empty()) {
        LOG_ERROR("No files to download - siblings list is empty!");
        LOG_ERROR("This usually means the API call failed or returned no files");
        std::cout << "[DEBUG] Calling NotifyDownloadFailed due to empty siblings: " << original_model_id_ << std::endl;
        NotifyDownloadFailed(original_model_id_, "No files to download - siblings list is empty");
        return;
    }
    
    LOG_DEBUG_TAG("Files to download:", "HfModelDownloader");
    for (const auto& sibling : repo_info.siblings) {
        LOG_DEBUG_TAG("  - " + sibling, "HfModelDownloader");
    }
    
    // Create download path - use GetDownloadPath to get the correct path with provider prefix
    auto model_folder = GetDownloadPath(repo_info.model_id);
    
    LOG_DEBUG_TAG("Model folder (direct storage): " + model_folder.string(), "HfModelDownloader");
    
    // If already complete, return early
    if (IsDownloadComplete(model_folder)) {
        LOG_DEBUG_TAG("HfModelDownloader: model already complete at " + model_folder.string(), "HfModelDownloader");
        NotifyDownloadFinished(original_model_id_, GetDownloadPath(repo_info.model_id).string());
        return;
    }
    // Ensure directory exists and mark downloading
    std::error_code ec;
    std::filesystem::create_directories(model_folder, ec);
    if (ec) {
        NotifyDownloadFailed(original_model_id_, "Failed to create directories: " + ec.message());
        return;
    }
    MarkDownloading(model_folder);
    
    int64_t total_size = 0;
    int64_t downloaded_size = 0;
    
    auto task_list = collectTaskList(model_folder, repo_info, total_size, downloaded_size);
    
    LOG_DEBUG_TAG("Created " + std::to_string(task_list.size()) + " download tasks", "HfModelDownloader");
    
    if (task_list.empty()) {
        LOG_ERROR("No download tasks created!");
        NotifyDownloadFailed(original_model_id_, "No download tasks created");
        return;
    }
    
    // Download files
    for (const auto& task : task_list) {
        LOG_DEBUG_TAG("Starting download of: " + task.relativePath, "HfModelDownloader");
        
        if (IsPaused(original_model_id_)) {
            NotifyDownloadPaused(original_model_id_);
            return;
        }
        
        std::string error_info;
        if (!downloadFile(task.fileMetadata.location, task.downloadPath, task.fileMetadata, task.relativePath, error_info)) {
            ClearMarkers(model_folder);
            NotifyDownloadFailed(original_model_id_, "Failed to download file: " + error_info);
            return;
        }
        
        // Update progress
        downloaded_size += task.downloadedSize;
        NotifyDownloadProgress(original_model_id_, "file", task.relativePath, 
                             downloaded_size, total_size);
    }
    
    // Validate by size and mark completion
    std::vector<std::pair<std::string, int64_t>> manifest_entries;
    manifest_entries.reserve(task_list.size());
    for (const auto& t : task_list) {
        manifest_entries.emplace_back(t.relativePath, t.fileMetadata.size);
    }
    if (!ValidateFilesBySize(model_folder, manifest_entries)) {
        ClearMarkers(model_folder);
        NotifyDownloadFailed(original_model_id_, "Validation failed: file sizes do not match repo metadata");
        return;
    }
    if (!MarkComplete(model_folder, manifest_entries)) {
        ClearMarkers(model_folder);
        NotifyDownloadFailed(original_model_id_, "Failed to finalize completion markers");
        return;
    }
    // Download completed - notify (no symlink needed with direct storage)
    auto model_path = GetDownloadPath(repo_info.model_id);
    LOG_DEBUG_TAG("Download completed at: " + model_path.string(), "HfModelDownloader");
    std::cout << "[DEBUG] Calling NotifyDownloadFinished: " << original_model_id_ << std::endl;
    NotifyDownloadFinished(original_model_id_, model_path.string());
}

void HfModelDownloader::downloadHfRepoInner(const RepoInfo& repo_info) {
    // This method is now handled by downloadHfRepo
    downloadHfRepo(repo_info);
}

std::vector<FileDownloadTask> HfModelDownloader::collectTaskList(
    const std::filesystem::path& model_folder,
    const RepoInfo& repo_info,
    int64_t& total_size,
    int64_t& downloaded_size) {
    
    std::vector<FileDownloadTask> tasks;
    
    LOG_DEBUG_TAG("collectTaskList: Processing " + std::to_string(repo_info.siblings.size()) + " siblings", "HfModelDownloader");
    
    // Get metadata for all files first
    auto metadata_list = requestMetadataList(repo_info);
    
    for (size_t i = 0; i < repo_info.siblings.size(); ++i) {
        const auto& sub_file = repo_info.siblings[i];
        const auto& metadata = metadata_list[i];
        
        FileDownloadTask task;
        task.etag = metadata.etag;
        task.relativePath = sub_file;
        task.fileMetadata = metadata;
        
        // Simplified: direct download path (ModelScope style)
        // model_folder/filename (no blobs or snapshots)
        task.downloadPath = model_folder / sub_file;
        
        // Calculate downloaded size from existing files
        if (std::filesystem::exists(task.downloadPath)) {
            task.downloadedSize = std::filesystem::file_size(task.downloadPath);
        } else if (std::filesystem::exists(task.GetIncompletePath())) {
            task.downloadedSize = std::filesystem::file_size(task.GetIncompletePath());
        } else {
            task.downloadedSize = 0;
        }
        
        total_size += metadata.size;
        downloaded_size += task.downloadedSize;
        
        LOG_DEBUG_TAG("  Added task: " + sub_file + 
                     " -> path: " + task.downloadPath.string(), "HfModelDownloader");
        
        tasks.push_back(task);
    }
    
    LOG_DEBUG_TAG("collectTaskList: Created " + std::to_string(tasks.size()) + " tasks", "HfModelDownloader");
    
    return tasks;
}

std::vector<HfFileMetadata> HfModelDownloader::requestMetadataList(const RepoInfo& repo_info) {
    std::vector<HfFileMetadata> metadata_list;
    
    for (const auto& sub_file : repo_info.siblings) {
        std::string url = "https://" + getHfApiClient()->GetHost() + "/" + 
                         repo_info.model_id + "/resolve/main/" + sub_file;
        
        LOG_DEBUG_TAG("requestMetadataList: Getting metadata for " + sub_file + " from " + url, "HfModelDownloader");
        
        std::string error_info;
        auto metadata = getFileMetadata(url, error_info);
        metadata_list.push_back(metadata);
    }
    
    return metadata_list;
}

HfFileMetadata HfModelDownloader::getFileMetadata(const std::string& url, std::string& error_info) {
    return HfFileMetadataUtils::GetFileMetadata(url, error_info);
}

bool HfModelDownloader::downloadFile(const std::string& url, const std::filesystem::path& destination_path,
                                    const HfFileMetadata& metadata, const std::string& file_name,
                                    std::string& error_info) {
    
    LOG_DEBUG_TAG("downloadFile: Starting download of " + file_name, "HfModelDownloader");
    LOG_DEBUG_TAG("downloadFile: URL: " + url, "HfModelDownloader");
    
    // Use ModelFileDownloader for actual file download
    ModelFileDownloader downloader;
    
    // Create FileDownloadTask (simplified for direct storage)
    FileDownloadTask task;
    task.etag = metadata.etag;
    task.relativePath = file_name;
    task.fileMetadata = metadata;
    
    // Simplified: direct download path (ModelScope style)
    // destination_path is the final file location
    task.downloadPath = destination_path;
    task.downloadedSize = 0;
    
    LOG_DEBUG_TAG("downloadFile: Created download task for " + file_name, "HfModelDownloader");
    LOG_DEBUG_TAG("downloadFile: Download path: " + task.downloadPath.string(), "HfModelDownloader");
    
    // Create a simple FileDownloadListener implementation
    class SimpleFileDownloadListener : public FileDownloadListener {
    public:
        SimpleFileDownloadListener(std::string& error_info, std::function<void(const std::string&, int64_t, int64_t)> callback) 
            : error_info_(error_info), callback_(callback) {}
        
        bool onDownloadDelta(const std::string* fileName, int64_t downloadedBytes, int64_t totalBytes, int64_t delta) override {
            // Use unified progress system with ModelScope style
            if (totalBytes > 0) {
                std::string name = fileName ? *fileName : "file";
                if (callback_) {
                    callback_(name, downloadedBytes, totalBytes);
                }
            }
            return false; // Don't pause
        }
        
    private:
        std::string& error_info_;
        std::function<void(const std::string&, int64_t, int64_t)> callback_;
    };
    
    auto progress_callback = [this, &file_name](const std::string& name, int64_t downloaded, int64_t total) {
        this->NotifyDownloadProgress(this->original_model_id_, "file", file_name, downloaded, total);
    };
    
    SimpleFileDownloadListener listener(error_info, progress_callback);
    
    try {
        LOG_DEBUG_TAG("downloadFile: Calling downloader.DownloadFile()", "HfModelDownloader");
        downloader.DownloadFile(task, listener);
        LOG_DEBUG_TAG("downloadFile: Download completed successfully for " + file_name, "HfModelDownloader");
    } catch (const FileDownloadException& e) {
        error_info = "Download failed: " + std::string(e.what());
        LOG_ERROR("downloadFile: " + error_info);
        return false;
    } catch (const DownloadPausedException& e) {
        error_info = "Download paused: " + std::string(e.what());
        LOG_ERROR("downloadFile: " + error_info);
        return false;
    } catch (const std::exception& e) {
        error_info = "Download failed with exception: " + std::string(e.what());
        LOG_ERROR("downloadFile: " + error_info);
        return false;
    }
    
    return true;
}

std::shared_ptr<HfApiClient> HfModelDownloader::getHfApiClient() {
    if (!hf_api_client_) {
        hf_api_client_ = std::make_shared<HfApiClient>();
    }
    return hf_api_client_;
}

std::string HfModelDownloader::getHfModelId(const std::string& model_id) {
    LOG_DEBUG_TAG("getHfModelId called with model_id = " + model_id, "HfModelDownloader");
    
    if (model_id.rfind("HuggingFace/", 0) == 0) {
        std::string result = model_id.substr(12);
        LOG_DEBUG_TAG("Stripping 'HuggingFace/' prefix, result = " + result, "HfModelDownloader");
        return result;
    }
    if (!model_id.empty() && (model_id[0] == ':' || model_id[0] == '/')) {
        std::string result = model_id.substr(1);
        LOG_DEBUG_TAG("Stripping leading ':' or '/', result = " + result, "HfModelDownloader");
        return result;
    }
    LOG_DEBUG_TAG("No prefix to strip, returning as is: " + model_id, "HfModelDownloader");
    return model_id;
}

std::string HfModelDownloader::getCachePathRoot(const std::string& model_download_path_root) {
    return model_download_path_root;
}

std::filesystem::path HfModelDownloader::getModelPath(const std::string& cache_root_path, const std::string& model_id) {
    return std::filesystem::path(cache_root_path) / model_id;
}

} // namespace mnn::downloader
