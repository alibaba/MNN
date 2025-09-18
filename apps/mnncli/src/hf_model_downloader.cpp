//
// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#include "hf_model_downloader.hpp"
#include "file_utils.hpp"
#include "model_file_downloader.hpp"
#include "log_utils.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <thread>
#include <chrono>

namespace mnncli {

HfModelDownloader::HfModelDownloader(const std::string& cache_root_path)
    : ModelRepoDownloader(cache_root_path), hf_api_client_(nullptr) {
    
    // Create HTTP client for metadata requests
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    metadata_client_ = std::make_shared<httplib::SSLClient>("huggingface.co");
#else
    metadata_client_ = std::make_shared<httplib::Client>("huggingface.co");
#endif
    // Note: httplib doesn't have direct timeout settings like OkHttp, 
    // but we can handle this in the request logic
}

void HfModelDownloader::download(const std::string& model_id) {
    try {
        LOG_INFO("Starting download for model: " + model_id);
        notifyDownloadTaskAdded();
        
        // Get repository info
        auto client = getHfApiClient();
        std::string error_info;
        LOG_DEBUG_TAG("Fetching repository info for: " + model_id, "HfModelDownloader");
        auto repo_info = client->GetRepoInfo(getHfModelId(model_id), "main", error_info);
        
        if (!error_info.empty()) {
            LOG_ERROR("Failed to fetch repo info: " + error_info);
            notifyDownloadFailed(model_id, "Failed to fetch repo info: " + error_info);
            notifyDownloadTaskRemoved();
            return;
        }
        
        LOG_DEBUG_TAG("Repository info fetched successfully", "HfModelDownloader");
        
        // Notify repo info
        notifyRepoInfo(model_id, 0, 0); // TODO: Add proper timestamp and size calculation
        
        // Start download
        downloadHfRepo(repo_info);
        
        notifyDownloadTaskRemoved();
        LOG_INFO("Download completed for model: " + model_id);
        
    } catch (const std::exception& e) {
        LOG_ERROR("Download failed with exception: " + std::string(e.what()));
        notifyDownloadFailed(model_id, "Download failed: " + std::string(e.what()));
        notifyDownloadTaskRemoved();
    }
}

void HfModelDownloader::pause(const std::string& model_id) {
    addPausedModel(model_id);
    notifyDownloadPaused(model_id);
}

void HfModelDownloader::resume(const std::string& model_id) {
    removePausedModel(model_id);
    // Restart download
    download(model_id);
}

std::filesystem::path HfModelDownloader::getDownloadPath(const std::string& model_id) {
    return getModelPath(cache_root_path_, model_id);
}

bool HfModelDownloader::deleteRepo(const std::string& model_id) {
    auto hf_model_id = getHfModelId(model_id);
    auto repo_folder_name = FileUtils::RepoFolderName(hf_model_id, "model");
    auto hf_storage_folder = std::filesystem::path(cache_root_path_) / repo_folder_name;
    
    LOG_INFO("Removing storage folder: " + hf_storage_folder.string());
    
    if (std::filesystem::exists(hf_storage_folder)) {
        std::error_code ec;
        std::filesystem::remove_all(hf_storage_folder, ec);
        if (ec) {
            LOG_ERROR("Failed to remove storage folder " + hf_storage_folder.string() + ": " + ec.message());
            return false;
        }
    }
    
    auto hf_link_folder = getDownloadPath(model_id);
    LOG_INFO("Removing link folder: " + hf_link_folder.string());
    
    if (std::filesystem::exists(hf_link_folder)) {
        std::filesystem::remove(hf_link_folder);
    }
    
    return true;
}

int64_t HfModelDownloader::getRepoSize(const std::string& model_id) {
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

bool HfModelDownloader::checkUpdate(const std::string& model_id) {
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
    LOG_DEBUG_TAG("  Siblings count: " + std::to_string(repo_info.siblings.size()), "HfModelDownloader");
    
    if (repo_info.siblings.empty()) {
        LOG_ERROR("No files to download - siblings list is empty!");
        LOG_ERROR("This usually means the API call failed or returned no files");
        return;
    }
    
    LOG_DEBUG_TAG("Files to download:", "HfModelDownloader");
    for (const auto& sibling : repo_info.siblings) {
        LOG_DEBUG_TAG("  - " + sibling, "HfModelDownloader");
    }
    
    // Create download tasks
    auto storage_folder = std::filesystem::path(cache_root_path_) / 
                         FileUtils::RepoFolderName(repo_info.model_id, "model");
    
    auto parent_pointer_path = storage_folder / "snapshots" / repo_info.sha;
    
    LOG_DEBUG_TAG("Storage folder: " + storage_folder.string(), "HfModelDownloader");
    LOG_DEBUG_TAG("Parent pointer path: " + parent_pointer_path.string(), "HfModelDownloader");
    
    int64_t total_size = 0;
    int64_t downloaded_size = 0;
    
    auto task_list = collectTaskList(storage_folder, parent_pointer_path, repo_info, total_size, downloaded_size);
    
    LOG_DEBUG_TAG("Created " + std::to_string(task_list.size()) + " download tasks", "HfModelDownloader");
    
    if (task_list.empty()) {
        LOG_ERROR("No download tasks created!");
        return;
    }
    
    // Download files
    for (const auto& task : task_list) {
        LOG_DEBUG_TAG("Starting download of: " + task.relativePath, "HfModelDownloader");
        
        if (isPaused(repo_info.model_id)) {
            notifyDownloadPaused(repo_info.model_id);
            return;
        }
        
        std::string error_info;
        if (!downloadFile(task.fileMetadata.location, task.pointerPath, task.fileMetadata, task.relativePath, error_info)) {
            notifyDownloadFailed(repo_info.model_id, "Failed to download file: " + error_info);
            return;
        }
        
        // Update progress
        downloaded_size += task.downloadedSize;
        notifyDownloadProgress(repo_info.model_id, "file", task.relativePath, 
                             downloaded_size, total_size);
    }
    
    // Create symlink
    auto link_path = getDownloadPath(repo_info.model_id);
    std::filesystem::create_directories(link_path.parent_path());
    std::filesystem::create_symlink(parent_pointer_path, link_path);
    
    notifyDownloadFinished(repo_info.model_id, link_path.string());
}

void HfModelDownloader::downloadHfRepoInner(const RepoInfo& repo_info) {
    // This method is now handled by downloadHfRepo
    downloadHfRepo(repo_info);
}

std::vector<FileDownloadTask> HfModelDownloader::collectTaskList(
    const std::filesystem::path& storage_folder,
    const std::filesystem::path& parent_pointer_path,
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
        
        // Blob path: storage_folder/blobs/etag (stores actual file content)
        task.blobPath = storage_folder / "blobs" / metadata.etag;
        
        // Blob incomplete path: storage_folder/blobs/etag.incomplete (temporary download file)
        task.blobPathIncomplete = storage_folder / "blobs" / (metadata.etag + ".incomplete");
        
        // Pointer path: parent_pointer_path/sub_file (creates symlink to blob)
        task.pointerPath = parent_pointer_path / sub_file;
        
        // Calculate downloaded size from existing files
        // Only check file size if etag is valid (non-empty)
        if (!metadata.etag.empty()) {
            if (std::filesystem::exists(task.blobPath)) {
                task.downloadedSize = std::filesystem::file_size(task.blobPath);
            } else if (std::filesystem::exists(task.blobPathIncomplete)) {
                task.downloadedSize = std::filesystem::file_size(task.blobPathIncomplete);
            } else {
                task.downloadedSize = 0;
            }
        } else {
            LOG_WARNING("collectTaskList: Empty etag for file " + sub_file + ", skipping size check");
            task.downloadedSize = 0;
        }
        
        total_size += metadata.size;
        downloaded_size += task.downloadedSize;
        
        LOG_DEBUG_TAG("  Added task: " + sub_file + 
                     " -> blob: " + task.blobPath.string() + 
                     ", pointer: " + task.pointerPath.string(), "HfModelDownloader");
        
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
        
        std::string error_info;
        auto metadata = getFileMetadata(url, error_info);
        metadata_list.push_back(metadata);
    }
    
    return metadata_list;
}

HfFileMetadata HfModelDownloader::getFileMetadata(const std::string& url, std::string& error_info) {
    return HfFileMetadataUtils::getFileMetadata(url, error_info);
}

bool HfModelDownloader::downloadFile(const std::string& url, const std::filesystem::path& destination_path,
                                    const HfFileMetadata& metadata, const std::string& file_name,
                                    std::string& error_info) {
    
    LOG_DEBUG_TAG("downloadFile: Starting download of " + file_name, "HfModelDownloader");
    LOG_DEBUG_TAG("downloadFile: URL: " + url, "HfModelDownloader");
    
    // Use ModelFileDownloader for actual file download (matching Android API)
    ModelFileDownloader downloader;
    
    // Create FileDownloadTask matching Android structure
    FileDownloadTask task;
    task.etag = metadata.etag;
    task.relativePath = file_name;
    task.fileMetadata = metadata;
    
    // Blob path: storage_folder/blobs/etag (stores actual file content)
    // The destination_path is already the pointer path, so we need to go up to the storage folder
    auto storage_folder = destination_path.parent_path().parent_path().parent_path(); // Go up from "snapshots/sha/filename" to storage folder
    
    // Ensure etag is valid before constructing blob paths
    if (metadata.etag.empty()) {
        error_info = "Invalid metadata: empty etag for file " + file_name;
        LOG_ERROR("downloadFile: " + error_info);
        return false;
    }
    
    task.blobPath = storage_folder / "blobs" / metadata.etag;
    
    // Blob incomplete path: storage_folder/blobs/etag.incomplete (temporary download file)
    task.blobPathIncomplete = storage_folder / "blobs" / (metadata.etag + ".incomplete");
    
    // Pointer path: destination_path (creates symlink to blob)
    task.pointerPath = destination_path;
    task.downloadedSize = 0;
    
    LOG_DEBUG_TAG("downloadFile: Created download task for " + file_name, "HfModelDownloader");
    LOG_DEBUG_TAG("downloadFile: Blob path: " + task.blobPath.string(), "HfModelDownloader");
    LOG_DEBUG_TAG("downloadFile: Pointer path: " + task.pointerPath.string(), "HfModelDownloader");
    
    // Create a simple FileDownloadListener implementation
    class SimpleFileDownloadListener : public FileDownloadListener {
    public:
        SimpleFileDownloadListener(std::string& error_info) : error_info_(error_info) {}
        
        bool onDownloadDelta(const std::string* fileName, int64_t downloadedBytes, int64_t totalBytes, int64_t delta) override {
            // Simple progress logging
            if (totalBytes > 0) {
                double percentage = (static_cast<double>(downloadedBytes) / totalBytes) * 100.0;
                printf("\rDownloading %s: %.2f%%", fileName ? fileName->c_str() : "file", percentage);
                fflush(stdout);
            }
            return false; // Don't pause
        }
        
    private:
        std::string& error_info_;
    };
    
    SimpleFileDownloadListener listener(error_info);
    
    try {
        LOG_DEBUG_TAG("downloadFile: Calling downloader.downloadFile()", "HfModelDownloader");
        downloader.downloadFile(task, listener);
        LOG_DEBUG_TAG("downloadFile: Download completed successfully for " + file_name, "HfModelDownloader");
        printf("\nDownload completed successfully\n");
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
    
    // Check if file already exists and use actual size (like Kotlin does)
    LOG_DEBUG_TAG("downloadFile: Checking if blob file exists: " + task.blobPath.string(), "HfModelDownloader");
    
    if (std::filesystem::exists(task.blobPath)) {
        int64_t actual_size = std::filesystem::file_size(task.blobPath);
        LOG_DEBUG_TAG("downloadFile: Blob file exists, actual size: " + std::to_string(actual_size) + " bytes", "HfModelDownloader");
        
        // Use the actual file size as downloaded size (like Kotlin does)
        task.downloadedSize = actual_size;
        
        // Skip download if file is complete (like Kotlin logic)
        if (actual_size >= metadata.size) {
            LOG_DEBUG_TAG("downloadFile: File already complete, skipping download: " + file_name, "HfModelDownloader");
            return true;
        }
        
        LOG_DEBUG_TAG("downloadFile: File exists but incomplete, will resume download: " + file_name, "HfModelDownloader");
    } else {
        LOG_DEBUG_TAG("downloadFile: Blob file does not exist, will download: " + task.blobPath.string(), "HfModelDownloader");
        task.downloadedSize = 0;
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
    // Normalize prefixes to extract plain owner/repo
    // Support both colon and slash styles, e.g.,
    //   "HuggingFace:owner/repo", "hf:owner/repo",
    //   "HuggingFace/owner/repo", "hf/owner/repo"
    if (model_id.rfind("HuggingFace/", 0) == 0) {
        return model_id.substr(12);
    }
    if (!model_id.empty() && (model_id[0] == ':' || model_id[0] == '/')) {
        return model_id.substr(1);
    }
    return model_id;
}

std::string HfModelDownloader::getCachePathRoot(const std::string& model_download_path_root) {
    return model_download_path_root;
}

std::filesystem::path HfModelDownloader::getModelPath(const std::string& cache_root_path, const std::string& model_id) {
    return std::filesystem::path(cache_root_path) / FileUtils::GetFileName(model_id);
}

} // namespace mnncli
