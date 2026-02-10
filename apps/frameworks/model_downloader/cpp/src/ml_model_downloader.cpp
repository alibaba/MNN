//
// Created by ruoyi.sjd on 2024/12/18.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#include "ml_model_downloader.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <httplib.h>
#include <iomanip>

namespace mnn::downloader {

MlModelDownloader::MlModelDownloader(const std::string& cache_root_path)
    : ModelRepoDownloader(GetCachePathRoot(cache_root_path)), verbose_(false) {
    ml_api_client_ = std::make_unique<MlApiClient>();
}

// Implement pure virtual functions from ModelRepoDownloader base class
void MlModelDownloader::Download(const std::string& model_id) {
    std::string error_info;
    if (!DownloadModel(model_id, error_info, verbose_)) {
        NotifyDownloadFailed(model_id, error_info);
    } else {
        NotifyDownloadFinished(model_id, GetDownloadPath(model_id).string());
    }
}

void MlModelDownloader::Pause(const std::string& model_id) {
    AddPausedModel(model_id);
    NotifyDownloadPaused(model_id);
}

void MlModelDownloader::Resume(const std::string& model_id) {
    RemovePausedModel(model_id);
    Download(model_id);
}

std::filesystem::path MlModelDownloader::GetDownloadPath(const std::string& model_id) {
    return GetModelPath(cache_root_path_, model_id);
}

bool MlModelDownloader::DeleteRepo(const std::string& model_id) {
    return DeleteRepoImpl(model_id);
}

int64_t MlModelDownloader::GetRepoSize(const std::string& model_id) {
    std::string error_info;
    return GetRepoSizeWithError(model_id, error_info);
}

bool MlModelDownloader::CheckUpdate(const std::string& model_id) {
    std::string error_info;
    return CheckUpdateWithError(model_id, error_info);
}

bool MlModelDownloader::DownloadModel(const std::string& model_id, std::string& error_info, bool verbose) {
    std::cout << "MlModelDownloader download: " << model_id << std::endl;
    
    try {
        return DownloadMlRepo(model_id, error_info);
    } catch (const std::exception& e) {
        error_info = "Download failed: " + std::string(e.what());
        return false;
    }
}

bool MlModelDownloader::DownloadMlRepo(const std::string& model_id, std::string& error_info) {
    // Get the Modelers repository path from model_id
    // Assuming model_id is in format "owner/repo" or "ml:owner/repo"
    std::string modelers_id = model_id;
    if (modelers_id.find("ml:") == 0) {
        modelers_id = modelers_id.substr(3);
    }
    
    std::cout << "MlModelDownloader downloadMlRepo: " << model_id << " modelersId: " << modelers_id << std::endl;
    
    // Parse owner/repo from the repository path
    size_t slash_pos = modelers_id.find('/');
    if (slash_pos == std::string::npos) {
        error_info = "Invalid model ID format, expected: owner/repo";
        return false;
    }
    
    std::string owner = modelers_id.substr(0, slash_pos);
    std::string repo = modelers_id.substr(slash_pos + 1);
    
    // Get repository information
    MlRepoInfo repo_info = ml_api_client_->GetRepoInfo(modelers_id, "main", error_info);
    if (!error_info.empty()) {
        return false;
    }
    
    std::cout << "downloadMlRepo repoInfo: " << repo_info.model_id << std::endl;
    
    return DownloadMlRepoInner(model_id, modelers_id, repo_info, error_info);
}

bool MlModelDownloader::DownloadMlRepoInner(const std::string& model_id, const std::string& modelers_id, 
                                           const MlRepoInfo& ml_repo_info, std::string& error_info) {
    std::cout << "downloadMlRepoInner" << std::endl;
    
    // Check if already downloaded
    auto folder_link_file = GetModelPath(cache_root_path_, model_id);
    if (std::filesystem::exists(folder_link_file)) {
        std::cout << "downloadMlRepoInner already exists at " << folder_link_file.string() << std::endl;
        return true;
    }
    
    // Create storage folder structure
    auto repo_folder_name = FileUtils::RepoFolderName(modelers_id, "model");
    auto storage_folder = std::filesystem::path(cache_root_path_) / repo_folder_name;
    auto parent_pointer_path = FileUtils::GetPointerPathParent(storage_folder, "_no_sha_");
    
    // Create necessary directories
    std::error_code ec;
    std::filesystem::create_directories(storage_folder / "blobs", ec);
    std::filesystem::create_directories(parent_pointer_path, ec);
    
    if (ec) {
        error_info = "Failed to create directories: " + ec.message();
        return false;
    }
    
    // Collect download tasks
    int64_t total_size = 0;
    int64_t downloaded_size = 0;
    auto download_tasks = CollectMlTaskList(modelers_id, storage_folder, parent_pointer_path, 
                                           ml_repo_info, total_size, downloaded_size);
    
    std::cout << "downloadMlRepoInner downloadTaskList: " << download_tasks.size() << std::endl;
    
    // Download each file
    bool has_error = false;
    for (const auto& task : download_tasks) {
        const std::string& file_path = task.first;
        const std::filesystem::path& destination_path = task.second;
        
        // Find file info
        auto it = std::find_if(ml_repo_info.files.begin(), ml_repo_info.files.end(),
                              [&file_path](const MlFileInfo& f) { return f.path == file_path; });
        
        if (it == ml_repo_info.files.end()) {
            std::cerr << "File info not found for: " << file_path << std::endl;
            continue;
        }
        
        // Create download URL for Modelers
        std::string download_url = "https://modelers.cn/api/v1/models/" + modelers_id + "/files/" + file_path;
        
        std::cout << "Downloading file: " << file_path << " (" << it->size << " bytes)" << std::endl;
        
        if (!DownloadFile(download_url, destination_path, it->size, file_path, model_id, error_info)) {
            has_error = true;
            break;
        }
        
        downloaded_size += it->size;
        
        // Show progress using unified system
        NotifyDownloadProgress(model_id, "file", file_path, downloaded_size, total_size);
    }
    
    if (!has_error) {
        // Create the main symlink
        std::filesystem::create_symlink(parent_pointer_path, folder_link_file, ec);
        if (ec) {
            std::cerr << "Failed to create main symlink: " << ec.message() << std::endl;
            error_info = "Failed to create symlink: " + ec.message();
            return false;
        }
        
        std::cout << "✅ Model downloaded successfully to " << folder_link_file.string() << std::endl;
        return true;
    } else {
        std::cerr << "❌ Download failed with errors" << std::endl;
        return false;
    }
}

std::vector<std::pair<std::string, std::filesystem::path>> MlModelDownloader::CollectMlTaskList(
    const std::string& modelers_id,
    const std::filesystem::path& storage_folder,
    const std::filesystem::path& parent_pointer_path,
    const MlRepoInfo& ml_repo_info,
    int64_t& total_size,
    int64_t& downloaded_size) {
    
    std::vector<std::pair<std::string, std::filesystem::path>> file_download_tasks;
    
    for (const auto& sub_file : ml_repo_info.files) {
        if (sub_file.type == "dir") {
            continue; // Skip directories
        }
        
        // Create file paths
        auto blob_path = storage_folder / "blobs" / sub_file.sha256;
        auto pointer_path = parent_pointer_path / sub_file.path;
        
        // Check if already downloaded
        int64_t file_downloaded_size = 0;
        if (std::filesystem::exists(blob_path)) {
            file_downloaded_size = std::filesystem::file_size(blob_path);
        }
        
        total_size += sub_file.size;
        downloaded_size += file_downloaded_size;
        
        file_download_tasks.emplace_back(sub_file.path, pointer_path);
    }
    
    return file_download_tasks;
}

bool MlModelDownloader::DownloadFile(const std::string& url, const std::filesystem::path& destination_path, 
                                    int64_t expected_size, const std::string& file_name, const std::string& model_id, std::string& error_info) {
    // Parse URL to get host and path
    size_t protocol_end = url.find("://");
    if (protocol_end == std::string::npos) {
        error_info = "Invalid URL format: " + url;
        return false;
    }
    
    size_t host_start = protocol_end + 3;
    size_t path_start = url.find('/', host_start);
    if (path_start == std::string::npos) {
        error_info = "Invalid URL format: " + url;
        return false;
    }
    
    std::string host = url.substr(host_start, path_start - host_start);
    std::string path = url.substr(path_start);
    
    // Create parent directories
    std::error_code ec;
    std::filesystem::create_directories(destination_path.parent_path(), ec);
    if (ec) {
        error_info = "Failed to create parent directories: " + ec.message();
        return false;
    }
    
    // Create HTTP client
    httplib::SSLClient client(host, 443);
    httplib::Headers headers;
    headers.emplace("User-Agent", "MNN-CLI/1.0");
    headers.emplace("Accept", "*/*");
    
    // Open output file
    std::ofstream output(destination_path, std::ios::binary);
    if (!output.is_open()) {
        error_info = "Failed to open output file: " + destination_path.string();
        return false;
    }
    
    // Download progress tracking
    int64_t downloaded = 0;
    int64_t content_length = expected_size;
    
    // Perform the download
    auto res = client.Get(path, headers,
        [&](const httplib::Response& response) -> bool {
            // Handle response headers
            auto content_length_str = response.get_header_value("Content-Length");
            if (!content_length_str.empty()) {
                content_length = std::stoll(content_length_str);
            }
            return true;
        },
        [&](const char* data, size_t data_length) -> bool {
            // Write data to file
            output.write(data, data_length);
            downloaded += data_length;
            
            // Show progress
            if (content_length > 0) {
                // Use unified progress reporting
                NotifyDownloadProgress(model_id, "file", file_name, downloaded, content_length);
            }
            return true;
        }
    );
    
    output.close();
    
    if (!res || res->status < 200 || res->status >= 300) {
        error_info = "Download failed for " + file_name + " with status: " + 
                    (res ? std::to_string(res->status) : "no response");
        return false;
    }
    
    // Verify file size
    if (std::filesystem::exists(destination_path)) {
        int64_t actual_size = std::filesystem::file_size(destination_path);
        if (actual_size != expected_size) {
            error_info = "File size mismatch for " + file_name + ": expected " + 
                        std::to_string(expected_size) + ", got " + std::to_string(actual_size);
            return false;
        }
    }
    
    std::cout << "Downloaded: " << file_name << " (" << (downloaded / (1024 * 1024)) << " MB)" << std::endl;
    
    return true;
}

bool MlModelDownloader::DeleteRepoImpl(const std::string& model_id) {
    auto ml_model_id = model_id;
    if (ml_model_id.find("ml:") == 0) {
        ml_model_id = ml_model_id.substr(3);
    }
    
    auto ml_repo_folder_name = FileUtils::RepoFolderName(ml_model_id, "model");
    auto ml_storage_folder = std::filesystem::path(cache_root_path_) / ml_repo_folder_name;
    
    std::cout << "removeStorageFolder: " << ml_storage_folder.string() << std::endl;
    
    if (std::filesystem::exists(ml_storage_folder)) {
        std::error_code ec;
        std::filesystem::remove_all(ml_storage_folder, ec);
        if (ec) {
            std::cerr << "remove storageFolder " << ml_storage_folder.string() << " failed: " << ec.message() << std::endl;
            return false;
        }
    }
    
    auto ml_link_folder = GetDownloadPath(model_id);
    std::cout << "removeMlLinkFolder: " << ml_link_folder.string() << std::endl;
    
    if (std::filesystem::exists(ml_link_folder)) {
        std::filesystem::remove(ml_link_folder);
    }
    
    return true;
}

int64_t MlModelDownloader::GetRepoSizeWithError(const std::string& model_id, std::string& error_info) {
    // Get repository info to calculate size
    std::string modelers_id = model_id;
    if (modelers_id.find("ml:") == 0) {
        modelers_id = modelers_id.substr(3);
    }
    
    MlRepoInfo repo_info = ml_api_client_->GetRepoInfo(modelers_id, "main", error_info);
    if (!error_info.empty()) {
        return 0;
    }
    
    // Calculate total size of all files (excluding directories)
    int64_t total_size = 0;
    for (const auto& file : repo_info.files) {
        if (file.type != "dir") {
            total_size += file.size;
        }
    }
    
    return total_size;
}

bool MlModelDownloader::CheckUpdateWithError(const std::string& model_id, std::string& error_info) {
    // For now, just get repository info to check for updates
    // In a full implementation, this would compare timestamps
    std::string modelers_id = model_id;
    if (modelers_id.find("ml:") == 0) {
        modelers_id = modelers_id.substr(3);
    }
    
    MlRepoInfo repo_info = ml_api_client_->GetRepoInfo(modelers_id, "main", error_info);
    return error_info.empty();
}

std::string MlModelDownloader::GetCachePathRoot(const std::string& model_download_path_root) {
    return model_download_path_root + "/modelers";
}

std::filesystem::path MlModelDownloader::GetModelPath(const std::string& models_download_path_root, const std::string& model_id) {
    return std::filesystem::path(models_download_path_root) / FileUtils::GetFileName(model_id);
}

void MlModelDownloader::ShowProgressBar(double progress, int64_t downloaded_size, int64_t total_size) {
    // Not used anymore with NotifyDownloadProgress
}

void MlModelDownloader::ShowFileDownloadProgress(const std::string& file_name, double percentage, int64_t downloaded_size, int64_t total_size) {
    // Not used anymore with NotifyDownloadProgress
}

} // namespace mnn::downloader
