//
// Created by ruoyi.sjd on 2024/12/18.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#include "ms_model_downloader.hpp"
#include "log_utils.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <httplib.h>
#include <iomanip>

namespace mnncli {

MsModelDownloader::MsModelDownloader(const std::string& cache_root_path)
    : ModelRepoDownloader(GetCachePathRoot(cache_root_path)) {
    ms_api_client_ = std::make_unique<MsApiClient>();
}

// Implement pure virtual functions from ModelRepoDownloader base class
void MsModelDownloader::download(const std::string& model_id) {
    std::string error_info;
    if (!DownloadModel(model_id, error_info, mnncli::LogUtils::isVerbose())) {
        notifyDownloadFailed(model_id, error_info);
    } else {
        notifyDownloadFinished(model_id, GetDownloadPath(model_id).string());
    }
}

void MsModelDownloader::pause(const std::string& model_id) {
    addPausedModel(model_id);
    notifyDownloadPaused(model_id);
}

void MsModelDownloader::resume(const std::string& model_id) {
    removePausedModel(model_id);
    download(model_id);
}

std::filesystem::path MsModelDownloader::getDownloadPath(const std::string& model_id) {
    return GetDownloadPath(model_id);
}

bool MsModelDownloader::deleteRepo(const std::string& model_id) {
    return DeleteRepo(model_id);
}

int64_t MsModelDownloader::getRepoSize(const std::string& model_id) {
    std::string error_info;
    return GetRepoSize(model_id, error_info);
}

bool MsModelDownloader::checkUpdate(const std::string& model_id) {
    std::string error_info;
    return CheckUpdate(model_id, error_info);
}

bool MsModelDownloader::DownloadModel(const std::string& model_id, std::string& error_info, bool verbose) {
    if (verbose) {
        LOG_DEBUG_TAG("MsModelDownloader download: " + model_id, "MsModelDownloader");
    }
    
    try {
        return DownloadMsRepo(model_id, error_info);
    } catch (const std::exception& e) {
        error_info = "Download failed: " + std::string(e.what());
        return false;
    }
}

bool MsModelDownloader::DownloadMsRepo(const std::string& model_id, std::string& error_info) {
    // Get the ModelScope repository path from model_id
    // Assuming model_id is in format "owner/repo" or "ms:owner/repo"
    std::string model_scope_id = model_id;
    if (model_scope_id.find("ModelScope/") == 0) {
        model_scope_id = model_scope_id.substr(11);
    }
    
    if (mnncli::LogUtils::isVerbose()) {
        LOG_DEBUG_TAG("MsModelDownloader downloadMsRepo: " + model_id + " modelScopeId: " + model_scope_id, "MsModelDownloader");
    }
    
    // Parse owner/repo from the repository path
    size_t slash_pos = model_scope_id.find('/');
    if (slash_pos == std::string::npos) {
        error_info = "Invalid model ID format, expected: owner/repo";
        return false;
    }
    
    std::string owner = model_scope_id.substr(0, slash_pos);
    std::string repo = model_scope_id.substr(slash_pos + 1);
    
    // Get repository information
    MsRepoInfo repo_info = ms_api_client_->GetRepoInfo(model_scope_id, "main", error_info);
    if (!error_info.empty()) {
        return false;
    }
    
    if (mnncli::LogUtils::isVerbose()) {
        LOG_DEBUG_TAG("downloadMsRepo repoInfo: " + repo_info.model_id, "MsModelDownloader");
    }
    
    return DownloadMsRepoInner(model_id, model_scope_id, repo_info, error_info);
}

bool MsModelDownloader::DownloadMsRepoInner(const std::string& model_id, const std::string& model_scope_id, 
                                           const MsRepoInfo& ms_repo_info, std::string& error_info) {
    if (mnncli::LogUtils::isVerbose()) {
        LOG_DEBUG_TAG("downloadMsRepoInner", "MsModelDownloader");
    }
    
    // Check if already downloaded
    auto folder_link_file = GetModelPath(cache_root_path_, model_id);
    if (std::filesystem::exists(folder_link_file)) {
        if (mnncli::LogUtils::isVerbose()) {
            std::cout << "downloadMsRepoInner already exists at " << folder_link_file.string() << std::endl;
        }
        return true;
    }
    
    // Create storage folder structure
    auto repo_folder_name = FileUtils::RepoFolderName(model_scope_id, "model");
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
    auto download_tasks = CollectMsTaskList(model_scope_id, storage_folder, parent_pointer_path, 
                                           ms_repo_info, total_size, downloaded_size);
    
    if (mnncli::LogUtils::isVerbose()) {
        std::cout << "downloadMsRepoInner downloadTaskList: " << download_tasks.size() << std::endl;
    }
    
    // Download each file
    bool has_error = false;
    for (const auto& task : download_tasks) {
        const std::string& file_path = task.first;
        const std::filesystem::path& destination_path = task.second;
        
        // Find file info
        auto it = std::find_if(ms_repo_info.files.begin(), ms_repo_info.files.end(),
                              [&file_path](const MsFileInfo& f) { return f.path == file_path; });
        
        if (it == ms_repo_info.files.end()) {
            if (mnncli::LogUtils::isVerbose()) {
                std::cerr << "File info not found for: " << file_path << std::endl;
            }
            continue;
        }
        
        // Create download URL for ModelScope
        std::string download_url = "https://modelscope.cn/api/v1/models/" + model_scope_id + "/repo";
        download_url += "?Revision=master&FilePath=" + file_path;
        
        if (mnncli::LogUtils::isVerbose()) {
            std::cout << "Downloading file: " << file_path << " (" << it->size << " bytes)" << std::endl;
        }
        
        if (!DownloadFile(download_url, destination_path, it->size, file_path, error_info)) {
            has_error = true;
            break;
        }
        
        downloaded_size += it->size;
        
        // Show progress
        if (mnncli::LogUtils::isVerbose()) {
            double progress = (static_cast<double>(downloaded_size) / total_size) * 100.0;
            ShowProgressBar(progress, downloaded_size, total_size);
        }
    }
    
    if (mnncli::LogUtils::isVerbose()) {
        std::cout << std::endl; // New line after progress
    }
    
    if (!has_error) {
        // Create the main symlink
        std::filesystem::create_symlink(parent_pointer_path, folder_link_file, ec);
        if (ec) {
            if (mnncli::LogUtils::isVerbose()) {
                std::cerr << "Failed to create main symlink: " << ec.message() << std::endl;
            }
            error_info = "Failed to create symlink: " + ec.message();
            return false;
        }
        
        if (mnncli::LogUtils::isVerbose()) {
            std::cout << "✅ Model downloaded successfully to " << folder_link_file.string() << std::endl;
        }
        return true;
    } else {
        if (mnncli::LogUtils::isVerbose()) {
            std::cerr << "❌ Download failed with errors" << std::endl;
        }
        return false;
    }
}

std::vector<std::pair<std::string, std::filesystem::path>> MsModelDownloader::CollectMsTaskList(
    const std::string& model_scope_id,
    const std::filesystem::path& storage_folder,
    const std::filesystem::path& parent_pointer_path,
    const MsRepoInfo& ms_repo_info,
    int64_t& total_size,
    int64_t& downloaded_size) {
    
    std::vector<std::pair<std::string, std::filesystem::path>> file_download_tasks;
    
    for (const auto& sub_file : ms_repo_info.files) {
        if (sub_file.type == "tree") {
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

bool MsModelDownloader::DownloadFile(const std::string& url, const std::filesystem::path& destination_path, 
                                    int64_t expected_size, const std::string& file_name, std::string& error_info) {
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
    
    // Handle redirects first - check for redirect status codes (301, 302, 303, 307, 308)
    std::string final_url = url;
    std::string final_host = host;
    std::string final_path = path;
    
    // Create HTTP client for redirect check
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    httplib::SSLClient redirect_client(host, 443);
#else
    httplib::Client redirect_client(host, 80);
#endif
    httplib::Headers redirect_headers;
    redirect_headers.emplace("User-Agent", "MNN-CLI/1.0");
    redirect_headers.emplace("Accept", "*/*");
    
    // Check for redirects first
    auto redirect_res = redirect_client.Get(path, redirect_headers);
    if (redirect_res && (redirect_res->status >= 301 && redirect_res->status <= 308)) {
        // Handle redirect
        auto location_header = redirect_res->get_header_value("Location");
        if (!location_header.empty()) {
            if (mnncli::LogUtils::isVerbose()) {
                std::cout << "Redirect detected: " << redirect_res->status << " -> " << location_header << std::endl;
            }
            
            // Parse the new URL
            if (location_header.find("http") == 0) {
                // Absolute URL
                size_t new_protocol_end = location_header.find("://");
                if (new_protocol_end != std::string::npos) {
                    size_t new_host_start = new_protocol_end + 3;
                    size_t new_path_start = location_header.find('/', new_host_start);
                    if (new_path_start != std::string::npos) {
                        final_host = location_header.substr(new_host_start, new_path_start - new_host_start);
                        final_path = location_header.substr(new_path_start);
                        final_url = location_header;
                    }
                }
            } else if (location_header[0] == '/') {
                // Absolute path on same host
                final_path = location_header;
                final_url = "https://" + host + location_header;
            } else {
                // Relative path
                size_t last_slash = path.find_last_of('/');
                if (last_slash != std::string::npos) {
                    std::string base_path = path.substr(0, last_slash + 1);
                    final_path = base_path + location_header;
                    final_url = "https://" + host + final_path;
                }
            }
            
            if (mnncli::LogUtils::isVerbose()) {
                std::cout << "Following redirect to: " << final_url << std::endl;
            }
        }
    }
    
    // Create parent directories
    std::error_code ec;
    std::filesystem::create_directories(destination_path.parent_path(), ec);
    if (ec) {
        error_info = "Failed to create parent directories: " + ec.message();
        return false;
    }
    
    // Create HTTP client for actual download
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    httplib::SSLClient client(final_host, 443);
#else
    httplib::Client client(final_host, 80);
#endif
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
    auto res = client.Get(final_path, headers,
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
                double percentage = (static_cast<double>(downloaded) / content_length) * 100.0;
                ShowFileDownloadProgress(file_name, percentage, downloaded, content_length);
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
    
    if (mnncli::LogUtils::isVerbose()) {
        std::cout << std::endl; // New line after progress
    }
    if (mnncli::LogUtils::isVerbose()) {
        std::cout << "Downloaded: " << file_name << " (" << (downloaded / (1024 * 1024)) << " MB)" << std::endl;
    }
    
    return true;
}

std::filesystem::path MsModelDownloader::GetDownloadPath(const std::string& model_id) {
    return GetModelPath(cache_root_path_, model_id);
}

bool MsModelDownloader::DeleteRepo(const std::string& model_id) {
    auto ms_model_id = model_id;
    if (ms_model_id.find("ms:") == 0) {
        ms_model_id = ms_model_id.substr(3);
    }
    
    auto ms_repo_folder_name = FileUtils::RepoFolderName(ms_model_id, "model");
    auto ms_storage_folder = std::filesystem::path(cache_root_path_) / ms_repo_folder_name;
    
    if (mnncli::LogUtils::isVerbose()) {
        std::cout << "removeStorageFolder: " << ms_storage_folder.string() << std::endl;
    }
    
    if (std::filesystem::exists(ms_storage_folder)) {
        std::error_code ec;
        std::filesystem::remove_all(ms_storage_folder, ec);
        if (ec) {
            if (mnncli::LogUtils::isVerbose()) {
                std::cerr << "remove storageFolder " << ms_storage_folder.string() << " failed: " << ec.message() << std::endl;
            }
            return false;
        }
    }
    
    auto ms_link_folder = GetDownloadPath(model_id);
    if (mnncli::LogUtils::isVerbose()) {
        std::cout << "removeMsLinkFolder: " << ms_link_folder.string() << std::endl;
    }
    
    if (std::filesystem::exists(ms_link_folder)) {
        std::filesystem::remove(ms_link_folder);
    }
    
    return true;
}

int64_t MsModelDownloader::GetRepoSize(const std::string& model_id, std::string& error_info) {
    // Get repository info to calculate size
    std::string model_scope_id = model_id;
    if (model_scope_id.find("ms:") == 0) {
        model_scope_id = model_scope_id.substr(3);
    }
    
    MsRepoInfo repo_info = ms_api_client_->GetRepoInfo(model_scope_id, "main", error_info);
    if (!error_info.empty()) {
        return 0;
    }
    
    // Calculate total size of all files (excluding directories)
    int64_t total_size = 0;
    for (const auto& file : repo_info.files) {
        if (file.type != "tree") {
            total_size += file.size;
        }
    }
    
    return total_size;
}

bool MsModelDownloader::CheckUpdate(const std::string& model_id, std::string& error_info) {
    // For now, just get repository info to check for updates
    // In a full implementation, this would compare timestamps
    std::string model_scope_id = model_id;
    if (model_scope_id.find("ms:") == 0) {
        model_scope_id = model_scope_id.substr(3);
    }
    
    MsRepoInfo repo_info = ms_api_client_->GetRepoInfo(model_scope_id, "main", error_info);
    return error_info.empty();
}

std::string MsModelDownloader::GetCachePathRoot(const std::string& model_download_path_root) {
    return model_download_path_root + "/modelscope";
}

std::filesystem::path MsModelDownloader::GetModelPath(const std::string& models_download_path_root, const std::string& model_id) {
    return std::filesystem::path(models_download_path_root) / FileUtils::GetFileName(model_id);
}

void MsModelDownloader::ShowProgressBar(double progress, int64_t downloaded_size, int64_t total_size) {
    const int bar_width = 50;
    int pos = static_cast<int>(bar_width * progress / 100.0);
    
    // Clear the line and show progress bar
    std::cout << "\r[";
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) {
            std::cout << "█";  // Filled block
        } else if (i == pos) {
            std::cout << "▶";  // Arrow
        } else {
            std::cout << "░";  // Light block
        }
    }
    std::cout << "] ";
    
    // Show percentage and size info
    std::cout << std::fixed << std::setprecision(1) << progress << "% ";
    std::cout << "(" << (downloaded_size / (1024 * 1024)) << " MB / " << (total_size / (1024 * 1024)) << " MB)" << std::flush;
}

void MsModelDownloader::ShowFileDownloadProgress(const std::string& file_name, double percentage, int64_t downloaded_size, int64_t total_size) {
    const int bar_width = 40;
    int pos = static_cast<int>(bar_width * percentage / 100.0);
    
    // Clear the line and show progress bar with file name
    std::cout << "\r" << file_name << " [";
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) {
            std::cout << "█";  // Filled block
        } else if (i == pos) {
            std::cout << "▶";  // Arrow
        } else {
            std::cout << "░";  // Light block
        }
    }
    std::cout << "] ";
    
    // Show percentage and size info
    std::cout << std::fixed << std::setprecision(1) << percentage << "% ";
    std::cout << "(" << (downloaded_size / (1024 * 1024)) << " MB)" << std::flush;
}

} // namespace mnncli
