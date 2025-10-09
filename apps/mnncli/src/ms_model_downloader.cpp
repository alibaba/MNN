//
// Created by ruoyi.sjd on 2024/12/18.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#include "ms_model_downloader.hpp"
#include "log_utils.hpp"
#include "user_interface.hpp"
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
void MsModelDownloader::Download(const std::string& model_id) {
    std::string error_info;
    if (!DownloadModel(model_id, error_info, mnncli::LogUtils::IsVerbose())) {
        NotifyDownloadFailed(model_id, error_info);
    } else {
        NotifyDownloadFinished(model_id, GetDownloadPath(model_id).string());
    }
}

void MsModelDownloader::Pause(const std::string& model_id) {
    AddPausedModel(model_id);
    NotifyDownloadPaused(model_id);
}

void MsModelDownloader::Resume(const std::string& model_id) {
    RemovePausedModel(model_id);
    Download(model_id);
}

std::filesystem::path MsModelDownloader::GetDownloadPath(const std::string& model_id) {
    // Parse model_id in format "ModelScope/MNN/Hunyuan-7B-Instruct-MNN"
    // Extract the repo path part and create modelscope/MNN/Hunyuan-7B-Instruct-MNN
    
    std::string repo_path;
    
    // Handle different model_id formats
    if (model_id.find("ModelScope/") == 0) {
        // Format: "ModelScope/MNN/Hunyuan-7B-Instruct-MNN"
        repo_path = model_id.substr(11); // Remove "ModelScope/"
    } else if (model_id.find("ms:") == 0) {
        // Format: "ms:MNN/Hunyuan-7B-Instruct-MNN"
        repo_path = model_id.substr(3); // Remove "ms:"
    } else {
        // Assume it's already in owner/repo format
        repo_path = model_id;
    }
    
    // Create path: modelscope/MNN/Hunyuan-7B-Instruct-MNN
    return std::filesystem::path(cache_root_path_) / repo_path;
}

bool MsModelDownloader::DeleteRepo(const std::string& model_id) {
    return DeleteRepoImpl(model_id);
}

int64_t MsModelDownloader::GetRepoSize(const std::string& model_id) {
    std::string error_info;
    return GetRepoSizeWithError(model_id, error_info);
}

bool MsModelDownloader::CheckUpdate(const std::string& model_id) {
    std::string error_info;
    return CheckUpdateWithError(model_id, error_info);
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
    
    if (mnncli::LogUtils::IsVerbose()) {
        LOG_DEBUG_TAG("MsModelDownloader downloadMsRepo: " + model_id + " modelScopeId: " + model_scope_id, "MsModelDownloader");
    }
    
    // Parse owner/repo from the repository path
    size_t slash_pos = model_scope_id.find('/');
    if (slash_pos == std::string::npos) {
        error_info = "Invalid model ID format, expected: owner/repo (e.g., 'Qwen/Qwen-7B-Chat')";
        return false;
    }
    
    std::string owner = model_scope_id.substr(0, slash_pos);
    std::string repo = model_scope_id.substr(slash_pos + 1);
    
    // Get repository information
    MsRepoInfo repo_info = ms_api_client_->GetRepoInfo(model_scope_id, "main", error_info);
    if (!error_info.empty()) {
        return false;
    }
    
    if (mnncli::LogUtils::IsVerbose()) {
        LOG_DEBUG_TAG("downloadMsRepo repoInfo: " + repo_info.model_id, "MsModelDownloader");
    }
    
    return DownloadMsRepoInner(model_id, model_scope_id, repo_info, error_info);
}

bool MsModelDownloader::DownloadMsRepoInner(const std::string& model_id, const std::string& model_scope_id, 
                                           const MsRepoInfo& ms_repo_info, std::string& error_info) {
    if (mnncli::LogUtils::IsVerbose()) {
        LOG_DEBUG_TAG("downloadMsRepoInner", "MsModelDownloader");
    }
    
    // Check if already downloaded
    auto model_folder = GetDownloadPath(model_id);
    if (std::filesystem::exists(model_folder)) {
        if (mnncli::LogUtils::IsVerbose()) {
            std::cout << "downloadMsRepoInner already exists at " << model_folder.string() << std::endl;
        }
        return true;
    }
    
    // Create direct storage folder structure 
    // Use GetDownloadPath to get the correct path: modelscope/MNN/Hunyuan-7B-Instruct-MNN
    auto direct_model_folder = GetDownloadPath(model_id);
    
    LOG_DEBUG_TAG("Model folder (direct storage): " + direct_model_folder.string(), "MsModelDownloader");
    
    // Create necessary directories
    std::error_code ec;
    std::filesystem::create_directories(direct_model_folder, ec);
    
    if (ec) {
        error_info = "Failed to create directories: " + ec.message();
        return false;
    }
    
    // Collect download tasks
    int64_t total_size = 0;
    int64_t downloaded_size = 0;
    auto download_tasks = CollectMsTaskList(model_id, direct_model_folder, 
                                           ms_repo_info, total_size, downloaded_size);
    
    if (mnncli::LogUtils::IsVerbose()) {
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
            if (mnncli::LogUtils::IsVerbose()) {
                std::cerr << "File info not found for: " << file_path << std::endl;
            }
            continue;
        }
        
        // Create download URL for ModelScope
        std::string download_url = "https://modelscope.cn/api/v1/models/" + model_scope_id + "/repo";
        download_url += "?Revision=master&FilePath=" + file_path;
        
        if (mnncli::LogUtils::IsVerbose()) {
            std::cout << "Downloading file: " << file_path << " (" << it->size << " bytes)" << std::endl;
        }
        
        if (!DownloadFile(download_url, destination_path, it->size, file_path, error_info)) {
            has_error = true;
            break;
        }
        
        downloaded_size += it->size;
        
        // Show progress using unified system
        if (mnncli::LogUtils::IsVerbose()) {
            float progress = static_cast<float>(downloaded_size) / total_size;
            std::string size_info = " (" + std::to_string(downloaded_size / (1024 * 1024)) + " MB / " + 
                                   std::to_string(total_size / (1024 * 1024)) + " MB)";
            std::string message = "Overall progress" + size_info;
            mnncli::UserInterface::ShowProgress(message, progress);
        }
    }
    
    if (mnncli::LogUtils::IsVerbose()) {
        std::cout << std::endl; // New line after progress
    }
    
    if (!has_error) {
        // Download completed - notify (no symlink needed with direct storage)
        auto model_path = GetDownloadPath(model_id);
        LOG_DEBUG_TAG("Download completed at: " + model_path.string(), "MsModelDownloader");
        if (mnncli::LogUtils::IsVerbose()) {
            std::cout << "✅ Model downloaded successfully to " << model_path.string() << std::endl;
        }
        return true;
    } else {
        if (mnncli::LogUtils::IsVerbose()) {
            std::cerr << "❌ Download failed with errors" << std::endl;
        }
        return false;
    }
}

std::vector<std::pair<std::string, std::filesystem::path>> MsModelDownloader::CollectMsTaskList(
    const std::string& model_id,
    const std::filesystem::path& model_folder,
    const MsRepoInfo& ms_repo_info,
    int64_t& total_size,
    int64_t& downloaded_size) {
    
    std::vector<std::pair<std::string, std::filesystem::path>> file_download_tasks;
    
    LOG_DEBUG_TAG("CollectMsTaskList: Processing " + std::to_string(ms_repo_info.files.size()) + " files", "MsModelDownloader");
    
    for (const auto& sub_file : ms_repo_info.files) {
        if (sub_file.type == "tree") {
            continue; // Skip directories
        }
        
        // Simplified: direct download path (HuggingFace style)
        // model_folder/filename (no blobs or snapshots)
        auto direct_path = model_folder / sub_file.path;
        
        // Check if already downloaded
        int64_t file_downloaded_size = 0;
        if (std::filesystem::exists(direct_path)) {
            file_downloaded_size = std::filesystem::file_size(direct_path);
        } else {
            // Check for incomplete file (.incomplete suffix)
            auto incomplete_path = std::filesystem::path(direct_path.string() + ".incomplete");
            if (std::filesystem::exists(incomplete_path)) {
                file_downloaded_size = std::filesystem::file_size(incomplete_path);
            }
        }
        
        total_size += sub_file.size;
        downloaded_size += file_downloaded_size;
        
        LOG_DEBUG_TAG("  Added task: " + sub_file.path + 
                     " -> path: " + direct_path.string(), "MsModelDownloader");
        
        file_download_tasks.emplace_back(sub_file.path, direct_path);
    }
    
    LOG_DEBUG_TAG("CollectMsTaskList: Created " + std::to_string(file_download_tasks.size()) + " tasks", "MsModelDownloader");
    
    return file_download_tasks;
}

bool MsModelDownloader::DownloadFile(const std::string& url, const std::filesystem::path& destination_path, 
                                    int64_t expected_size, const std::string& file_name, std::string& error_info) {
    // Check if file already exists and is complete
    if (std::filesystem::exists(destination_path)) {
        int64_t existing_size = std::filesystem::file_size(destination_path);
        if (existing_size == expected_size) {
            if (mnncli::LogUtils::IsVerbose()) {
                std::cout << "File already downloaded: " << file_name << std::endl;
            }
            return true;
        }
    }
    
    // Check for incomplete file (.incomplete suffix)
    auto incomplete_path = std::filesystem::path(destination_path.string() + ".incomplete");
    int64_t resume_from = 0;
    
    if (std::filesystem::exists(incomplete_path)) {
        resume_from = std::filesystem::file_size(incomplete_path);
        if (mnncli::LogUtils::IsVerbose()) {
            std::cout << "Resuming download from " << (resume_from / (1024 * 1024)) << " MB" << std::endl;
        }
    }
    
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
            if (mnncli::LogUtils::IsVerbose()) {
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
            
            if (mnncli::LogUtils::IsVerbose()) {
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
    
    // Add Range header for resume
    if (resume_from > 0) {
        headers.emplace("Range", "bytes=" + std::to_string(resume_from) + "-");
    }
    
    // Open output file in append mode (for resume)
    std::ofstream output;
    if (resume_from > 0) {
        output.open(incomplete_path, std::ios::binary | std::ios::app);
    } else {
        output.open(incomplete_path, std::ios::binary);
    }
    
    if (!output.is_open()) {
        error_info = "Failed to open output file: " + incomplete_path.string();
        return false;
    }
    
    // Download progress tracking (start from resume point)
    int64_t downloaded = resume_from;
    int64_t content_length = expected_size;
    
    // Perform the download
    auto res = client.Get(final_path, headers,
        [&](const httplib::Response& response) -> bool {
            // Handle response headers
            // NOTE: When using Range header, Content-Length is the REMAINING bytes, not total file size
            // We should use expected_size as the total, not Content-Length from response
            auto content_length_str = response.get_header_value("Content-Length");
            if (!content_length_str.empty()) {
                int64_t remaining_bytes = std::stoll(content_length_str);
                if (mnncli::LogUtils::IsVerbose()) {
                    std::cout << "Response Content-Length: " << remaining_bytes 
                             << " bytes (remaining to download)" << std::endl;
                }
            }
            // Use expected_size as content_length for progress calculation
            content_length = expected_size;
            return true;
        },
        [&](const char* data, size_t data_length) -> bool {
            // Write data to file
            output.write(data, data_length);
            downloaded += data_length;
            
            // Show progress using unified system with smart unit selection
            // Use expected_size (total file size) for progress calculation, not Content-Length
            if (expected_size > 0) {
                float progress = static_cast<float>(downloaded) / expected_size;
                
                // Use parent class utility methods for formatting
                std::string display_name = ModelRepoDownloader::ExtractFileName(file_name);
                std::string size_info = ModelRepoDownloader::FormatFileSizeInfo(downloaded, expected_size);
                
                std::string message = display_name + size_info;
                mnncli::UserInterface::ShowProgress(message, progress);
            }
            return true;
        }
    );
    
    output.close();
    
    // Check for partial content status (206) which is expected for resume
    if (!res || (res->status < 200 || res->status >= 300) && res->status != 206) {
        error_info = "Download failed for " + file_name + " with status: " + 
                    (res ? std::to_string(res->status) : "no response");
        return false;
    }
    
    // Verify file size of incomplete file
    if (std::filesystem::exists(incomplete_path)) {
        int64_t actual_size = std::filesystem::file_size(incomplete_path);
        if (actual_size == expected_size) {
            // Download complete, move from .incomplete to final destination
            std::error_code ec;
            std::filesystem::rename(incomplete_path, destination_path, ec);
            if (ec) {
                error_info = "Failed to move file from incomplete to final destination: " + ec.message();
                return false;
            }
            if (mnncli::LogUtils::IsVerbose()) {
                std::cout << "Download complete, moved to: " << destination_path.string() << std::endl;
            }
            return true;
        } else if (actual_size > expected_size) {
            error_info = "File size exceeds expected size for " + file_name + ": expected " + 
                        std::to_string(expected_size) + ", got " + std::to_string(actual_size);
            // Remove corrupted file
            std::filesystem::remove(incomplete_path);
            return false;
        } else {
            // Incomplete download, will be resumed next time
            if (mnncli::LogUtils::IsVerbose()) {
                std::cout << "Download incomplete (" << actual_size << " / " << expected_size << " bytes), saved for resume" << std::endl;
            }
            error_info = "Download incomplete, please run again to resume";
            return false;
        }
    }
    
    // If we get here, something went wrong
    error_info = "Download failed: incomplete file not found";
    return false;
}

bool MsModelDownloader::DeleteRepoImpl(const std::string& model_id) {
    // Simplified: delete model folder directly (HuggingFace style)
    auto model_folder = GetDownloadPath(model_id);
    
    LOG_INFO("Removing model folder: " + model_folder.string());
    
    if (std::filesystem::exists(model_folder)) {
        std::error_code ec;
        std::filesystem::remove_all(model_folder, ec);
        if (ec) {
            LOG_ERROR("Failed to remove model folder " + model_folder.string() + ": " + ec.message());
            return false;
        }
    }
    
    return true;
}

int64_t MsModelDownloader::GetRepoSizeWithError(const std::string& model_id, std::string& error_info) {
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

bool MsModelDownloader::CheckUpdateWithError(const std::string& model_id, std::string& error_info) {
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


} // namespace mnncli
