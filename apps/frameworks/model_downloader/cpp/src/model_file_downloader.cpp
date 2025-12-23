//
// Created by ruoyi.sjd on 2024/12/18.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#include "model_file_downloader.hpp"
#include "httplib.h"
#include "file_utils.hpp"
#include "hf_sha_verifier.hpp"
#include "log_utils.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cstring>
#include <functional>
#include <chrono>
#include <thread>
#include <mutex>

namespace mnn::downloader
{

    // Constructor matching Android OkHttpClient configuration
    ModelFileDownloader::ModelFileDownloader() : client_("", 443) {
        // Note: client_ is no longer used for downloads, but kept for compatibility
        // Each download will create its own client with the correct host
    }


    // Main download method - simplified for direct storage (ModelScope style)
    // Based on Android ModelFileDownloader.downloadFile() but without symlinks
    void ModelFileDownloader::DownloadFile(FileDownloadTask& fileDownloadTask, FileDownloadListener& fileDownloadListener)
    {
        LOG_DEBUG_TAG("Starting download for file: " + fileDownloadTask.relativePath, "ModelFileDownloader");
        
        // Simplified: only one path to validate and create
        if (fileDownloadTask.downloadPath.empty()) {
            throw FileDownloadException("Invalid download path");
        }
        
        // Create parent directory
        try {
            std::filesystem::create_directories(fileDownloadTask.downloadPath.parent_path());
            LOG_DEBUG_TAG("Directory created: " + fileDownloadTask.downloadPath.parent_path().string(), "ModelFileDownloader");
        } catch (const std::exception& e) {
            LOG_DEBUG_TAG("Error during directory creation: " + std::string(e.what()), "ModelFileDownloader");
            throw FileDownloadException("Failed to create directory: " + std::string(e.what()));
        }

        // Check if file already exists at final location
        if (std::filesystem::exists(fileDownloadTask.downloadPath)) {
            LOG_DEBUG("File already exists: " + fileDownloadTask.relativePath);
            return;
        }

        // Download the file directly to final location
        LOG_DEBUG_TAG("Starting download process", "ModelFileDownloader");
        try {
            std::mutex lock;
            {
                std::lock_guard<std::mutex> guard(lock);
                fs::path incompletePath = fileDownloadTask.GetIncompletePath();
                
                DownloadToTmpAndMove(
                    fileDownloadTask, 
                    incompletePath,                          // Temporary file (.incomplete)
                    fileDownloadTask.downloadPath,           // Final destination
                    fileDownloadTask.fileMetadata.location,  // Download URL
                    fileDownloadTask.fileMetadata.size,      // Expected size
                    fileDownloadTask.relativePath,           // Display name
                    fileDownloadListener);
                
                LOG_DEBUG_TAG("Download completed successfully: " + fileDownloadTask.downloadPath.string(), "ModelFileDownloader");
            }
        } catch (const FileDownloadException& e) {
            LOG_DEBUG_TAG("File download error: " + std::string(e.what()), "ModelFileDownloader");
            throw;
        } catch (const std::exception& e) {
            LOG_DEBUG_TAG("Error during download process: " + std::string(e.what()), "ModelFileDownloader");
            throw;
        }
    }

    // downloadToTmpAndMove method matching Android implementation exactly
    void ModelFileDownloader::DownloadToTmpAndMove(
        FileDownloadTask& fileDownloadTask,
        const fs::path& incompletePath,
        const fs::path& destinationPath,
        const std::string& urlToDownload,
        int64_t expectedSize,
        const std::string& fileName,
        FileDownloadListener& fileDownloadListener)
    {
        std::string theUrlToDownload = urlToDownload;
        
        // Check if destination already exists and validate it (like Kotlin)
        if (std::filesystem::exists(destinationPath)) {
            if (Validate(fileDownloadTask, destinationPath)) {
                LOG_DEBUG_TAG("Destination file already exists and is valid: " + destinationPath.string(), "ModelFileDownloader");
                return;
            } else {
                LOG_DEBUG_TAG("Destination file exists but is invalid, removing", "ModelFileDownloader");
                std::filesystem::remove(destinationPath);
                fileDownloadTask.downloadedSize = 0;
            }
        }
        
        // Check if incomplete file exists and read its size for resume (like Android lines 86-94)
        if (std::filesystem::exists(incompletePath)) {
            try {
                int64_t incomplete_size = std::filesystem::file_size(incompletePath);
                
                // If incomplete file size is valid, use it for resume
                if (incomplete_size > 0 && incomplete_size <= expectedSize) {
                    fileDownloadTask.downloadedSize = incomplete_size;
                    LOG_DEBUG_TAG("Found incomplete file with size: " + std::to_string(incomplete_size) + 
                                 " bytes, will resume download", "ModelFileDownloader");
                } else if (incomplete_size >= expectedSize) {
                    // File is complete, validate it
                    LOG_DEBUG_TAG("Incomplete file size >= expected size, validating", "ModelFileDownloader");
                    if (Validate(fileDownloadTask, incompletePath)) {
                        LOG_DEBUG_TAG("Incomplete file is valid, moving to destination", "ModelFileDownloader");
                        MoveWithPermissions(incompletePath, destinationPath);
                        return;
                    } else {
                        LOG_DEBUG_TAG("Incomplete file validation failed, removing and restarting", "ModelFileDownloader");
                        std::filesystem::remove(incompletePath);
                        fileDownloadTask.downloadedSize = 0;
                    }
                } else {
                    LOG_DEBUG_TAG("Incomplete file has invalid size: " + std::to_string(incomplete_size) + 
                                 ", removing", "ModelFileDownloader");
                    std::filesystem::remove(incompletePath);
                    fileDownloadTask.downloadedSize = 0;
                }
            } catch (const std::exception& e) {
                LOG_DEBUG_TAG("Error reading incomplete file size: " + std::string(e.what()) + 
                             ", starting from scratch", "ModelFileDownloader");
                fileDownloadTask.downloadedSize = 0;
            }
        } else {
            LOG_DEBUG_TAG("No incomplete file found, starting fresh download", "ModelFileDownloader");
            fileDownloadTask.downloadedSize = 0;
        }
        
        // Check for redirects first (like Android does)
        auto [host, path] = HfApiClient::ParseUrl(urlToDownload);
        httplib::Headers headers;
        headers.emplace("User-Agent", "MNN-CLI/1.0");
        headers.emplace("Accept", "*/*");
        
        // Add Hugging Face authentication token ONLY for huggingface.co requests
        // DO NOT add Authorization header for pre-signed CDN URLs
        if (host.find("huggingface.co") != std::string::npos) {
            if (const char* hf_token = std::getenv("HF_TOKEN")) {
                std::string auth_header = "Bearer " + std::string(hf_token);
                headers.emplace("Authorization", auth_header);
            }
        }
        
        auto head_res = client_.Head(path, headers);
        if (head_res && (head_res->status >= 301 && head_res->status <= 308)) {
            std::string location = head_res->get_header_value("Location");
            if (!location.empty()) {
                theUrlToDownload = location;
            }
        }
        
        LOG_DEBUG_TAG("downloadToTmpAndMove urlToDownload: " + theUrlToDownload + " to file: " + incompletePath.string() + " to destination: " + destinationPath.string(), "ModelFileDownloader");
        
        // Download with retry logic like Android (max 10 retries)
        if (fileDownloadTask.downloadedSize < expectedSize) {
            for (int i = 0; i < kMaxRetry; i++) {
                try {
                    LOG_DEBUG_TAG("downloadChunk try the " + std::to_string(i) + " turn", "ModelFileDownloader");
                    
                    DownloadChunk(fileDownloadTask, theUrlToDownload, incompletePath, expectedSize, fileName, &fileDownloadListener);
                    
                    LOG_DEBUG_TAG("downloadChunk try the " + std::to_string(i) + " turn finish", "ModelFileDownloader");
                    
                    // After download, validate the incomplete file (exactly like Kotlin lines 142-145)
                    if (!Validate(fileDownloadTask, incompletePath)) {
                        // If this was the last retry, throw exception
                        if (i == kMaxRetry - 1) {
                            LOG_DEBUG_TAG("Max retries reached after validation failure", "ModelFileDownloader");
                            throw FileDownloadException("File validation failed after max retries");
                        }
                        LOG_DEBUG_TAG("Download validation failed, removing incomplete file and resetting download size", "ModelFileDownloader");
                        std::filesystem::remove(incompletePath);
                        fileDownloadTask.downloadedSize = 0;
                        // Otherwise, retry by continuing the loop
                        LOG_DEBUG_TAG("Retrying download after validation failure (attempt " + std::to_string(i + 2) + "/" + std::to_string(kMaxRetry) + ")", "ModelFileDownloader");
                        std::this_thread::sleep_for(std::chrono::seconds(1));
                        continue;
                    }
                    
                    // Validation passed, break out of retry loop
                    LOG_DEBUG_TAG("Download validation succeeded", "ModelFileDownloader");
                    break;
                } catch (const DownloadPausedException& e) {
                    throw e;
                } catch (const std::exception& e) {
                    if (i == kMaxRetry - 1) {
                        LOG_DEBUG_TAG("Max retries reached, throwing exception: " + std::string(e.what()), "ModelFileDownloader");
                        throw e;
                    } else {
                        LOG_DEBUG_TAG("downloadChunk failed sleep and retrying: " + std::string(e.what()), "ModelFileDownloader");
                        try {
                            std::this_thread::sleep_for(std::chrono::seconds(1));
                        } catch (const std::exception& ex) {
                            LOG_DEBUG_TAG("Error during sleep: " + std::string(ex.what()), "ModelFileDownloader");
                            throw std::runtime_error(ex.what());
                        }
                    }
                }
            }
        }
        
        // Like Kotlin: validate and move (exactly matching Kotlin lines 163-169)
        if (Validate(fileDownloadTask, incompletePath)) {
            LOG_DEBUG_TAG("Validation passed, moving file to destination", "ModelFileDownloader");
            MoveWithPermissions(incompletePath, destinationPath);
        } else {
            LOG_DEBUG_TAG("Final validation failed, removing incomplete file and resetting download size", "ModelFileDownloader");
            std::filesystem::remove(incompletePath);
            fileDownloadTask.downloadedSize = 0;
            throw FileDownloadException("File validation failed after download");
        }
    }

    
    // downloadChunk method matching Android implementation exactly
    void ModelFileDownloader::DownloadChunk(
        FileDownloadTask& fileDownloadTask,
        const std::string& url,
        const fs::path& tempFile,
        int64_t expectedSize,
        const std::string& displayedFilename,
        FileDownloadListener* fileDownloadListener)
    {
        auto [host, path] = HfApiClient::ParseUrl(url);
        
        // Create HTTP client for download
        // For pre-signed URLs (like AWS S3), DO NOT specify port explicitly to avoid 
        // Host header mismatch (e.g., "Host: example.com:443" vs "Host: example.com")
        httplib::SSLClient download_client(host);
        download_client.set_connection_timeout(kConnectTimeoutSeconds, 0);
        download_client.set_read_timeout(kConnectTimeoutSeconds, 0);
        download_client.set_write_timeout(kConnectTimeoutSeconds, 0);
        download_client.enable_server_certificate_verification(false);
        // For pre-signed URLs, we need minimal headers - disable keep-alive to avoid Connection header
        download_client.set_keep_alive(false);
        // Set empty user agent to avoid automatic User-Agent header
        download_client.set_default_headers({});
        
        if (verbose_) {
            LOG_DEBUG("downloadChunk: Created new HTTP client for host: " + host);
        }
        
        httplib::Headers requestHeaders;
        // Always set Accept-Encoding: identity to prevent compression (matching Kotlin exactly)
        requestHeaders.emplace("Accept-Encoding", "identity");

        // Add Hugging Face authentication token ONLY for huggingface.co requests
        // DO NOT add Authorization header for pre-signed CDN URLs
        if (host.find("huggingface.co") != std::string::npos) {
            if (const char* hf_token = std::getenv("HF_TOKEN")) {
                std::string auth_header = "Bearer " + std::string(hf_token);
                requestHeaders.emplace("Authorization", auth_header);
            }
        }
        
        if (fileDownloadTask.downloadedSize >= expectedSize) {
            return;
        }
        
        int64_t downloadedBytes = fileDownloadTask.downloadedSize;

        if (fileDownloadTask.downloadedSize > 0) {
            requestHeaders.emplace("Range", "bytes=" + std::to_string(fileDownloadTask.downloadedSize) + "-");
        }
        
        // Open output file for writing
        std::ofstream output;
        try {
            output.open(tempFile, std::ios::binary | std::ios::app);
            if (!output.is_open()) {
                throw FileDownloadException("Failed to open output file: " + tempFile.string());
            }
        } catch (const std::exception& e) {
            throw FileDownloadException("File operation failed: " + std::string(e.what()));
        }
        
        LOG_DEBUG_TAG("Starting HTTP GET request to: " + url, "ModelFileDownloader");
        auto res = download_client.Get(path, requestHeaders, [&](const httplib::Response &response)
                              {
            LOG_DEBUG_TAG("Received HTTP response with status: " + std::to_string(response.status), "ModelFileDownloader");
            // Return false for non-successful status codes to stop the download
            if (response.status < 200 || response.status >= 300) {
                LOG_DEBUG_TAG("HTTP error status, stopping download: " + std::to_string(response.status), "ModelFileDownloader");
                return false;
            }
            return true; }, [&](const char *data, size_t data_length)
                              {
            output.write(data, data_length);
            downloadedBytes += data_length;
            fileDownloadTask.downloadedSize += data_length;
            
            if (fileDownloadListener != nullptr) {
                bool paused = fileDownloadListener->onDownloadDelta(
                    &displayedFilename, downloadedBytes, expectedSize, data_length);
                if (paused) {
                    throw DownloadPausedException("Download paused");
                }
            }
            return true; });

        LOG_DEBUG_TAG("HTTP request completed", "ModelFileDownloader");
        if (res) {
            LOG_DEBUG_TAG("HTTP response received with status: " + std::to_string(res->status), "ModelFileDownloader");
            if (!(res->status >= 200 && res->status < 300) && res->status != 416) {
                LOG_DEBUG_TAG("HTTP error status: " + std::to_string(res->status), "ModelFileDownloader");
                throw FileDownloadException("HTTP error: " + std::to_string(res->status));
            }
        } else {
            LOG_DEBUG_TAG("HTTP request failed with error: " + std::string(httplib::to_string(res.error())), "ModelFileDownloader");
            throw FileDownloadException("Connection error: " + std::string(httplib::to_string(res.error())));
        }

        output.flush();
        output.close();
    }

    // validate method matching Android implementation (exactly like Kotlin lines 240-247)
    bool ModelFileDownloader::Validate(const FileDownloadTask& fileDownloadTask, const fs::path& src)
    {
        LOG_DEBUG_TAG("validate: Checking file " + src.string(), "ModelFileDownloader");
        
        // Exactly matching Kotlin implementation:
        bool verify_result = true;
        
        // if (!fileDownloadTask.etag.isNullOrEmpty()) {
        if (!fileDownloadTask.etag.empty()) {
            LOG_DEBUG_TAG("validate: Verifying file hash with etag: " + fileDownloadTask.etag, "ModelFileDownloader");
            
            // verifyResult = HfShaVerifier.verify(fileDownloadTask.etag!!, src.toPath())
            verify_result = HfShaVerifier::verify(fileDownloadTask.etag, src);
            
            LOG_DEBUG_TAG("validate: verifyResult: " + std::string(verify_result ? "true" : "false"), "ModelFileDownloader");
        } else {
            LOG_DEBUG_TAG("validate: No etag, skipping verification", "ModelFileDownloader");
        }
        
        // return verifyResult
        return verify_result;
    }

    // Note: CreateSymlinkSafely removed - no longer needed with direct storage architecture

    // MoveWithPermissions method matching Android implementation
    void ModelFileDownloader::MoveWithPermissions(const fs::path& src, const fs::path& dest)
    {
        LOG_DEBUG("MoveWithPermissions " + src.string() + " to " + dest.string());
        
        // Ensure destination directory exists
        std::filesystem::create_directories(dest.parent_path());
        
        // Try rename first (fastest), fall back to copy+remove if it fails
        std::error_code ec;
        std::filesystem::rename(src, dest, ec);
        
        if (ec) {
            LOG_DEBUG("rename failed: " + ec.message() + ", trying copy+remove");
            // Fall back to copy + remove (for cross-filesystem moves)
            std::filesystem::copy_file(src, dest, std::filesystem::copy_options::overwrite_existing);
            std::filesystem::remove(src);
        }
        
        // Set permissions equivalent to Android setReadable(true, true), setWritable(true, true), setExecutable(false, false)
        std::filesystem::permissions(dest, 
            std::filesystem::perms::owner_read | std::filesystem::perms::owner_write |
            std::filesystem::perms::group_read | std::filesystem::perms::group_write |
            std::filesystem::perms::others_read | std::filesystem::perms::others_write);
    }

}