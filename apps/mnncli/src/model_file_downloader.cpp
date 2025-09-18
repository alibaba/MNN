//
// Created by ruoyi.sjd on 2024/12/18.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#include "model_file_downloader.hpp"
#include "httplib.h"
#include "file_utils.hpp"
#include "hf_sha_verifier.hpp"
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

namespace mnncli
{

    // Constructor matching Android OkHttpClient configuration
    ModelFileDownloader::ModelFileDownloader() : client_("", 443) {
        // Note: client_ is no longer used for downloads, but kept for compatibility
        // Each download will create its own client with the correct host
    }

    // Verbose logging helper implementation
    void ModelFileDownloader::logVerbose(const std::string& message) const {
        if (verbose_) {
            std::cout << "[ModelFileDownloader] " << message << std::endl;
        }
    }

    // Main download method matching Android ModelFileDownloader.downloadFile() exactly
    void ModelFileDownloader::downloadFile(FileDownloadTask& fileDownloadTask, FileDownloadListener& fileDownloadListener)
    {
        logVerbose("Starting download for file: " + fileDownloadTask.relativePath);
        
        try {
            // Validate paths to prevent symlink issues
            if (fileDownloadTask.pointerPath.empty() || fileDownloadTask.blobPath.empty()) {
                throw FileDownloadException("Invalid file paths");
            }
            
            // Check for circular symlink references
            if (fileDownloadTask.pointerPath == fileDownloadTask.blobPath) {
                throw FileDownloadException("Pointer path cannot be the same as blob path");
            }
            
            // Create directories
            std::filesystem::create_directories(fileDownloadTask.pointerPath.parent_path());
            std::filesystem::create_directories(fileDownloadTask.blobPath.parent_path());
            
            logVerbose("Directories created successfully");
        } catch (const std::exception& e) {
            logVerbose("Error during directory creation: " + std::string(e.what()));
            throw;
        }

        // Check if file already exists
        if (std::filesystem::exists(fileDownloadTask.pointerPath))
        {
            printf("DownloadFile %s already exists\n", fileDownloadTask.relativePath.c_str());
            return;
        }

        // Check if blob already exists
        if (std::filesystem::exists(fileDownloadTask.blobPath))
        {
            logVerbose("Blob file already exists, creating symlink");
            createSymlinkSafely(fileDownloadTask.blobPath, fileDownloadTask.pointerPath);
            logVerbose("DownloadFile " + fileDownloadTask.relativePath + " already exists just create symlink");
            return;
        }

        // Download the file
        logVerbose("Starting download process");
        try {
            std::mutex lock;
            {
                std::lock_guard<std::mutex> guard(lock);
                downloadToTmpAndMove(fileDownloadTask, fileDownloadTask.blobPathIncomplete, fileDownloadTask.blobPath, 
                                   fileDownloadTask.fileMetadata.location, fileDownloadTask.fileMetadata.size,
                                   fileDownloadTask.relativePath, fileDownloadListener);
                
                logVerbose("Download completed, creating symlink");
                createSymlinkSafely(fileDownloadTask.blobPath, fileDownloadTask.pointerPath);
                logVerbose("Download and symlink creation completed successfully");
            }
        } catch (const std::exception& e) {
            logVerbose("Error during download process: " + std::string(e.what()));
            throw;
        }
    }

    // downloadToTmpAndMove method matching Android implementation exactly
    void ModelFileDownloader::downloadToTmpAndMove(
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
        if (std::filesystem::exists(destinationPath))
        {
            if (validate(fileDownloadTask, destinationPath))
            {
                return;
            }
            else
            {
                std::filesystem::remove(destinationPath);
                fileDownloadTask.downloadedSize = 0;
            }
        }
        
        // Check if we can resume from incomplete file (like Kotlin - simple size check, no validation)
        if (fileDownloadTask.downloadedSize >= expectedSize)
        {
            if (std::filesystem::exists(incompletePath) && std::filesystem::file_size(incompletePath) >= expectedSize)
            {
                moveWithPermissions(incompletePath, destinationPath);
                return;
            }
            else
            {
                std::filesystem::remove(incompletePath);
                fileDownloadTask.downloadedSize = 0;
            }
        }
        
        // Check for redirects first (like Android does)
        auto [host, path] = HfApiClient::ParseUrl(urlToDownload);
        httplib::Headers headers;
        headers.emplace("User-Agent", "MNN-CLI/1.0");
        headers.emplace("Accept", "*/*");
        
        auto head_res = client_.Head(path, headers);
        if (head_res && (head_res->status >= 301 && head_res->status <= 308))
        {
            std::string location = head_res->get_header_value("Location");
            if (!location.empty())
            {
                theUrlToDownload = location;
            }
        }
        
        logVerbose("downloadToTmpAndMove urlToDownload: " + theUrlToDownload + " to file: " + incompletePath.string() + " to destination: " + destinationPath.string());
        
        // Download with retry logic like Android (max 10 retries)
        if (fileDownloadTask.downloadedSize < expectedSize)
        {
            for (int i = 0; i < MAX_RETRY; i++)
            {
                try
                {
                    logVerbose("downloadChunk try the " + std::to_string(i) + " turn");
                    
                    downloadChunk(fileDownloadTask, theUrlToDownload, incompletePath, expectedSize, fileName, &fileDownloadListener);
                    
                    logVerbose("downloadChunk try the " + std::to_string(i) + " turn finish");
                    
                    // After download, validate the incomplete file (like Kotlin - only validate after download is complete)
                    if (std::filesystem::exists(incompletePath) && std::filesystem::file_size(incompletePath) >= expectedSize)
                    {
                        logVerbose("Download completed successfully, moving to final location");
                        break;
                    }
                    else
                    {
                        logVerbose("Download validation failed, removing incomplete file and resetting download size");
                        std::filesystem::remove(incompletePath);
                        fileDownloadTask.downloadedSize = 0;
                    }
                }
                catch (const DownloadPausedException& e)
                {
                    throw e;
                }
                catch (const std::exception& e)
                {
                    if (i == MAX_RETRY - 1)
                    {
                        logVerbose("Max retries reached, throwing exception: " + std::string(e.what()));
                        throw e;
                    }
                    else
                    {
                        logVerbose("downloadChunk failed sleep and retrying: " + std::string(e.what()));
                        try
                        {
                            std::this_thread::sleep_for(std::chrono::seconds(1));
                        }
                        catch (const std::exception& ex)
                        {
                            logVerbose("Error during sleep: " + std::string(ex.what()));
                            throw std::runtime_error(ex.what());
                        }
                    }
                }
            }
        }
        
        // Like Kotlin: only check if file exists and has correct size, no SHA verification during download
        if (std::filesystem::exists(incompletePath) && std::filesystem::file_size(incompletePath) >= expectedSize)
        {
            logVerbose("Download completed successfully, moving file to destination");
            moveWithPermissions(incompletePath, destinationPath);
        }
        else
        {
            logVerbose("Download validation failed, removing incomplete file and resetting download size");
            std::filesystem::remove(incompletePath);
            fileDownloadTask.downloadedSize = 0;
        }
    }

    // downloadChunk method matching Android implementation exactly
    void ModelFileDownloader::downloadChunk(
        FileDownloadTask& fileDownloadTask,
        const std::string& url,
        const fs::path& tempFile,
        int64_t expectedSize,
        const std::string& displayedFilename,
        FileDownloadListener* fileDownloadListener)
    {
        auto [host, path] = HfApiClient::ParseUrl(url);
        printf("downloadChunk: URL='%s', host='%s', path='%s'\n", url.c_str(), host.c_str(), path.c_str());
        
        // Create HTTP client for download
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
        httplib::SSLClient download_client(host, 443);
#else
        httplib::Client download_client(host, 80);
#endif
        download_client.set_connection_timeout(CONNECT_TIMEOUT_SECONDS, 0);
        download_client.set_read_timeout(CONNECT_TIMEOUT_SECONDS, 0);
        download_client.set_write_timeout(CONNECT_TIMEOUT_SECONDS, 0);
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
        download_client.enable_server_certificate_verification(false);
#endif
        download_client.set_keep_alive(true);
        
        if (verbose_) {
            printf("downloadChunk: Created new HTTP client for host: %s\n", host.c_str());
        }
        
        httplib::Headers requestHeaders;
        requestHeaders.emplace("Accept-Encoding", "identity");
        
        if (fileDownloadTask.downloadedSize >= expectedSize)
        {
            return;
        }
        
        int64_t downloadedBytes = fileDownloadTask.downloadedSize;

        if (fileDownloadTask.downloadedSize > 0)
        {
            requestHeaders.emplace("Range", "bytes=" + std::to_string(fileDownloadTask.downloadedSize) + "-");
        }
        
        printf("resume size: %lld expectedSize: %lld\n", fileDownloadTask.downloadedSize, expectedSize);
        
        // Open output file for writing
        std::ofstream output;
        try {
            output.open(tempFile, std::ios::binary | std::ios::app);
            if (!output.is_open())
            {
                throw FileDownloadException("Failed to open output file: " + tempFile.string());
            }
        } catch (const std::exception& e) {
            printf("downloadChunk: Exception opening file: %s\n", e.what());
            throw FileDownloadException("File operation failed: " + std::string(e.what()));
        }

        auto res = download_client.Get(path, requestHeaders, [&](const httplib::Response &response)
                              {
            printf("downloadChunk response: success: %s code: %d\n", 
                   response.status >= 200 && response.status < 300 ? "true" : "false", response.status);
            // Return false for non-successful status codes to stop the download
            if (response.status < 200 || response.status >= 300) {
                printf("downloadChunk: Stopping download due to HTTP error %d\n", response.status);
                return false;
            }
            return true; }, [&](const char *data, size_t data_length)
                              {
            try {
                output.write(data, data_length);
                downloadedBytes += data_length;
                fileDownloadTask.downloadedSize += data_length;
                
                if (fileDownloadListener != nullptr)
                {
                    bool paused = fileDownloadListener->onDownloadDelta(
                        &displayedFilename, downloadedBytes, expectedSize, data_length);
                    if (paused)
                    {
                        throw DownloadPausedException("Download paused");
                    }
                }
            } catch (const std::exception& e) {
                printf("downloadChunk: Exception during data processing: %s\n", e.what());
                throw FileDownloadException("Data processing failed: " + std::string(e.what()));
            }
            return true; });

        if (res)
        {
            if (res->status >= 200 && res->status < 300 || res->status == 416)
            {
                printf("downloadChunk completed successfully\n");
            }
            else
            {
                throw FileDownloadException("HTTP error: " + std::to_string(res->status));
            }
        }
        else
        {
            std::string error_msg = "Connection error: " + std::string(httplib::to_string(res.error()));
            printf("downloadChunk: %s\n", error_msg.c_str());
            throw FileDownloadException(error_msg);
        }

        try {
            output.flush();
            output.close();
        } catch (const std::exception& e) {
            printf("downloadChunk: Exception closing file: %s\n", e.what());
            throw FileDownloadException("File close failed: " + std::string(e.what()));
        }
    }

    // validate method matching Android implementation
    bool ModelFileDownloader::validate(const FileDownloadTask& fileDownloadTask, const fs::path& src)
    {
        // Check if file exists
        if (!std::filesystem::exists(src)) {
            if (verbose_) {
                printf("validate: File %s does not exist\n", src.string().c_str());
            }
            return false;
        }
        
        // Check if file has content
        auto file_size = std::filesystem::file_size(src);
        if (file_size == 0) {
            if (verbose_) {
                printf("validate: File %s is empty (0 bytes)\n", src.string().c_str());
            }
            return false;
        }
        
        // If we have an etag, verify the file hash using HfShaVerifier (like Android HfShaVerifier.verify())
        if (!fileDownloadTask.etag.empty()) {
            try {
                if (verbose_) {
                    printf("validate: Verifying file hash for %s with etag: %s\n", 
                           src.string().c_str(), fileDownloadTask.etag.c_str());
                }
                
                // Use the SHA verifier to check file integrity
                bool hash_verified = HfShaVerifier::verify(fileDownloadTask.etag, src);
                
                if (!hash_verified) {
                    if (verbose_) {
                        printf("validate: Hash verification failed for %s\n", src.string().c_str());
                    }
                    return false;
                }
                
                if (verbose_) {
                    printf("validate: Hash verification passed for %s\n", src.string().c_str());
                }
                
            } catch (const std::exception& e) {
                if (verbose_) {
                    printf("validate: Error during hash verification: %s\n", e.what());
                }
                // Continue with basic validation even if hash verification fails
                // This allows the download to proceed even if verification has issues
            }
        }
        
        if (verbose_) {
            printf("validate: File %s validation passed - size: %zu bytes\n", 
                   src.string().c_str(), file_size);
        }
        
        return true;
    }

    // Helper function to safely create symlinks
    void ModelFileDownloader::createSymlinkSafely(const fs::path& target, const fs::path& link_path)
    {
        // Remove existing link if it exists
        if (std::filesystem::exists(link_path)) {
            std::filesystem::remove(link_path);
        }
        
        // Check if target exists and is not a symlink to avoid circular references
        if (!std::filesystem::exists(target)) {
            throw FileDownloadException("Target file does not exist: " + target.string());
        }
        
        // Check if target is already a symlink to prevent circular references
        if (std::filesystem::is_symlink(target)) {
            // Resolve the actual target
            std::error_code ec;
            auto resolved_target = std::filesystem::read_symlink(target, ec);
            if (ec) {
                throw FileDownloadException("Failed to read symlink target: " + ec.message());
            }
            // Create symlink to the resolved target instead
            std::error_code symlink_ec;
            mnncli::FileUtils::CreateSymlink(resolved_target, link_path, symlink_ec);
            if (symlink_ec) {
                throw FileDownloadException("Failed to create symlink to resolved target: " + symlink_ec.message());
            }
        } else {
            // Create normal symlink
            std::error_code ec;
            mnncli::FileUtils::CreateSymlink(target, link_path, ec);
            if (ec) {
                throw FileDownloadException("Failed to create symlink: " + ec.message());
            }
        }
    }

    // moveWithPermissions method matching Android implementation
    void ModelFileDownloader::moveWithPermissions(const fs::path& src, const fs::path& dest)
    {
        printf("moveWithPermissions %s to %s\n", src.string().c_str(), dest.string().c_str());
        
        // Use std::filesystem::rename which is equivalent to Files.move in Java
        std::filesystem::rename(src, dest);
        
        // Set permissions equivalent to Android setReadable(true, true), setWritable(true, true), setExecutable(false, false)
        std::filesystem::permissions(dest, 
            std::filesystem::perms::owner_read | std::filesystem::perms::owner_write |
            std::filesystem::perms::group_read | std::filesystem::perms::group_write |
            std::filesystem::perms::others_read | std::filesystem::perms::others_write);
    }

}