//
// Created by ruoyi.sjd on 2024/12/18.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//
#pragma once
#include <string>
#include <filesystem>
#include <stdexcept>
#include "httplib.h"
#include "hf_api_client.hpp"
#include "hf_file_metadata.hpp"

namespace fs = std::filesystem;
namespace mnn::downloader {

// HfFileMetadata is already defined in hf_file_metadata.hpp

// FileDownloadTask structure - simplified for direct storage (ModelScope style)
// Removed blob/pointer paths, using direct storage path only
struct FileDownloadTask {
    std::string etag;              // For SHA validation
    std::string relativePath;      // Relative path in model repo
    HfFileMetadata fileMetadata;   // File metadata (location, size, etc.)
    
    // Simplified: single download path (cache_root/owner/model/file)
    fs::path downloadPath;         // Final destination path
    
    int64_t downloadedSize = 0;    // For resume support
    
    // Helper to get incomplete path (.incomplete suffix)
    fs::path GetIncompletePath() const {
        return fs::path(downloadPath.string() + ".incomplete");
    }
};

// FileDownloadException matching Android FileDownloadException.kt exactly
class FileDownloadException : public std::runtime_error {
public:
    explicit FileDownloadException(const std::string& message) : std::runtime_error(message) {}
};

// DownloadPausedException matching Android implementation
class DownloadPausedException : public std::runtime_error {
public:
    explicit DownloadPausedException(const std::string& message) : std::runtime_error(message) {}
};

// FileDownloadListener interface matching Android ModelFileDownloader.FileDownloadListener exactly
class FileDownloadListener {
public:
    virtual ~FileDownloadListener() = default;
    virtual bool onDownloadDelta(
        const std::string* fileName,
        int64_t downloadedBytes,
        int64_t totalBytes,
        int64_t delta) = 0;
};

// ModelFileDownloader class matching Android ModelFileDownloader.kt exactly
class ModelFileDownloader {
public:
    ModelFileDownloader();
    
    // Set verbose mode for logging
    void SetVerbose(bool verbose) { verbose_ = verbose; }
    
    // Main download method matching Android ModelFileDownloader.downloadFile() exactly
    void DownloadFile(FileDownloadTask& fileDownloadTask, FileDownloadListener& fileDownloadListener);

private:
    // Private methods matching Android implementation exactly
    void DownloadToTmpAndMove(
        FileDownloadTask& fileDownloadTask,
        const fs::path& incompletePath,
        const fs::path& destinationPath,
        const std::string& urlToDownload,
        int64_t expectedSize,
        const std::string& fileName,
        FileDownloadListener& fileDownloadListener);

    void DownloadChunk(
        FileDownloadTask& fileDownloadTask,
        const std::string& url,
        const fs::path& tempFile,
        int64_t expectedSize,
        const std::string& displayedFilename,
        FileDownloadListener* fileDownloadListener);


    bool Validate(const FileDownloadTask& fileDownloadTask, const fs::path& src);
    void MoveWithPermissions(const fs::path& src, const fs::path& dest);
    // Note: CreateSymlinkSafely removed - no longer needed with direct storage

private:
    // HTTP client for downloads
    httplib::SSLClient client_;
    static constexpr int kConnectTimeoutSeconds = 30;
    static constexpr int kMaxRetry = 10;
    static constexpr int kBufferSize = 8192;
    
    // Verbose logging flag
    bool verbose_ = false;
};

}