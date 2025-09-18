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
namespace mnncli {

// HfFileMetadata is already defined in hf_file_metadata.hpp

// FileDownloadTask structure matching Android FileDownloadTask.kt exactly
struct FileDownloadTask {
    std::string etag;
    std::string relativePath;
    HfFileMetadata fileMetadata;
    fs::path blobPath;
    fs::path blobPathIncomplete;
    fs::path pointerPath;
    int64_t downloadedSize = 0;
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
    void setVerbose(bool verbose) { verbose_ = verbose; }
    
    // Main download method matching Android ModelFileDownloader.downloadFile() exactly
    void downloadFile(FileDownloadTask& fileDownloadTask, FileDownloadListener& fileDownloadListener);

private:
    // Private methods matching Android implementation exactly
    void downloadToTmpAndMove(
        FileDownloadTask& fileDownloadTask,
        const fs::path& incompletePath,
        const fs::path& destinationPath,
        const std::string& urlToDownload,
        int64_t expectedSize,
        const std::string& fileName,
        FileDownloadListener& fileDownloadListener);

    void downloadChunk(
        FileDownloadTask& fileDownloadTask,
        const std::string& url,
        const fs::path& tempFile,
        int64_t expectedSize,
        const std::string& displayedFilename,
        FileDownloadListener* fileDownloadListener);

    bool validate(const FileDownloadTask& fileDownloadTask, const fs::path& src);
    void moveWithPermissions(const fs::path& src, const fs::path& dest);
    void createSymlinkSafely(const fs::path& target, const fs::path& link_path);
    
    // Verbose logging helper
    void logVerbose(const std::string& message) const;

private:
    // HTTP client for downloads
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    httplib::SSLClient client_;
#else
    httplib::Client client_;
#endif
    static constexpr int CONNECT_TIMEOUT_SECONDS = 30;
    static constexpr int MAX_RETRY = 10;
    static constexpr int BUFFER_SIZE = 8192;
    
    // Verbose logging flag
    bool verbose_ = false;
};

}