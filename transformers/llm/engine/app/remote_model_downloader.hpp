//
// Created by ruoyi.sjd on 2024/12/18.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//
#pragma once
#include <string>
#include <functional>
#include "httplib.h"
#include <filesystem>
#include "hf_api_client.hpp"

namespace fs = std::filesystem;
namespace mls {

struct DownloadProgress {
    size_t content_length = 0;
    size_t downloaded = 0;
    bool success = false;
    std::string error_message;
};

class RemoteModelDownloader {
public:
    explicit RemoteModelDownloader(std::string host, int max_attempts = 3, int retry_delay_seconds = 2);

    std::string DownloadFile(
                      const std::filesystem::path& storage_folder,
                      const std::string& repo,
                      const std::string& revision,
                      const std::string& relative_path,
                      std::string& error_info);
    std::string DownloadWithRetries(
                      const fs::path& storage_folder,
                      const std::string& repo,
                      const std::string& revision,
                      const std::string& relative_path,
                      std::string& error_info,
                      int max_retries);
private:
    void DownloadToTmpAndMove(
        const fs::path& incomplete_path,
        const fs::path& destination_path,
        const std::string& url_to_download,
        httplib::Headers& headers,
        size_t expected_size,
        const std::string& file_name,
        bool force_download,
        std::string& error_info);

    void DownloadFileInner(
        const std::string& url,
        const std::filesystem::path& temp_file,
        const std::unordered_map<std::string, std::string>& proxies,
        size_t resume_size,
        const httplib::Headers& headers,
        const size_t expected_size,
        const std::string& displayed_filename,
        std::string& error_info);

    bool CheckDiskSpace(size_t required_size, const std::filesystem::path& path);

    void MoveWithPermissions(const std::filesystem::path& src, const std::filesystem::path& dest, std::string& error_info);

    HfFileMetadata GetFileMetadata(const std::string& url, std::string& error_info);

private:
    int max_attempts_;
    int retry_delay_seconds_;
    std::string host_;
};
}