//
// Created by ruoyi.sjd on 2024/12/18.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//
#pragma once
#include <string>
#include <functional>
#include "httplib.h"
#include <filesystem>

namespace fs = std::filesystem;
namespace mls {

struct Progress {
    size_t content_length = 0;
    size_t downloaded = 0;
    bool success = false;
    std::string error_message;
};

struct RepoInfo {
    std::string model_id;
    std::string revision;
    std::string sha;
    std::vector<std::string> siblings;
};

struct HfFileMetadata {
    std::string commit_hash;
    std::string location;
    std::string etag;
    size_t size;
};

class RemoteModelDownloader {
public:
    // 构造函数可选地接收重试次数、重试间隔等配置
    explicit RemoteModelDownloader(int max_attempts = 3, int retry_delay_seconds = 2);

    std::string DownloadFromHF(const std::string &repo,
                        const std::string &revision,
                        const std::string &file_on_repo,
                        const std::string &local_path = "",
                        const std::string &hf_token = "");


    RepoInfo getRepoInfo(const std::string &repo_name,
                         const std::string &revision,
                         const std::string &hf_token,
                         std::string& error_info);
private:
    void HttpGet(
        const std::string& url,
        const std::filesystem::path& temp_file,
        const std::unordered_map<std::string, std::string>& proxies,
        size_t resume_size,
        const httplib::Headers& headers,
        const std::optional<size_t>& expected_size,
        const std::string& displayed_filename,
        std::string& error_info);

    bool CheckDiskSpace(size_t required_size, const std::filesystem::path& path);
    void MoveWithPermissions(const std::filesystem::path& src, const std::filesystem::path& dest);

    HfFileMetadata GetFileMetadata(const std::string& url, std::string& error_info);

    std::string DownloadFile(const std::string &url,
                      const std::string &local_path,
                      const std::string &hf_token,
                      const std::string &repo,
                      const std::string &relative_path);


    bool ShouldDownload(const std::string &local_path,
                        const std::string &url,
                        const std::string &new_etag,
                        const std::string &new_last_modified);

    bool EnsureDirectoryExists(const std::string &path);

    // Ensure the directory for a given path exists


    // 解析或写入 metadata（存储在 local_path + ".json"）
    void LoadMetadata(const std::string &metadata_path,
                      std::string &old_etag,
                      std::string &old_last_modified,
                      std::string &old_url);

    void WriteMetadata(const std::string &metadata_path,
                       const std::string &url,
                       const std::string &etag,
                       const std::string &last_modified);

    bool PerformRequestWithRetry(std::function<bool()> request_func);

    void DownloadToTmpAndMove(
        const fs::path& incomplete_path,
        const fs::path& destination_path,
        const std::string& url_to_download,
        httplib::Headers& headers,
        size_t expected_size,
        const std::string& file_name,
        bool force_download
    );

private:
    int max_attempts_;
    int retry_delay_seconds_;
    std::string cache_path_;
    std::string host_{"huggingface.co"};
};
}