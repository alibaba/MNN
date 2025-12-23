//
// Created by ruoyi.sjd on 2024/12/24.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//
#pragma once

#include <functional>
#include <string>
#include <vector>

namespace mnn::downloader {

struct RepoInfo {
    std::string model_id;
    std::string revision;
    std::string sha;
    std::vector<std::string> siblings;
};

struct RepoItem {
    std::string model_id;
    std::string created_at;
    int downloads;
    std::vector<std::string> tags;
};

//a benchmark referenced llama.cpp
class HfApiClient {
public:
    std::vector<RepoItem> SearchRepos(const std::string& keyword);

    HfApiClient();

    static std::tuple<std::string, std::string> ParseUrl(const std::string &url);

    static bool PerformRequestWithRetry(std::function<bool()> request_func, int max_attempts, int retry_delay_seconds);

    RepoInfo GetRepoInfo(const std::string &repo_name,
                     const std::string &revision,
                     std::string& error_info);

    void DownloadRepo(const RepoInfo& repo_info);
    
    // Get the host
    std::string GetHost() const;

private:
    std::vector<RepoItem> SearchReposInner(const std::string& keyword, std::string& error_info);

    int max_attempts_{3};
    int retry_delay_seconds_{1};
    std::string host_{"huggingface.co"};
    std::string cache_path_{};

};

}//mnncli