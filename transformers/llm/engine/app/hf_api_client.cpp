
//
// Created by ruoyi.sjd on 2024/12/24.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//
#include "hf_api_client.hpp"
#include "remote_model_downloader.hpp"

#include <httplib.h>
#include <thread>
#include <rapidjson/document.h>
#include <cstdlib>
#include "file_utils.hpp"
#include "functional"
#include "mls_config.hpp"

namespace mls {

std::tuple<std::string, std::string> HfApiClient::ParseUrl(const std::string &url) {
    auto https_pos = url.find("https://");
    std::string host;
    std::string path;
    if (https_pos != std::string::npos) {
        auto host_start = https_pos + 8;
        auto path_start = url.find('/', host_start);
        if (path_start != std::string::npos) {
            host = url.substr(host_start, path_start - host_start);
            path = url.substr(path_start);
            return {host, path};
        }
    }
    return {"", ""};
}

std::vector<RepoItem> HfApiClient::SearchRepos(const std::string& keyword) {
    std::string error_info;
    auto result =  SearchReposInner(keyword, error_info);
    if (!error_info.empty()) {
        printf("Failed to search repos: %s", error_info.c_str());
    }
    return result;
}

std::vector<RepoItem> HfApiClient::SearchReposInner(const std::string& keyword, std::string& error_info) {
    httplib::SSLClient cli(this->host_, 443);
    httplib::Headers headers;
    std::string path = "/api/models?search=" + keyword + "&author=taobao-mnn&limit=100";
    std::vector<RepoItem> repo_list;
    auto res = cli.Get(path, headers);
    if (!res) {
        error_info = "No response received from the server.";
        return {};
    }
    if (res->status != 200) {
        error_info = "Failed to fetch repository list. HTTP Status: " + std::to_string(res->status);
        return {};
    }
    // Parse the JSON response
    rapidjson::Document doc;
    if (doc.Parse(res->body.c_str()).HasParseError()) {
        error_info = "Failed to parse JSON response";
    }

    // Ensure the response is an array
    if (!doc.IsArray()) {
        error_info = "Unexpected JSON format: Expected an array of repositories.";
    }
    // Iterate through each repository object in the array
    auto repo_array = doc.GetArray();
    auto repo_size = repo_array.Size();
    for (int i = 0; i < repo_size; i++) {
        const auto& item = repo_array[i];
        if (!item.IsObject()) {
            continue; // Skip if not an object
        }

        mls::RepoItem repo_info;

        // Extract "modelId"
        if (item.HasMember("modelId") && item["modelId"].IsString()) {
            repo_info.model_id = item["modelId"].GetString();
        }

        if (item.HasMember("createdAt") && item["createdAt"].IsString()) {
            repo_info.created_at = item["createdAt"].GetString();
        }

        // Extract "downloads"
        if (item.HasMember("downloads") && item["downloads"].IsInt()) {
            repo_info.downloads = item["downloads"].GetInt();
        }

        // Extract "tags"
        if (item.HasMember("tags") && item["tags"].IsArray()) {
            const auto& tags = item["tags"];
            for (const auto& tag : tags.GetArray()) {
                if (tag.IsString()) {
                    repo_info.tags.emplace_back(tag.GetString());
                }
            }
        }
        // Add the populated RepoInfo to the list
        repo_list.emplace_back(std::move(repo_info));
    }
    return repo_list;
}

mls::RepoInfo HfApiClient::GetRepoInfo(
    const std::string& repo_name,
    const std::string& revision,
    std::string& error_info) {
    // Construct the API URL
    const std::string path =  "/api/models/" + repo_name + "/revision/" + revision;
    // Parsed repository info
    RepoInfo repo_info;

    // Perform the API request
    auto request_func = [&]() -> bool {
        // Parse host and path from the URL
        // Make the HTTPS request
        httplib::SSLClient cli(this->host_, 443);
        httplib::Headers headers;
        auto res = cli.Get(path, headers);
        if (!res || res->status != 200) {
            return false;
        }

        // Parse the JSON response
        rapidjson::Document doc;
        if (doc.Parse(res->body.c_str()).HasParseError()) {
            error_info = "Failed to parse JSON response";
            return {};
        }

        // Extract fields
        if (doc.HasMember("modelId") && doc["modelId"].IsString()) {
            repo_info.model_id = doc["modelId"].GetString();
        }
        if (doc.HasMember("sha") && doc["sha"].IsString()) {
            repo_info.sha = doc["sha"].GetString();
        }
        if (doc.HasMember("revision") && doc["revision"].IsString()) {
            repo_info.revision = doc["revision"].GetString();
        }
        if (doc.HasMember("siblings") && doc["siblings"].IsArray()) {
            const rapidjson::Value& siblings = doc["siblings"];
            for (rapidjson::Value::ConstValueIterator it = siblings.Begin(); it != siblings.End(); ++it) {
                if (it->IsObject() && it->HasMember("rfilename") && (*it)["rfilename"].IsString()) {
                    repo_info.siblings.emplace_back((*it)["rfilename"].GetString());
                }
            }
        }

        return true;
    };

    if (!HfApiClient::PerformRequestWithRetry(request_func, 3, 1)) {
        error_info = "Failed to fetch repository info after retries";
        return {};
    }

    return repo_info;
}

HfApiClient::HfApiClient() {
    cache_path_ = FileUtils::GetBaseCacheDir();
    if (const char* hf_endpoint  = std::getenv("HF_ENDPOINT")) {
        std::string path;
        std::tie(this->host_, path) = ParseUrl(std::string(hf_endpoint));
    }
}

bool HfApiClient::PerformRequestWithRetry(std::function<bool()> request_func, int max_attempts, int retry_delay_seconds) {
   int attempts_left = max_attempts;
   int attempt_count = 0;

   while (attempts_left > 0) {
       attempt_count++;
       if (request_func()) {
           return true;
       }
       attempts_left--;
       int backoff_ms = static_cast<int>(std::pow(retry_delay_seconds, attempt_count - 1) * 1000);
       std::this_thread::sleep_for(std::chrono::milliseconds(backoff_ms));
   }
   return false;
}

void HfApiClient::DownloadRepo(const RepoInfo& repo_info) {
    mls::RemoteModelDownloader model_downloader{this->host_, this->max_attempts_, this->retry_delay_seconds_};
    std::string error_info;
    bool has_error = false;
    auto repo_folder_name = FileUtils::RepoFolderName(repo_info.model_id, "model");
    fs::path storage_folder = fs::path(this->cache_path_) / repo_folder_name;
    const auto parent_pointer_path = FileUtils::GetPointerPathParent(storage_folder,  repo_info.sha);
    const auto folder_link_path = fs::path(this->cache_path_) / FileUtils::GetFileName(repo_info.model_id);
    std::error_code ec;
    bool downloaded = is_symlink(folder_link_path, ec);
    if (downloaded) {
        printf("already donwnloaded at %s\n", folder_link_path.string().c_str());
        return;
    }
    for (auto & sub_file :  repo_info.siblings) {
        model_downloader.DownloadWithRetries(storage_folder, repo_info.model_id, repo_info.sha, sub_file, error_info, 3);
        has_error = has_error || !error_info.empty();
        if (has_error) {
            fprintf(stderr, "DownloadFile error at file: %s error message: %s",sub_file.c_str(), error_info.c_str());
            break;
        }
    }
    if (!has_error) {
        std::error_code ec;
        FileUtils::CreateSymlink(parent_pointer_path, folder_link_path, ec);
        if (ec) {
            fprintf(stderr, "DownlodRepo CreateSymlink error: %s", ec.message().c_str());
        }
    }
}

}
