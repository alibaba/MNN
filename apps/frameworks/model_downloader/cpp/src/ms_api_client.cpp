//
// Created by ruoyi.sjd on 2024/12/18.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#include "ms_api_client.hpp"

#include "file_utils.hpp"
#include "dl_config.hpp"
#include <httplib.h>
#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>

namespace mnn::downloader {

MsApiClient::MsApiClient() : host_("modelscope.cn"), max_attempts_(3), retry_delay_seconds_(2) {
    cache_path_ = FileUtils::ExpandTilde(mnn::downloader::kCachePath);
}

MsRepoInfo MsApiClient::GetRepoInfo(const std::string& repo_name, const std::string& revision, std::string& error_info) {
    // Parse owner/repo from the repository name
    size_t slash_pos = repo_name.find('/');
    if (slash_pos == std::string::npos) {
        error_info = "Invalid repository format, expected: owner/repo";
        return {};
    }
    
    std::string owner = repo_name.substr(0, slash_pos);
    std::string repo = repo_name.substr(slash_pos + 1);
    
    // Construct the ModelScope API URL - use the correct endpoint from Android code
    std::string path = "/api/v1/models/" + owner + "/" + repo + "/repo/files?Recursive=1";
    
    // Perform the API request
    auto request_func = [&]() -> bool {
        // Create HTTP client
        httplib::SSLClient cli(host_, 443);
        httplib::Headers headers;
        headers.emplace("User-Agent", "MNN-CLI/1.0");
        headers.emplace("Accept", "application/json");
        
        auto res = cli.Get(path, headers);
        if (!res || res->status != 200) {
            if (res) {
                error_info = "ModelScope API request failed with status " + std::to_string(res->status);
                if (res->status == 404) {
                    error_info += " - Repository not found. Check if the model exists on ModelScope.";
                }
            } else {
                error_info = "ModelScope API request failed - no response";
            }
            return false;
        }

        // Parse the JSON response
        rapidjson::Document doc;
        if (doc.Parse(res->body.c_str()).HasParseError()) {
            error_info = "Failed to parse ModelScope API response: " + 
                        std::string(rapidjson::GetParseError_En(doc.GetParseError()));
            return false;
        }

        // Extract repository information
        if (!doc.HasMember("Data") || !doc["Data"].IsObject()) {
            error_info = "Invalid ModelScope API response: missing 'Data' field";
            return false;
        }

        const rapidjson::Value& data = doc["Data"];
        
        // Extract basic info - use the correct field names from the API response
        if (data.HasMember("Files") && data["Files"].IsArray()) {
            const rapidjson::Value& files = data["Files"];
            for (rapidjson::Value::ConstValueIterator it = files.Begin(); it != files.End(); ++it) {
                if (it->IsObject()) {
                    MsFileInfo file_info;
                    
                    if (it->HasMember("Path") && (*it)["Path"].IsString()) {
                        file_info.path = (*it)["Path"].GetString();
                    }
                    
                    if (it->HasMember("Type") && (*it)["Type"].IsString()) {
                        file_info.type = (*it)["Type"].GetString();
                    }
                    
                    if (it->HasMember("Size") && (*it)["Size"].IsInt64()) {
                        file_info.size = (*it)["Size"].GetInt64();
                    }
                    
                    if (it->HasMember("Sha256") && (*it)["Sha256"].IsString()) {
                        file_info.sha256 = (*it)["Sha256"].GetString();
                    }
                    
                    // Only add non-directory files (Type is "blob" for files, not "tree")
                    if (file_info.type == "blob") {
                        repo_info_.files.push_back(file_info);
                    }
                }
            }
        }
        
        // Set model_id and revision from the first file if available
        if (!repo_info_.files.empty()) {
            const auto& first_file = repo_info_.files[0];
            if (first_file.path.find('/') == std::string::npos) {
                // This is a root-level file, use the repo name as model_id
                repo_info_.model_id = owner + "/" + repo;
            }
            repo_info_.revision = "master"; // Default revision
        }

        return true;
    };

    if (!PerformRequestWithRetry(request_func, max_attempts_, retry_delay_seconds_)) {
        error_info = "Failed to fetch ModelScope repository info after retries";
        return {};
    }

    return repo_info_;
}

bool MsApiClient::PerformRequestWithRetry(std::function<bool()> request_func, int max_attempts, int retry_delay_seconds) {
    int attempts_left = max_attempts;
    int attempt_count = 0;

    while (attempts_left > 0) {
        attempt_count++;
        if (request_func()) {
            return true;
        }
        attempts_left--;
        
        if (attempts_left > 0) {
            int backoff_ms = static_cast<int>(std::pow(retry_delay_seconds, attempt_count - 1) * 1000);
            std::this_thread::sleep_for(std::chrono::milliseconds(backoff_ms));
        }
    }
    return false;
}

void MsApiClient::DownloadRepo(const MsRepoInfo& repo_info) {
    // This would implement the actual download logic using the file information
    // from the repository info
    std::cout << "Downloading repository: " << repo_info.model_id << std::endl;
    std::cout << "Files to download: " << repo_info.files.size() << std::endl;
    
    // TODO: Implement actual file download logic
    // This would create download tasks for each file in repo_info.files
}

} // namespace mnn::downloader
