//
// Created by ruoyi.sjd on 2024/12/18.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#include "ml_api_client.hpp"
#include "file_utils.hpp"
#include "dl_config.hpp"
#include <httplib.h>
#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <iostream>
#include <algorithm>
#include <cmath>

namespace mnn::downloader {

MlApiClient::MlApiClient() : host_("modelers.cn"), max_attempts_(3), retry_delay_seconds_(2) {
    cache_path_ = FileUtils::ExpandTilde(mnn::downloader::kCachePath);
}

MlRepoInfo MlApiClient::GetRepoInfo(const std::string& repo_name, const std::string& revision, std::string& error_info) {
    // Parse modelGroup/modelPath from the repository name
    size_t slash_pos = repo_name.find('/');
    if (slash_pos == std::string::npos) {
        error_info = "Invalid repository format, expected: modelGroup/modelPath";
        return {};
    }
    
    std::string model_group = repo_name.substr(0, slash_pos);
    std::string model_path = repo_name.substr(slash_pos + 1);
    
    // Construct the Modelers API URL
    std::string path = "/api/v1/file/" + model_group + "/" + model_path;
    
    // Perform the API request
    auto request_func = [&]() -> bool {
        // Create HTTP client
        httplib::SSLClient cli(host_, 443);
        httplib::Headers headers;
        headers.emplace("User-Agent", "MNN-CLI/1.0");
        headers.emplace("Accept", "application/json");
        
        // Add query parameter for root path
        std::string full_path = path + "?path=/";
        
        auto res = cli.Get(full_path, headers);
        if (!res || res->status != 200) {
            if (res) {
                error_info = "Modelers API request failed with status " + std::to_string(res->status);
                if (res->status == 404) {
                    error_info += " - Repository not found. Check if the model exists on Modelers.";
                }
            } else {
                error_info = "Modelers API request failed - no response";
            }
            return false;
        }

        // Parse the JSON response
        rapidjson::Document doc;
        if (doc.Parse(res->body.c_str()).HasParseError()) {
            error_info = "Failed to parse Modelers API response";
            return false;
        }

        // Extract repository information
        if (!doc.HasMember("Data") || !doc["Data"].IsObject()) {
            error_info = "Invalid Modelers API response: missing 'Data' field";
            return false;
        }

        const rapidjson::Value& data = doc["Data"];
        
        // Extract basic info
        if (data.HasMember("ModelId") && data["ModelId"].IsString()) {
            repo_info_.model_id = data["ModelId"].GetString();
        }
        
        if (data.HasMember("Revision") && data["Revision"].IsString()) {
            repo_info_.revision = data["Revision"].GetString();
        }
        
        // Extract files information
        if (data.HasMember("Files") && data["Files"].IsArray()) {
            const rapidjson::Value& files = data["Files"];
            for (rapidjson::Value::ConstValueIterator it = files.Begin(); it != files.End(); ++it) {
                if (it->IsObject()) {
                    MlFileInfo file_info;
                    
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
                    
                    // Only add non-directory files
                    if (file_info.type != "tree") {
                        repo_info_.files.push_back(file_info);
                    }
                }
            }
        }

        return true;
    };

    if (!PerformRequestWithRetry(request_func, max_attempts_, retry_delay_seconds_)) {
        error_info = "Failed to fetch Modelers repository info after retries";
        return {};
    }

    return repo_info_;
}

bool MlApiClient::PerformRequestWithRetry(std::function<bool()> request_func, int max_attempts, int retry_delay_seconds) {
    int attempts_left = max_attempts;
    int attempt_count = 0;

    while (attempts_left > 0) {
        attempt_count++;
        if (request_func()) {
            return true;
        }
        attempts_left--;
        
        if (attempts_left > 0) {
            int backoff_ms = static_cast<int>(pow(retry_delay_seconds, attempt_count - 1) * 1000);
            std::this_thread::sleep_for(std::chrono::milliseconds(backoff_ms));
        }
    }
    return false;
}

void MlApiClient::DownloadRepo(const MlRepoInfo& repo_info) {
    // This would implement the actual download logic using the file information
    // from the repository info
    std::cout << "Downloading repository: " << repo_info.model_id << std::endl;
    std::cout << "Files to download: " << repo_info.files.size() << std::endl;
    
    // TODO: Implement actual file download logic
    // This would create download tasks for each file in repo_info.files
}

} // namespace mnn::downloader
