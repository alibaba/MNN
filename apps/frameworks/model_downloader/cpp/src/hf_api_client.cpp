
//
// Created by ruoyi.sjd on 2024/12/24.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//
#include "hf_api_client.hpp"
#include "model_file_downloader.hpp"

#include <httplib.h>
#include <thread>
#include <rapidjson/document.h>
#include <cstdlib>
#include "file_utils.hpp"
#include <functional>
#include <cmath>

#include "log_utils.hpp"
#include "dl_config.hpp"

namespace mnn::downloader {

static inline std::string Trim(const std::string& input) {
    const char* whitespace = " \t\n\r";
    const auto start = input.find_first_not_of(whitespace);
    if (start == std::string::npos) {
        return "";
    }
    const auto end = input.find_last_not_of(whitespace);
    return input.substr(start, end - start + 1);
}

std::tuple<std::string, std::string> HfApiClient::ParseUrl(const std::string &url) {
    const std::string cleaned = Trim(url);
    if (cleaned.empty()) {
        return {"", ""};
    }

    size_t host_start = 0;
    if (cleaned.rfind("https://", 0) == 0) {
        host_start = 8; // length of "https://"
    } else if (cleaned.rfind("http://", 0) == 0) {
        host_start = 7; // length of "http://"
    } else {
        host_start = 0; // no scheme; treat as host[/path]
    }

    const size_t path_start = cleaned.find('/', host_start);
    if (path_start != std::string::npos) {
        const std::string host = cleaned.substr(host_start, path_start - host_start);
        const std::string path = cleaned.substr(path_start);
        return {host, path};
    }
    // no explicit path; treat remainder as host and default path "/"
    const std::string host = cleaned.substr(host_start);
    if (host.empty()) {
        return {"", ""};
    }
    return {host, "/"};
}

std::vector<RepoItem> HfApiClient::SearchRepos(const std::string& keyword) {
    std::string error_info;
    auto result =  SearchReposInner(keyword, error_info);
    if (!error_info.empty()) {
        LOG_ERROR("Failed to search repos: " + error_info);
    }
    return result;
}

std::vector<RepoItem> HfApiClient::SearchReposInner(const std::string& keyword, std::string& error_info) {
    // Create HTTP client
    httplib::SSLClient cli(GetHost(), 443);
    
    // Configure SSL client with proper timeouts and settings
    cli.set_connection_timeout(30, 0);
    cli.set_read_timeout(30, 0);
    cli.set_write_timeout(30, 0);
    cli.enable_server_certificate_verification(false);
    cli.set_keep_alive(true);
    
    httplib::Headers headers;
    headers.emplace("User-Agent", "MNN-CLI/1.0");
    if (const char* hf_token = std::getenv("HF_TOKEN")) {
        std::string auth_header = "Bearer " + std::string(hf_token);
        headers.emplace("Authorization", auth_header);
        LOG_DEBUG_TAG("🔑 Using HF_TOKEN for authentication", "AUTH");
    } else {
        LOG_DEBUG_TAG("⚠️  No HF_TOKEN found. Some models may require authentication.", "AUTH");
        LOG_DEBUG_TAG("   To authenticate, export HF_TOKEN=your_token_here", "AUTH");
    }
    
    std::string path = "/api/models?search=" + keyword + "&author=taobao-mnn&limit=100";
    std::vector<RepoItem> repo_list;
    auto res = cli.Get(path, headers);
    if (!res) {
        error_info = "No response received from the server.";
        return {};
    }
    if (res->status != 200) {
        if (res->status == 401) {
            error_info = "Authentication required. Please export HF_TOKEN=your_token_here";
        } else if (res->status == 403) {
            error_info = "Access forbidden. The repository might be private.";
        } else if (res->status >= 500) {
            error_info = "Server error. Please try again later.";
        } else {
            error_info = "Failed to fetch repository list. HTTP Status: " + std::to_string(res->status);
        }
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

        mnn::downloader::RepoItem repo_info;

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

mnn::downloader::RepoInfo HfApiClient::GetRepoInfo(
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
        // Create HTTP client
        httplib::SSLClient cli(GetHost(), 443);
        
        // Configure SSL client with proper timeouts and settings
        cli.set_connection_timeout(30, 0);  // 30 seconds connection timeout
        cli.set_read_timeout(30, 0);        // 30 seconds read timeout
        cli.set_write_timeout(30, 0);       // 30 seconds write timeout
        
        // Enable server certificate verification (but allow self-signed for testing)
        cli.enable_server_certificate_verification(false);
        
        // Set SSL context options
        cli.set_keep_alive(true);
        
        httplib::Headers headers;
        headers.emplace("User-Agent", "MNN-CLI/1.0");
        headers.emplace("Accept", "application/json");
        headers.emplace("Connection", "keep-alive");
        
        // Add Hugging Face authentication token if available
        if (const char* hf_token = std::getenv("HF_TOKEN")) {
            std::string auth_header = "Bearer " + std::string(hf_token);
            headers.emplace("Authorization", auth_header);
            LOG_DEBUG_TAG("🔑 Using HF_TOKEN for authentication", "AUTH");
        } else {
            LOG_DEBUG_TAG("⚠️  No HF_TOKEN found. Some models may require authentication.", "AUTH");
            LOG_DEBUG_TAG("   To authenticate, export HF_TOKEN=your_token_here", "AUTH");
        }
        
        LOG_DEBUG_TAG("🔍 Making request to: https://" + GetHost() + path, "API_REQUEST");
        
        auto res = cli.Get(path, headers);
        if (!res || res->status != 200) {
            std::string error_msg = "API request failed";
            if (res) {
                error_msg += " with status " + std::to_string(res->status);
                if (res->status == 401) {
                    error_msg += " - Authentication required. Please export HF_TOKEN=your_token_here";
                } else if (res->status == 404) {
                    error_msg += " - Repository not found. Check if the model exists.";
                } else if (res->status == 403) {
                    error_msg += " - Access forbidden. The repository might be private.";
                } else if (res->status >= 500) {
                    error_msg += " - Server error. Please try again later.";
                }
                LOG_ERROR(error_msg);
                LOG_DEBUG_TAG("   Response headers:", "API_RESPONSE");
                for (const auto& header : res->headers) {
                    LOG_DEBUG_TAG("     " + header.first + ": " + header.second, "API_RESPONSE");
                }
                if (!res->body.empty()) {
                    LOG_DEBUG_TAG("   Response body preview: " + res->body.substr(0, 200) + "...", "API_RESPONSE");
                }
            } else {
                error_msg += " - No response received";
                LOG_ERROR(error_msg);
                
                // Check for SSL errors
                auto ssl_error = cli.get_openssl_verify_result();
                if (ssl_error != 0) {
                    LOG_ERROR("SSL verification failed: " + std::to_string(ssl_error));
                    error_info = "SSL verification failed";
                    return {};
                }
                
                // Check if it's a connection timeout or other network issue
                LOG_DEBUG_TAG("   Possible causes:", "CONNECTIVITY");
                LOG_DEBUG_TAG("   - Network connectivity issues", "CONNECTIVITY");
                LOG_DEBUG_TAG("   - SSL/TLS certificate problems", "CONNECTIVITY");
                LOG_DEBUG_TAG("   - Firewall blocking HTTPS traffic", "CONNECTIVITY");
                LOG_DEBUG_TAG("   - DNS resolution failure for " + GetHost(), "CONNECTIVITY");
                LOG_DEBUG_TAG("   - Server is down or unreachable", "CONNECTIVITY");
                
                // Try a simple connection test
                LOG_DEBUG_TAG("   Testing basic connectivity...", "CONNECTIVITY");
                // Create HTTP client
                httplib::SSLClient test_cli(GetHost(), 443);
                test_cli.set_connection_timeout(10, 0);
                test_cli.enable_server_certificate_verification(false);
                auto test_res = test_cli.Get("/");
                if (test_res) {
                    LOG_DEBUG_TAG("   Basic connectivity: OK (got response)", "CONNECTIVITY");
                } else {
                    LOG_DEBUG_TAG("   Basic connectivity: FAILED (no response)", "CONNECTIVITY");
                }
            }
            return false;
        }

        LOG_DEBUG_TAG("✅ API response received successfully", "API_RESPONSE");
        LOG_DEBUG_TAG("   Status: " + std::to_string(res->status), "API_RESPONSE");
        LOG_DEBUG_TAG("   Content-Length: " + res->get_header_value("Content-Length"), "API_RESPONSE");
        LOG_DEBUG_TAG("   Content-Type: " + res->get_header_value("Content-Type"), "API_RESPONSE");

        // Parse the JSON response
        rapidjson::Document doc;
        if (doc.Parse(res->body.c_str()).HasParseError()) {
            error_info = "Failed to parse JSON response";
            LOG_ERROR(error_info);
            LOG_DEBUG_TAG("   Parse error at position: " + std::to_string(doc.GetErrorOffset()), "JSON_PARSE");
            LOG_DEBUG_TAG("   Response preview: " + res->body.substr(0, 200) + "...", "JSON_PARSE");
            return false;
        }

        LOG_DEBUG_TAG("✅ JSON parsed successfully", "JSON_PARSE");

        // Extract fields
        if (doc.HasMember("modelId") && doc["modelId"].IsString()) {
            repo_info.model_id = doc["modelId"].GetString();
            LOG_DEBUG_TAG("   Model ID: " + repo_info.model_id, "JSON_PARSE");
        } else {
            LOG_DEBUG_TAG("⚠️  Missing or invalid modelId field", "JSON_PARSE");
        }
        
        if (doc.HasMember("sha") && doc["sha"].IsString()) {
            repo_info.sha = doc["sha"].GetString();
            LOG_DEBUG_TAG("   SHA: " + repo_info.sha, "JSON_PARSE");
        } else {
            LOG_DEBUG_TAG("⚠️  Missing or invalid sha field", "JSON_PARSE");
        }
        
        if (doc.HasMember("revision") && doc["revision"].IsString()) {
            repo_info.revision = doc["revision"].GetString();
            LOG_DEBUG_TAG("   Revision: " + repo_info.revision, "JSON_PARSE");
        } else {
            LOG_DEBUG_TAG("⚠️  Missing or invalid revision field", "JSON_PARSE");
        }
        
        if (doc.HasMember("siblings") && doc["siblings"].IsArray()) {
            const rapidjson::Value& siblings = doc["siblings"];
            LOG_DEBUG("   Siblings array size: " + std::to_string(siblings.Size()));
            
            for (rapidjson::Value::ConstValueIterator it = siblings.Begin(); it != siblings.End(); ++it) {
                if (it->IsObject() && it->HasMember("rfilename") && (*it)["rfilename"].IsString()) {
                    repo_info.siblings.emplace_back((*it)["rfilename"].GetString());
                }
            }
            LOG_DEBUG("   Valid siblings found: " + std::to_string(repo_info.siblings.size()));
        } else {
            LOG_DEBUG("⚠️  Missing or invalid siblings field");
        }

        return true;
    };

    if (!HfApiClient::PerformRequestWithRetry(request_func, 3, 1)) {
        error_info = "Failed to fetch repository info after retries";
        LOG_ERROR(error_info);
        return {};
    }

    LOG_DEBUG_TAG("✅ Repository info retrieved successfully", "API_REQUEST");
    return repo_info;
}

HfApiClient::HfApiClient() {
    cache_path_ = FileUtils::ExpandTilde(mnn::downloader::kCachePath);
    // 默认使用 huggingface.co，只有当设置了 HF_ENDPOINT 环境变量时才使用自定义端点
    if (const char* hf_endpoint  = std::getenv("HF_ENDPOINT")) {
        std::string path;
        std::tie(this->host_, path) = ParseUrl(std::string(hf_endpoint));
    } else {
        this->host_ = "huggingface.co";
    }
}

std::string HfApiClient::GetHost() const {
    // 如果设置了 HF_ENDPOINT 环境变量，使用它；否则使用默认的 huggingface.co
    if (const char* hf_endpoint = std::getenv("HF_ENDPOINT")) {
        std::string host, path;
        std::tie(host, path) = ParseUrl(std::string(hf_endpoint));
        if (!host.empty()) {
            return host;
        }
    }
    return "huggingface.co";
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
    // TODO: Update this method to use the new ModelFileDownloader API
    // For now, just log that this method needs to be updated
    printf("DownloadRepo method needs to be updated to use new ModelFileDownloader API\n");
    printf("Model: %s, SHA: %s, Siblings: %zu\n", 
           repo_info.model_id.c_str(), repo_info.sha.c_str(), repo_info.siblings.size());
    
    // This method needs to be completely rewritten to use the new Android-style API
    // The old RemoteModelDownloader::DownloadWithRetries method no longer exists
}

}
