
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
#include "mnncli_config.hpp"

namespace mnncli {

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
        printf("Failed to search repos: %s", error_info.c_str());
    }
    return result;
}

std::vector<RepoItem> HfApiClient::SearchReposInner(const std::string& keyword, std::string& error_info) {
    // Create HTTP client
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    httplib::SSLClient cli(GetHost(), 443);
#else
    httplib::Client cli(GetHost(), 80);
#endif
    
    // Configure SSL client with proper timeouts and settings
    cli.set_connection_timeout(30, 0);
    cli.set_read_timeout(30, 0);
    cli.set_write_timeout(30, 0);
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    cli.enable_server_certificate_verification(false);
#endif
    cli.set_keep_alive(true);
    
    httplib::Headers headers;
    headers.emplace("User-Agent", "MNN-CLI/1.0");
    headers.emplace("Accept", "application/json");
    headers.emplace("Connection", "keep-alive");
    
    // Add Hugging Face authentication token if available
    if (const char* hf_token = std::getenv("HF_TOKEN")) {
        std::string auth_header = "Bearer " + std::string(hf_token);
        headers.emplace("Authorization", auth_header);
        std::cout << "🔑 Using HF_TOKEN for authentication" << std::endl;
    } else {
        std::cout << "⚠️  No HF_TOKEN found. Some models may require authentication." << std::endl;
        std::cout << "   To authenticate, export HF_TOKEN=your_token_here" << std::endl;
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

        mnncli::RepoItem repo_info;

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

mnncli::RepoInfo HfApiClient::GetRepoInfo(
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
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
        httplib::SSLClient cli(GetHost(), 443);
#else
        httplib::Client cli(GetHost(), 80);
#endif
        
        // Configure SSL client with proper timeouts and settings
        cli.set_connection_timeout(30, 0);  // 30 seconds connection timeout
        cli.set_read_timeout(30, 0);        // 30 seconds read timeout
        cli.set_write_timeout(30, 0);       // 30 seconds write timeout
        
        // Enable server certificate verification (but allow self-signed for testing)
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
        cli.enable_server_certificate_verification(false);
#endif
        
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
            std::cout << "🔑 Using HF_TOKEN for authentication" << std::endl;
        } else {
            std::cout << "⚠️  No HF_TOKEN found. Some models may require authentication." << std::endl;
            std::cout << "   To authenticate, export HF_TOKEN=your_token_here" << std::endl;
        }
        
        std::cout << "🔍 Making request to: https://" << GetHost() << path << std::endl;
        
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
                std::cout << "❌ " << error_msg << std::endl;
                std::cout << "   Response headers:" << std::endl;
                for (const auto& header : res->headers) {
                    std::cout << "     " << header.first << ": " << header.second << std::endl;
                }
                if (!res->body.empty()) {
                    std::cout << "   Response body preview: " << res->body.substr(0, 200) << "..." << std::endl;
                }
            } else {
                error_msg += " - No response received";
                std::cout << "❌ " << error_msg << std::endl;
                
                // Check for SSL errors
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
                auto ssl_error = cli.get_openssl_verify_result();
                if (ssl_error != 0) {
                    std::cerr << "SSL verification failed: " << ssl_error << std::endl;
                    error_info = "SSL verification failed";
                    return {};
                }
#endif
                
                // Check if it's a connection timeout or other network issue
                std::cout << "   Possible causes:" << std::endl;
                std::cout << "   - Network connectivity issues" << std::endl;
                std::cout << "   - SSL/TLS certificate problems" << std::endl;
                std::cout << "   - Firewall blocking HTTPS traffic" << std::endl;
                std::cout << "   - DNS resolution failure for " << GetHost() << std::endl;
                std::cout << "   - Server is down or unreachable" << std::endl;
                
                // Try a simple connection test
                std::cout << "   Testing basic connectivity..." << std::endl;
                // Create HTTP client
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
                httplib::SSLClient test_cli(GetHost(), 443);
#else
                httplib::Client test_cli(GetHost(), 80);
#endif
                test_cli.set_connection_timeout(10, 0);
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
                test_cli.enable_server_certificate_verification(false);
#endif
                auto test_res = test_cli.Get("/");
                if (test_res) {
                    std::cout << "   Basic connectivity: OK (got response)" << std::endl;
                } else {
                    std::cout << "   Basic connectivity: FAILED (no response)" << std::endl;
                }
            }
            return false;
        }

        std::cout << "✅ API response received successfully" << std::endl;
        std::cout << "   Status: " << res->status << std::endl;
        std::cout << "   Content-Length: " << res->get_header_value("Content-Length") << std::endl;
        std::cout << "   Content-Type: " << res->get_header_value("Content-Type") << std::endl;

        // Parse the JSON response
        rapidjson::Document doc;
        if (doc.Parse(res->body.c_str()).HasParseError()) {
            error_info = "Failed to parse JSON response";
            std::cout << "❌ " << error_info << std::endl;
            std::cout << "   Parse error at position: " << doc.GetErrorOffset() << std::endl;
            std::cout << "   Response preview: " << res->body.substr(0, 200) << "..." << std::endl;
            return false;
        }

        std::cout << "✅ JSON parsed successfully" << std::endl;

        // Extract fields
        if (doc.HasMember("modelId") && doc["modelId"].IsString()) {
            repo_info.model_id = doc["modelId"].GetString();
            std::cout << "   Model ID: " << repo_info.model_id << std::endl;
        } else {
            std::cout << "⚠️  Missing or invalid modelId field" << std::endl;
        }
        
        if (doc.HasMember("sha") && doc["sha"].IsString()) {
            repo_info.sha = doc["sha"].GetString();
            std::cout << "   SHA: " << repo_info.sha << std::endl;
        } else {
            std::cout << "⚠️  Missing or invalid sha field" << std::endl;
        }
        
        if (doc.HasMember("revision") && doc["revision"].IsString()) {
            repo_info.revision = doc["revision"].GetString();
            std::cout << "   Revision: " << repo_info.revision << std::endl;
        } else {
            std::cout << "⚠️  Missing or invalid revision field" << std::endl;
        }
        
        if (doc.HasMember("siblings") && doc["siblings"].IsArray()) {
            const rapidjson::Value& siblings = doc["siblings"];
            std::cout << "   Siblings array size: " << siblings.Size() << std::endl;
            
            for (rapidjson::Value::ConstValueIterator it = siblings.Begin(); it != siblings.End(); ++it) {
                if (it->IsObject() && it->HasMember("rfilename") && (*it)["rfilename"].IsString()) {
                    repo_info.siblings.emplace_back((*it)["rfilename"].GetString());
                }
            }
            std::cout << "   Valid siblings found: " << repo_info.siblings.size() << std::endl;
        } else {
            std::cout << "⚠️  Missing or invalid siblings field" << std::endl;
        }

        return true;
    };

    if (!HfApiClient::PerformRequestWithRetry(request_func, 3, 1)) {
        error_info = "Failed to fetch repository info after retries";
        std::cout << "❌ " << error_info << std::endl;
        return {};
    }

    std::cout << "✅ Repository info retrieved successfully" << std::endl;
    return repo_info;
}

HfApiClient::HfApiClient() {
    cache_path_ = FileUtils::GetBaseCacheDir();
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
