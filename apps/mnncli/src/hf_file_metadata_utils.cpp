//
// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#include "hf_file_metadata_utils.hpp"
#include "httplib.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include "log_utils.hpp"

namespace mnncli {

// Helper function to normalize ETag (remove quotes if present, like Android)
std::string normalizeETag(const std::string& etag) {
    if (etag.empty()) {
        return etag;
    }
    
    std::string result = etag;
    // Remove leading quote
    if (!result.empty() && result.front() == '"') {
        result = result.substr(1);
    }
    // Remove trailing quote
    if (!result.empty() && result.back() == '"') {
        result = result.substr(0, result.length() - 1);
    }
    
    return result;
}

HfFileMetadata HfFileMetadataUtils::GetFileMetadata(const std::string& url, std::string& error_info) {
    // Create a default client if none provided
    std::shared_ptr<
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
        httplib::SSLClient
#else
        httplib::Client
#endif
    > default_client;
    
    if (!default_client) {
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
        default_client = std::make_shared<httplib::SSLClient>("huggingface.co");
#else
        default_client = std::make_shared<httplib::Client>("huggingface.co");
#endif
    }
    return GetFileMetadata(url, default_client, error_info);
}

HfFileMetadata HfFileMetadataUtils::GetFileMetadata(const std::string& url, 
                                                   std::shared_ptr<
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
                                                   httplib::SSLClient
#else
                                                   httplib::Client
#endif
                                                   > client,
                                                   std::string& error_info) {
    try {
        // Parse the URL to extract host and path
        std::string host, path;
        if (!ParseUrl(url, host, path)) {
            error_info = "Invalid URL format";
            return {};
        }
        
        // Create a local client if none provided
        std::unique_ptr<
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
            httplib::SSLClient
#else
            httplib::Client
#endif
        > local_client;
        
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
        httplib::SSLClient* client_ptr = client.get();
#else
        httplib::Client* client_ptr = client.get();
#endif
        
        if (!client_ptr) {
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
            local_client = std::make_unique<httplib::SSLClient>(host, 443);
#else
            local_client = std::make_unique<httplib::Client>(host, 80);
#endif
            client_ptr = local_client.get();
        }
        
        // Set headers for the request (matching Android implementation)
        httplib::Headers headers;
        headers.emplace("User-Agent", "MNN-CLI/1.0");
        headers.emplace("Accept", "*/*");
        headers.emplace("Accept-Encoding", "identity");  // Critical: prevents gzip compression
        
        // Add Hugging Face authentication token ONLY for huggingface.co requests
        // DO NOT add Authorization header for pre-signed CDN URLs
        if (host.find("huggingface.co") != std::string::npos) {
            if (const char* hf_token = std::getenv("HF_TOKEN")) {
                std::string auth_header = "Bearer " + std::string(hf_token);
                headers.emplace("Authorization", auth_header);
                LOG_DEBUG("[DEBUG] Using HF_TOKEN for authentication (host: " + host + ")");
            }
        } else {
            LOG_DEBUG("[DEBUG] Skipping HF_TOKEN for CDN host: " + host + " (using pre-signed URL)");
        }
        
        // Make the HEAD request to get metadata (following official HF implementation)
        LOG_DEBUG("[DEBUG] Making HEAD request to: " + url);
        auto res = client_ptr->Head(path, headers);
        
        // If HEAD fails with 404, try GET request (some CDN endpoints don't support HEAD)
        if (!res || res->status == 404) {
            LOG_DEBUG("[DEBUG] HEAD request failed (status: " + std::to_string(res ? res->status : -1) + "), trying GET request");
            res = client_ptr->Get(path, headers);
        }
        if (!res) {
            error_info = "Failed to connect to server";
            LOG_DEBUG("[DEBUG] HEAD request failed: " + error_info);
            return {};
        }
        
        // Follow redirects manually (like official HF implementation)
        std::string finalLocation = url;  // Default to original URL
        int redirectCount = 0;
        const int maxRedirects = 5;
        
        while (res->status >= 301 && res->status <= 308 && redirectCount < maxRedirects) {
            std::string location = res->get_header_value("Location");
            if (location.empty()) {
                LOG_DEBUG("[DEBUG] Redirect without Location header, status: " + std::to_string(res->status));
                break;
            }
            
            LOG_DEBUG("[DEBUG] Redirect detected, status: " + std::to_string(res->status) + ", location: " + location);
            
            // Handle relative URLs in redirect Location header
            if (location.front() == '/') {
                // Extract the base URL (scheme + host + port) from the original URL
                size_t schemeEnd = url.find("://");
                if (schemeEnd != std::string::npos) {
                    std::string scheme = url.substr(0, schemeEnd);
                    size_t hostStart = schemeEnd + 3;
                    size_t hostEnd = url.find('/', hostStart);
                    if (hostEnd == std::string::npos) {
                        hostEnd = url.length();
                    }
                    std::string hostPort = url.substr(hostStart, hostEnd - hostStart);
                    finalLocation = scheme + "://" + hostPort + location;
                } else {
                    finalLocation = location;
                }
            } else {
                finalLocation = location;
            }
            
            // Parse the redirect URL and make a new request
            auto [redirectHost, redirectPath] = HfApiClient::ParseUrl(finalLocation);
            LOG_DEBUG("[DEBUG] Following redirect to: " + finalLocation);
            
            // Create new client for redirect
            std::unique_ptr<httplib::Client> redirectClient;
            std::unique_ptr<httplib::SSLClient> redirectSSLClient;
            
            if (finalLocation.find("https://") == 0) {
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
                redirectSSLClient = std::make_unique<httplib::SSLClient>(redirectHost, 443);
                res = redirectSSLClient->Head(redirectPath, headers);
#else
                LOG_DEBUG("[DEBUG] HTTPS redirect but SSL not supported");
                break;
#endif
            } else {
                redirectClient = std::make_unique<httplib::Client>(redirectHost, 80);
                res = redirectClient->Head(redirectPath, headers);
            }
            
            if (!res) {
                LOG_DEBUG("[DEBUG] Redirect request failed");
                break;
            }
            
            redirectCount++;
        }
        
        if (res->status != 200) {
            error_info = "HTTP error: " + std::to_string(res->status);
            LOG_DEBUG("[DEBUG] Request failed with status: " + std::to_string(res->status));
            return {};
        }
        
        // Debug: Print all response headers from final response
        LOG_DEBUG("[DEBUG] Response headers for " + finalLocation + ":");
        for (const auto& header : res->headers) {
            LOG_DEBUG("  " + header.first + ": " + header.second);
        }
        
        // Extract metadata from response headers (matching Android implementation)
        HfFileMetadata metadata;
        metadata.location = finalLocation;  // Use final location (original or redirected)
        
        // Get file size - prioritize x-linked-size over Content-Length (following official HF implementation)
        auto linked_size = res->get_header_value("x-linked-size");
        auto content_length = res->get_header_value("Content-Length");
        
        // Official implementation: r.headers.get("X-Linked-Size") or r.headers.get("Content-Length")
        if (!linked_size.empty()) {
            try {
                metadata.size = std::stoull(linked_size);
            } catch (const std::exception& e) {
                // Ignore parsing errors for size
            }
        } else if (!content_length.empty()) {
            try {
                metadata.size = std::stoull(content_length);
            } catch (const std::exception& e) {
                // Ignore parsing errors for size
            }
        }
        
        // Get ETag for hash verification - prioritize x-linked-etag over ETag (like Android)
        auto linked_etag = res->get_header_value("x-linked-etag");
        auto etag = res->get_header_value("ETag");
        
        if (!linked_etag.empty()) {
            metadata.etag = normalizeETag(linked_etag);
        } else if (!etag.empty()) {
            metadata.etag = normalizeETag(etag);
        }
        
        // Get commit hash (like Android)
        auto commit_hash = res->get_header_value("x-repo-commit");
        if (!commit_hash.empty()) {
            metadata.commit_hash = commit_hash;
        }
        
        return metadata;
        
    } catch (const std::exception& e) {
        error_info = "Exception: " + std::string(e.what());
        return {};
    }
}

bool HfFileMetadataUtils::ParseUrl(const std::string& url, std::string& host, std::string& path) {
    // Simple URL parsing for https://host/path format
    if (url.substr(0, 8) != "https://") {
        return false;
    }
    
    size_t host_start = 8;
    size_t path_start = url.find('/', host_start);
    
    if (path_start == std::string::npos) {
        host = url.substr(host_start);
        path = "/";
    } else {
        host = url.substr(host_start, path_start - host_start);
        path = url.substr(path_start);
    }
    
    return !host.empty();
}

int64_t HfFileMetadataUtils::ParseContentLength(const std::string& content_length) {
    if (content_length.empty()) {
        return 0;
    }
    
    try {
        return std::stoll(content_length);
    } catch (const std::exception&) {
        return 0;
    }
}

std::string HfFileMetadataUtils::NormalizeETag(const std::string& etag) {
    if (etag.empty()) {
        return etag;
    }
    
    // Remove quotes if present
    if (etag.length() >= 2 && etag.front() == '"' && etag.back() == '"') {
        return etag.substr(1, etag.length() - 2);
    }
    
    return etag;
}

std::string HfFileMetadataUtils::HandleRedirects(const std::string& original_url, 
                                                const std::string& location_header) {
    if (location_header.empty()) {
        return original_url;
    }
    
    // If location is absolute URL, use it directly
    if (location_header.find("http://") == 0 || location_header.find("https://") == 0) {
        return location_header;
    }
    
    // If location is relative, construct absolute URL
    if (location_header.front() == '/') {
        // Extract base URL (scheme + host + port) from original URL
        auto [host, path] = HfApiClient::ParseUrl(original_url);
        if (!host.empty()) {
            return "https://" + host + location_header;
        }
    }
    
    // For relative paths without leading slash, append to original path
    auto [host, path] = HfApiClient::ParseUrl(original_url);
    if (!host.empty()) {
        // Find the last slash in the path and truncate there
        size_t last_slash = path.find_last_of('/');
        if (last_slash != std::string::npos) {
            std::string base_path = path.substr(0, last_slash + 1);
            return "https://" + host + base_path + location_header;
        }
    }
    
    return original_url;
}

void HfFileMetadataUtils::ParseHuggingFaceHeaders(const httplib::Headers& headers, 
                                                   HfFileMetadata& metadata) {
    // Parse HuggingFace specific headers first, fallback to standard headers
    
    // ETag handling
    std::string linked_etag = GetHeaderValue(headers, kHeaderXLinkedEtag);
    std::string standard_etag = GetHeaderValue(headers, kHeaderEtag);
    
    if (!linked_etag.empty()) {
        metadata.etag = normalizeETag(linked_etag);
    } else if (!standard_etag.empty()) {
        metadata.etag = normalizeETag(standard_etag);
    }
    
    // Size handling
    std::string linked_size = GetHeaderValue(headers, kHeaderXLinkedSize);
    std::string content_length = GetHeaderValue(headers, kHeaderContentLength);
    
    if (!linked_size.empty()) {
        metadata.size = ParseContentLength(linked_size);
    } else if (!content_length.empty()) {
        metadata.size = ParseContentLength(content_length);
    }
    
    // Commit hash
    std::string commit_hash = GetHeaderValue(headers, kHeaderXRepoCommit);
    if (!commit_hash.empty()) {
        metadata.commit_hash = commit_hash;
    }
}

// Helper method to get header value from multimap
std::string HfFileMetadataUtils::GetHeaderValue(const httplib::Headers& headers, const std::string& key) {
    auto it = headers.find(key);
    if (it != headers.end()) {
        return it->second;
    }
    return "";
}

} // namespace mnncli
