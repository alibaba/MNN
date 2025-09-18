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

namespace mnncli {

HfFileMetadata HfFileMetadataUtils::getFileMetadata(const std::string& url, std::string& error_info) {
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
    return getFileMetadata(url, default_client, error_info);
}

HfFileMetadata HfFileMetadataUtils::getFileMetadata(const std::string& url, 
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
        if (!parseUrl(url, host, path)) {
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
        
        // Set headers for the request
        httplib::Headers headers;
        headers.emplace("User-Agent", "MNN-CLI/1.0");
        headers.emplace("Accept", "*/*");
        
        // Make the HEAD request to get metadata
        auto res = client_ptr->Head(path, headers);
        if (!res) {
            error_info = "Failed to connect to server";
            return {};
        }
        
        if (res->status != 200) {
            error_info = "HTTP error: " + std::to_string(res->status);
            return {};
        }
        
        // Extract metadata from response headers
        HfFileMetadata metadata;
        metadata.location = url;
        
        // Get file size
        auto content_length = res->get_header_value("Content-Length");
        if (!content_length.empty()) {
            try {
                metadata.size = std::stoull(content_length);
            } catch (const std::exception& e) {
                // Ignore parsing errors for size
            }
        }
        
        // Get ETag for hash verification
        auto etag = res->get_header_value("ETag");
        if (!etag.empty()) {
            metadata.etag = etag;
        }
        
        return metadata;
        
    } catch (const std::exception& e) {
        error_info = "Exception: " + std::string(e.what());
        return {};
    }
}

bool HfFileMetadataUtils::parseUrl(const std::string& url, std::string& host, std::string& path) {
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

int64_t HfFileMetadataUtils::parseContentLength(const std::string& content_length) {
    if (content_length.empty()) {
        return 0;
    }
    
    try {
        return std::stoll(content_length);
    } catch (const std::exception&) {
        return 0;
    }
}

std::string HfFileMetadataUtils::normalizeETag(const std::string& etag) {
    if (etag.empty()) {
        return etag;
    }
    
    // Remove quotes if present
    if (etag.length() >= 2 && etag.front() == '"' && etag.back() == '"') {
        return etag.substr(1, etag.length() - 2);
    }
    
    return etag;
}

std::string HfFileMetadataUtils::handleRedirects(const std::string& original_url, 
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

void HfFileMetadataUtils::parseHuggingFaceHeaders(const httplib::Headers& headers, 
                                                   HfFileMetadata& metadata) {
    // Parse HuggingFace specific headers first, fallback to standard headers
    
    // ETag handling
    std::string linked_etag = getHeaderValue(headers, HEADER_X_LINKED_ETAG);
    std::string standard_etag = getHeaderValue(headers, HEADER_ETAG);
    
    if (!linked_etag.empty()) {
        metadata.etag = normalizeETag(linked_etag);
    } else if (!standard_etag.empty()) {
        metadata.etag = normalizeETag(standard_etag);
    }
    
    // Size handling
    std::string linked_size = getHeaderValue(headers, HEADER_X_LINKED_SIZE);
    std::string content_length = getHeaderValue(headers, HEADER_CONTENT_LENGTH);
    
    if (!linked_size.empty()) {
        metadata.size = parseContentLength(linked_size);
    } else if (!content_length.empty()) {
        metadata.size = parseContentLength(content_length);
    }
    
    // Commit hash
    std::string commit_hash = getHeaderValue(headers, HEADER_X_REPO_COMMIT);
    if (!commit_hash.empty()) {
        metadata.commit_hash = commit_hash;
    }
}

// Helper method to get header value from multimap
std::string HfFileMetadataUtils::getHeaderValue(const httplib::Headers& headers, const std::string& key) {
    auto it = headers.find(key);
    if (it != headers.end()) {
        return it->second;
    }
    return "";
}

} // namespace mnncli
