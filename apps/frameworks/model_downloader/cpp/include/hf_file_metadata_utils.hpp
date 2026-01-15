//
// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#pragma once

#include "hf_file_metadata.hpp"
#include "hf_api_client.hpp"
#include "httplib.h"
#include <string>
#include <memory>

namespace mnn::downloader {

// HuggingFace file metadata utilities
class HfFileMetadataUtils {
public:
    // Get file metadata from HuggingFace
    static HfFileMetadata GetFileMetadata(const std::string& url, std::string& error_info);
    
    // Get file metadata using custom HTTP client
    static HfFileMetadata GetFileMetadata(const std::string& url, 
                                         std::shared_ptr<httplib::SSLClient> client,
                                         std::string& error_info);
    
private:
    // Parse URL to extract host and path
    static bool ParseUrl(const std::string& url, std::string& host, std::string& path);
    
    // Parse content length from header
    static int64_t ParseContentLength(const std::string& content_length);
    
    // Normalize ETag (remove quotes)
    static std::string NormalizeETag(const std::string& etag);
    
    // Handle redirects and extract location
    static std::string HandleRedirects(const std::string& original_url, 
                                      const std::string& location_header);
    
    // Parse HuggingFace specific headers
    static void ParseHuggingFaceHeaders(const httplib::Headers& headers, 
                                       HfFileMetadata& metadata);
    
    // Helper method to get header value from multimap
    static std::string GetHeaderValue(const httplib::Headers& headers, const std::string& key);
    
    // Print headers for debugging
    static void PrintHeaders(const httplib::Headers& headers, const std::string& url);
    
    // Constants for HuggingFace headers
    static constexpr const char* kHeaderXRepoCommit = "x-repo-commit";
    static constexpr const char* kHeaderXLinkedEtag = "x-linked-etag";
    static constexpr const char* kHeaderXLinkedSize = "x-linked-size";
    static constexpr const char* kHeaderEtag = "ETag";
    static constexpr const char* kHeaderContentLength = "Content-Length";
    static constexpr const char* kHeaderLocation = "Location";
    static constexpr const char* kHeaderAcceptEncoding = "Accept-Encoding";
};

} // namespace mnn::downloader
