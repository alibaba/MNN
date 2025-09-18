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

namespace mnncli {

// HuggingFace file metadata utilities
class HfFileMetadataUtils {
public:
    // Get file metadata from HuggingFace
    static HfFileMetadata getFileMetadata(const std::string& url, std::string& error_info);
    
    // Get file metadata using custom HTTP client
    static HfFileMetadata getFileMetadata(const std::string& url, 
                                         std::shared_ptr<
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
                                         httplib::SSLClient
#else
                                         httplib::Client
#endif
                                         > client,
                                         std::string& error_info);
    
private:
    // Parse URL to extract host and path
    static bool parseUrl(const std::string& url, std::string& host, std::string& path);
    
    // Parse content length from header
    static int64_t parseContentLength(const std::string& content_length);
    
    // Normalize ETag (remove quotes)
    static std::string normalizeETag(const std::string& etag);
    
    // Handle redirects and extract location
    static std::string handleRedirects(const std::string& original_url, 
                                      const std::string& location_header);
    
    // Parse HuggingFace specific headers
    static void parseHuggingFaceHeaders(const httplib::Headers& headers, 
                                       HfFileMetadata& metadata);
    
    // Helper method to get header value from multimap
    static std::string getHeaderValue(const httplib::Headers& headers, const std::string& key);
    
    // Constants for HuggingFace headers
    static constexpr const char* HEADER_X_REPO_COMMIT = "x-repo-commit";
    static constexpr const char* HEADER_X_LINKED_ETAG = "x-linked-etag";
    static constexpr const char* HEADER_X_LINKED_SIZE = "x-linked-size";
    static constexpr const char* HEADER_ETAG = "ETag";
    static constexpr const char* HEADER_CONTENT_LENGTH = "Content-Length";
    static constexpr const char* HEADER_LOCATION = "Location";
    static constexpr const char* HEADER_ACCEPT_ENCODING = "Accept-Encoding";
};

} // namespace mnncli
