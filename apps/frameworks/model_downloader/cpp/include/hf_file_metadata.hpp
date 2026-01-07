//
// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#pragma once

#include <string>

namespace mnn::downloader {

// HuggingFace file metadata structure
struct HfFileMetadata {
    std::string location;      // URL after redirects
    std::string etag;          // Normalized ETag
    int64_t size;              // File size in bytes
    std::string commit_hash;   // Repository commit hash
    
    HfFileMetadata() : size(0) {}
    
    // Check if metadata is valid
    bool isValid() const {
        return !location.empty() && !etag.empty() && size > 0;
    }
};

} // namespace mnn::downloader
