//
// Created by ruoyi.sjd on 2025/5/8.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//
#pragma once

#include <string>
#include <filesystem>
#include <memory>
#include <openssl/sha.h>
#include <openssl/evp.h>

namespace fs = std::filesystem;

namespace mnn::downloader {

class HfShaVerifier {
public:
    /**
     * Verify file hash against expected ETag
     * @param etag The expected ETag from HuggingFace API
     * @param file Path to the file to verify
     * @return true if verification passes, false otherwise
     */
    static bool verify(const std::string& etag, const fs::path& file);
    
    /**
     * Calculate Git SHA1 hash (blob format)
     * @param file Path to the file
     * @return SHA1 hash as hex string
     */
    static std::string gitSha1Hex(const fs::path& file);
    
    /**
     * Calculate SHA-256 hash
     * @param file Path to the file
     * @return SHA-256 hash as hex string
     */
    static std::string sha256Hex(const fs::path& file);
    
    /**
     * Calculate digest hash for a given algorithm
     * @param file Path to the file
     * @param algo Hash algorithm name (e.g., "SHA-256")
     * @return Hash as hex string
     */
    static std::string digestHex(const fs::path& file, const std::string& algo);

private:
    /**
     * Convert bytes to hex string
     * @param data Pointer to byte data
     * @param length Length of data
     * @return Hex string representation
     */
    static std::string bytesToHex(const unsigned char* data, size_t length);
    
    /**
     * Convert string to lowercase
     * @param str Input string
     * @return Lowercase string
     */
    static std::string toLower(const std::string& str);
};

} // namespace mnn::downloader

