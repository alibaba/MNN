//
// Created by ruoyi.sjd on 2025/5/8.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//
#include "hf_sha_verifier.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <iostream>

namespace mnncli {

bool HfShaVerifier::verify(const std::string& etag, const fs::path& file) {
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    if (!fs::exists(file)) {
        std::cerr << "HfShaVerifier: File " << file.string() << " does not exist" << std::endl;
        return false;
    }
    
    std::string expected = toLower(etag);
    // Remove quotes from ETag if present
    if (!expected.empty() && expected.front() == '"') {
        expected = expected.substr(1);
    }
    if (!expected.empty() && expected.back() == '"') {
        expected = expected.substr(0, expected.length() - 1);
    }
    
    std::string actual;
    if (expected.length() == 40) {
        actual = gitSha1Hex(file);
    } else if (expected.length() == 64) {
        actual = sha256Hex(file);
    } else {
        std::cerr << "HfShaVerifier: Unexpected ETag length: " << expected.length() << std::endl;
        return false;
    }
    
    std::cout << "HfShaVerifier: Verifying " << file.string() << ": expected=" << expected << " actual=" << actual << std::endl;
    return expected == actual;
#else
    // SHA verification not available without OpenSSL
    std::cerr << "HfShaVerifier: SHA verification not available without OpenSSL support" << std::endl;
    return false;
#endif
}

std::string HfShaVerifier::gitSha1Hex(const fs::path& file) {
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    try {
        // Get file size
        auto file_size = fs::file_size(file);
        
        // Create SHA1 context
        EVP_MD_CTX* ctx = EVP_MD_CTX_new();
        if (!ctx) {
            throw std::runtime_error("Failed to create EVP context");
        }
        
        // Initialize SHA1
        if (EVP_DigestInit_ex(ctx, EVP_sha1(), nullptr) != 1) {
            EVP_MD_CTX_free(ctx);
            throw std::runtime_error("Failed to initialize SHA1");
        }
        
        // Add "blob {size}\0" prefix (Git format)
        std::string prefix = "blob " + std::to_string(file_size);
        if (EVP_DigestUpdate(ctx, prefix.c_str(), prefix.length()) != 1) {
            EVP_MD_CTX_free(ctx);
            throw std::runtime_error("Failed to update SHA1 with prefix");
        }
        // Add null terminator
        if (EVP_DigestUpdate(ctx, "\0", 1) != 1) {
            EVP_MD_CTX_free(ctx);
            throw std::runtime_error("Failed to update SHA1 with null terminator");
        }
        
        // Read and hash file content
        std::ifstream file_stream(file, std::ios::binary);
        if (!file_stream) {
            EVP_MD_CTX_free(ctx);
            throw std::runtime_error("Failed to open file for reading");
        }
        
        const size_t buffer_size = 8192;
        char buffer[buffer_size];
        
        while (file_stream.read(buffer, buffer_size)) {
            if (EVP_DigestUpdate(ctx, buffer, file_stream.gcount()) != 1) {
                EVP_MD_CTX_free(ctx);
                throw std::runtime_error("Failed to update SHA1 with file content");
            }
        }
        
        // Finalize hash
        unsigned char hash[EVP_MAX_MD_SIZE];
        unsigned int hash_length;
        if (EVP_DigestFinal_ex(ctx, hash, &hash_length) != 1) {
            EVP_MD_CTX_free(ctx);
            throw std::runtime_error("Failed to finalize SHA1");
        }
        
        EVP_MD_CTX_free(ctx);
        
        return bytesToHex(hash, hash_length);
        
    } catch (const std::exception& e) {
        std::cerr << "HfShaVerifier: Error calculating Git SHA1: " << e.what() << std::endl;
        return "";
    }
#else
    // SHA calculation not available without OpenSSL
    std::cerr << "HfShaVerifier: SHA calculation not available without OpenSSL support" << std::endl;
    return "";
#endif
}

std::string HfShaVerifier::sha256Hex(const fs::path& file) {
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    return digestHex(file, "SHA-256");
#else
    // SHA calculation not available without OpenSSL
    std::cerr << "HfShaVerifier: SHA calculation not available without OpenSSL support" << std::endl;
    return "";
#endif
}

std::string HfShaVerifier::digestHex(const fs::path& file, const std::string& algo) {
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    try {
        // Create EVP context
        EVP_MD_CTX* ctx = EVP_MD_CTX_new();
        if (!ctx) {
            throw std::runtime_error("Failed to create EVP context");
        }
        
        // Get digest algorithm
        const EVP_MD* md = nullptr;
        if (algo == "SHA-256") {
            md = EVP_sha256();
        } else if (algo == "SHA-1") {
            md = EVP_sha1();
        } else {
            EVP_MD_CTX_free(ctx);
            throw std::runtime_error("Unsupported algorithm: " + algo);
        }
        
        // Initialize digest
        if (EVP_DigestInit_ex(ctx, md, nullptr) != 1) {
            EVP_MD_CTX_free(ctx);
            throw std::runtime_error("Failed to initialize " + algo);
        }
        
        // Read and hash file content
        std::ifstream file_stream(file, std::ios::binary);
        if (!file_stream) {
            EVP_MD_CTX_free(ctx);
            throw std::runtime_error("Failed to open file for reading");
        }
        
        const size_t buffer_size = 8192;
        char buffer[buffer_size];
        
        while (file_stream.read(buffer, buffer_size)) {
            if (EVP_DigestUpdate(ctx, buffer, file_stream.gcount()) != 1) {
                EVP_MD_CTX_free(ctx);
                throw std::runtime_error("Failed to update " + algo + " with file content");
            }
        }
        
        // Finalize hash
        unsigned char hash[EVP_MAX_MD_SIZE];
        unsigned int hash_length;
        if (EVP_DigestFinal_ex(ctx, hash, &hash_length) != 1) {
            EVP_MD_CTX_free(ctx);
            throw std::runtime_error("Failed to finalize " + algo);
        }
        
        EVP_MD_CTX_free(ctx);
        
        return bytesToHex(hash, hash_length);
        
    } catch (const std::exception& e) {
        std::cerr << "HfShaVerifier: Error calculating " << algo << ": " << e.what() << std::endl;
        return "";
    }
#else
    // SHA calculation not available without OpenSSL
    std::cerr << "HfShaVerifier: SHA calculation not available without OpenSSL support" << std::endl;
    return "";
#endif
}

std::string HfShaVerifier::bytesToHex(const unsigned char* data, size_t length) {
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for (size_t i = 0; i < length; ++i) {
        ss << std::setw(2) << static_cast<int>(data[i]);
    }
    return ss.str();
}

std::string HfShaVerifier::toLower(const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

} // namespace mnncli
