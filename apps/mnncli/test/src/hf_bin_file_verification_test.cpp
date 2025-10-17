//
// Created by AI Assistant on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//
// Test to investigate SHA1 verification issues with .bin files vs .json files
// Focus on embeddings_bf16.bin from taobao-mnn/SmolVLM-256M-Instruct-MNN

#include "../include/hf_file_metadata_utils.hpp"
#include "../include/hf_api_client.hpp"
#include "../include/hf_sha_verifier.hpp"
#include "../include/log_utils.hpp"
#include "../include/model_file_downloader.hpp"
#include <iostream>
#include <cassert>
#include <string>
#include <filesystem>
#include <chrono>
#include <thread>
#include <fstream>
#include <iomanip>

using namespace mnncli;

// Test helper functions
void assertTrue(bool condition, const std::string& testName) {
  if (!condition) {
    std::cerr << "❌ Test failed: " << testName << std::endl;
    assert(false);
  } else {
    std::cout << "✅ " << testName << " passed" << std::endl;
  }
}

void assertFalse(bool condition, const std::string& testName) {
  if (condition) {
    std::cerr << "❌ Test failed: " << testName << std::endl;
    assert(false);
  } else {
    std::cout << "✅ " << testName << " passed" << std::endl;
  }
}

void printTestHeader(const std::string& testName) {
  std::cout << "\n==========================================" << std::endl;
  std::cout << "Testing: " << testName << std::endl;
  std::cout << "==========================================" << std::endl;
}

void printSectionHeader(const std::string& section) {
  std::cout << "\n------------------------------------------" << std::endl;
  std::cout << section << std::endl;
  std::cout << "------------------------------------------" << std::endl;
}

// Compare two files metadata side by side
void compareFileMetadata(const std::string& file1_name, const HfFileMetadata& meta1,
                        const std::string& file2_name, const HfFileMetadata& meta2) {
  printSectionHeader("Metadata Comparison");
  
  std::cout << std::left << std::setw(30) << "Property" 
            << std::setw(50) << file1_name 
            << std::setw(50) << file2_name << std::endl;
  std::cout << std::string(130, '-') << std::endl;
  
  // Compare locations (URLs)
  std::cout << std::left << std::setw(30) << "Location (URL):" << std::endl;
  std::cout << std::left << std::setw(30) << "" 
            << std::setw(50) << meta1.location.substr(0, 48) 
            << std::setw(50) << meta2.location.substr(0, 48) << std::endl;
  
  // Check if locations are CDN or direct
  bool meta1_is_cdn = (meta1.location.find("cdn-lfs") != std::string::npos) ||
                      (meta1.location.find("cdn.") != std::string::npos);
  bool meta2_is_cdn = (meta2.location.find("cdn-lfs") != std::string::npos) ||
                      (meta2.location.find("cdn.") != std::string::npos);
  
  std::cout << std::left << std::setw(30) << "  Is CDN:" 
            << std::setw(50) << (meta1_is_cdn ? "YES ⚠️" : "NO")
            << std::setw(50) << (meta2_is_cdn ? "YES ⚠️" : "NO") << std::endl;
  
  // Compare ETags
  std::cout << std::left << std::setw(30) << "ETag:" 
            << std::setw(50) << meta1.etag.substr(0, 48)
            << std::setw(50) << meta2.etag.substr(0, 48) << std::endl;
  
  std::cout << std::left << std::setw(30) << "  ETag Length:" 
            << std::setw(50) << std::to_string(meta1.etag.length())
            << std::setw(50) << std::to_string(meta2.etag.length()) << std::endl;
  
  std::cout << std::left << std::setw(30) << "  Hash Type:" 
            << std::setw(50) << (meta1.etag.length() == 40 ? "SHA-1" : (meta1.etag.length() == 64 ? "SHA-256" : "Unknown"))
            << std::setw(50) << (meta2.etag.length() == 40 ? "SHA-1" : (meta2.etag.length() == 64 ? "SHA-256" : "Unknown")) << std::endl;
  
  // Compare sizes
  std::cout << std::left << std::setw(30) << "Size (bytes):" 
            << std::setw(50) << std::to_string(meta1.size)
            << std::setw(50) << std::to_string(meta2.size) << std::endl;
  
  std::cout << std::left << std::setw(30) << "  Size (MB):" 
            << std::setw(50) << std::to_string(meta1.size / (1024 * 1024))
            << std::setw(50) << std::to_string(meta2.size / (1024 * 1024)) << std::endl;
  
  // Compare commit hashes
  std::cout << std::left << std::setw(30) << "Commit Hash:" 
            << std::setw(50) << (meta1.commit_hash.empty() ? "(empty)" : meta1.commit_hash.substr(0, 48))
            << std::setw(50) << (meta2.commit_hash.empty() ? "(empty)" : meta2.commit_hash.substr(0, 48)) << std::endl;
  
  std::cout << std::string(130, '-') << std::endl;
  
  // Highlight differences
  if (meta1_is_cdn != meta2_is_cdn) {
    std::cout << "⚠️  WARNING: One file uses CDN, the other doesn't!" << std::endl;
    std::cout << "   This may cause different verification behavior." << std::endl;
  }
  
  if (meta1.etag.length() != meta2.etag.length()) {
    std::cout << "⚠️  WARNING: Different ETag hash types detected!" << std::endl;
  }
}

// Test metadata retrieval with detailed header inspection
void testFileMetadataDetailed(const std::string& model_id, const std::string& filename) {
  printTestHeader("Detailed Metadata Test: " + filename);

  try {
    std::string file_url = "https://huggingface.co/" + model_id + "/resolve/main/" + filename;
    std::cout << "Testing URL: " << file_url << std::endl;

    // Get metadata
    std::string error_info;
    HfFileMetadata metadata = HfFileMetadataUtils::GetFileMetadata(file_url, error_info);

    std::cout << "Error info: '" << error_info << "'" << std::endl;
    std::cout << "Metadata location: '" << metadata.location << "'" << std::endl;
    std::cout << "Metadata etag: '" << metadata.etag << "'" << std::endl;
    std::cout << "Metadata size: " << metadata.size << std::endl;

    assertTrue(error_info.empty(), "Metadata retrieval succeeded for " + filename);
    assertTrue(!metadata.location.empty(), "Location is not empty");
    assertTrue(!metadata.etag.empty(), "ETag is not empty");
    assertTrue(metadata.size > 0, "File size is positive");
    
    std::cout << "\nFile metadata for " << filename << ":" << std::endl;
    std::cout << "  Location: " << metadata.location << std::endl;
    std::cout << "  ETag: " << metadata.etag << std::endl;
    std::cout << "  ETag Length: " << metadata.etag.length() << " (Type: " 
              << (metadata.etag.length() == 40 ? "SHA-1" : (metadata.etag.length() == 64 ? "SHA-256" : "Unknown"))
              << ")" << std::endl;
    std::cout << "  Size: " << metadata.size << " bytes (" << (metadata.size / (1024.0 * 1024.0)) << " MB)" << std::endl;
    std::cout << "  Commit Hash: " << (metadata.commit_hash.empty() ? "(empty)" : metadata.commit_hash) << std::endl;
    
    // Check if URL is CDN
    bool is_cdn = (metadata.location.find("cdn-lfs") != std::string::npos) ||
                  (metadata.location.find("cdn.") != std::string::npos);
    
    if (is_cdn) {
      std::cout << "  ⚠️  File is served from CDN!" << std::endl;
    } else {
      std::cout << "  ✅ File is served directly from HuggingFace" << std::endl;
    }
    
    std::cout << "\n✅ Detailed metadata test passed for " << filename << std::endl;
    
  } catch (const std::exception& e) {
    std::cerr << "❌ Detailed metadata test failed for " << filename << ": " << e.what() << std::endl;
    assert(false);
  }
}

// Test file download and SHA verification
void testFileDownloadAndVerification(const std::string& model_id, const std::string& filename, 
                                    const std::string& cache_path) {
  printTestHeader("Download and Verification Test: " + filename);
  
  try {
    std::string file_url = "https://huggingface.co/" + model_id + "/resolve/main/" + filename;
    std::filesystem::path file_cache_path = std::filesystem::path(cache_path) / filename;
    
    // Clean up any existing test file
    if (std::filesystem::exists(file_cache_path)) {
      std::filesystem::remove(file_cache_path);
    }
    
    // Create download directory
    std::filesystem::create_directories(cache_path);
    
    std::cout << "Downloading: " << file_url << std::endl;
    std::cout << "To: " << file_cache_path << std::endl;
    
    // Get metadata first
    std::string error_info;
    HfFileMetadata metadata = HfFileMetadataUtils::GetFileMetadata(file_url, error_info);
    
    assertTrue(error_info.empty(), "Metadata retrieval succeeded");
    
    std::cout << "\nDownload metadata:" << std::endl;
    std::cout << "  Expected size: " << metadata.size << " bytes" << std::endl;
    std::cout << "  Expected ETag: " << metadata.etag << std::endl;
    std::cout << "  Download URL: " << metadata.location << std::endl;
    
    // Create download task
    FileDownloadTask task;
    task.etag = metadata.etag;
    task.relativePath = filename;
    task.fileMetadata = metadata;
    task.downloadPath = file_cache_path;
    task.downloadedSize = 0;
    
    // Create a download listener
    class TestDownloadListener : public FileDownloadListener {
    public:
      TestDownloadListener(int64_t expected_size, const std::string& filename) 
        : expected_size_(expected_size), filename_(filename) {}
      
      bool onDownloadDelta(const std::string* fileName, int64_t downloadedBytes, 
                          int64_t totalBytes, int64_t delta) override {
        if (totalBytes > 0) {
          float progress = static_cast<float>(downloadedBytes) / totalBytes;
          
          // Show progress every 10%
          int current_pct = static_cast<int>(progress * 100);
          if (current_pct >= last_shown_pct_ + 10 || current_pct >= 99) {
            std::cout << "  Progress: " << current_pct << "% (" 
                     << (downloadedBytes / (1024 * 1024)) << " MB / "
                     << (totalBytes / (1024 * 1024)) << " MB)" << std::endl;
            last_shown_pct_ = current_pct;
          }
        }
        return false; // Don't pause
      }
      
    private:
      int64_t expected_size_;
      std::string filename_;
      int last_shown_pct_ = -10;
    };
    
    TestDownloadListener listener(metadata.size, filename);
    
    // Download the file
    ModelFileDownloader downloader;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::cout << "\nStarting download..." << std::endl;
    downloader.DownloadFile(task, listener);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "\n✅ Download completed in " << duration.count() << " ms" << std::endl;
    
    // Verify file was downloaded
    assertTrue(std::filesystem::exists(file_cache_path), "Downloaded file exists");
    
    // Get actual file size
    int64_t actual_size = std::filesystem::file_size(file_cache_path);
    std::cout << "Actual file size: " << actual_size << " bytes" << std::endl;
    std::cout << "Expected file size: " << metadata.size << " bytes" << std::endl;
    
    // Validate file size
    assertTrue(actual_size == metadata.size, "File size matches expected size");
    
    // Now perform SHA verification
    printSectionHeader("SHA Verification");
    
    std::cout << "Verifying file integrity..." << std::endl;
    std::cout << "  File: " << file_cache_path << std::endl;
    std::cout << "  Expected ETag: " << metadata.etag << std::endl;
    std::cout << "  ETag Length: " << metadata.etag.length() << std::endl;
    std::cout << "  Hash Type: " << (metadata.etag.length() == 40 ? "SHA-1 (Git format)" : 
                                   (metadata.etag.length() == 64 ? "SHA-256" : "Unknown")) << std::endl;
    
    // Calculate the actual hash
    std::string actual_hash;
    if (metadata.etag.length() == 40) {
      std::cout << "\nCalculating Git SHA-1 hash..." << std::endl;
      actual_hash = HfShaVerifier::gitSha1Hex(file_cache_path);
    } else if (metadata.etag.length() == 64) {
      std::cout << "\nCalculating SHA-256 hash..." << std::endl;
      actual_hash = HfShaVerifier::sha256Hex(file_cache_path);
    } else {
      std::cout << "❌ Unknown hash type!" << std::endl;
      assert(false);
    }
    
    std::cout << "  Calculated hash: " << actual_hash << std::endl;
    std::cout << "  Expected hash:   " << metadata.etag << std::endl;
    
    // Perform verification
    bool verify_result = HfShaVerifier::verify(metadata.etag, file_cache_path);
    
    if (verify_result) {
      std::cout << "\n✅ SHA verification PASSED for " << filename << std::endl;
    } else {
      std::cout << "\n❌ SHA verification FAILED for " << filename << std::endl;
      std::cout << "   This indicates a potential issue with:" << std::endl;
      std::cout << "   1. CDN serving different content" << std::endl;
      std::cout << "   2. ETag mismatch from metadata vs actual file" << std::endl;
      std::cout << "   3. Hash calculation logic issue" << std::endl;
      
      // Additional debugging
      std::cout << "\nDebug information:" << std::endl;
      std::cout << "  File size: " << actual_size << " bytes" << std::endl;
      std::cout << "  First 16 bytes of file: ";
      std::ifstream file(file_cache_path, std::ios::binary);
      char buffer[16];
      file.read(buffer, 16);
      for (int i = 0; i < 16 && i < file.gcount(); i++) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') 
                 << (static_cast<unsigned int>(static_cast<unsigned char>(buffer[i]))) << " ";
      }
      std::cout << std::dec << std::endl;
      file.close();
    }
    
    assertTrue(verify_result, "SHA verification passed for " + filename);
    
  } catch (const std::exception& e) {
    std::cerr << "❌ Download and verification test failed for " << filename << ": " 
              << e.what() << std::endl;
    assert(false);
  }
}

// Main comparison test
void testBinVsJsonComparison() {
  printTestHeader("BIN vs JSON File Comparison");

  try {
    std::string model_id = "taobao-mnn/SmolVLM-256M-Instruct-MNN";
    std::string bin_file = "embeddings_bf16.bin";
    std::string json_file = "config.json";

    std::cout << "Model: " << model_id << std::endl;
    std::cout << "Comparing: " << bin_file << " vs " << json_file << std::endl;

    // Get metadata for both files
    std::string bin_url = "https://huggingface.co/" + model_id + "/resolve/main/" + bin_file;
    std::string json_url = "https://huggingface.co/" + model_id + "/resolve/main/" + json_file;

    std::cout << "BIN URL: " << bin_url << std::endl;
    std::cout << "JSON URL: " << json_url << std::endl;

    std::string error_info_bin, error_info_json;
    HfFileMetadata bin_metadata = HfFileMetadataUtils::GetFileMetadata(bin_url, error_info_bin);
    HfFileMetadata json_metadata = HfFileMetadataUtils::GetFileMetadata(json_url, error_info_json);

    std::cout << "BIN Error info: '" << error_info_bin << "'" << std::endl;
    std::cout << "BIN Metadata location: '" << bin_metadata.location << "'" << std::endl;
    std::cout << "BIN Metadata etag: '" << bin_metadata.etag << "'" << std::endl;
    std::cout << "BIN Metadata size: " << bin_metadata.size << std::endl;

    std::cout << "JSON Error info: '" << error_info_json << "'" << std::endl;
    std::cout << "JSON Metadata location: '" << json_metadata.location << "'" << std::endl;
    std::cout << "JSON Metadata etag: '" << json_metadata.etag << "'" << std::endl;
    std::cout << "JSON Metadata size: " << json_metadata.size << std::endl;

    assertTrue(error_info_bin.empty(), "BIN metadata retrieval succeeded");
    assertTrue(error_info_json.empty(), "JSON metadata retrieval succeeded");
    
    // Compare metadata side by side
    compareFileMetadata(bin_file, bin_metadata, json_file, json_metadata);
    
    std::cout << "\n✅ Comparison test completed" << std::endl;
    
  } catch (const std::exception& e) {
    std::cerr << "❌ Comparison test failed: " << e.what() << std::endl;
    assert(false);
  }
}

int main(int argc, char* argv[]) {
  std::cout << "==========================================" << std::endl;
  std::cout << "HuggingFace BIN File Verification Test" << std::endl;
  std::cout << "Model: taobao-mnn/SmolVLM-256M-Instruct-MNN" << std::endl;
  std::cout << "Purpose: Investigate SHA1 verification issues" << std::endl;
  std::cout << "==========================================" << std::endl;

  // Parse command line arguments
  bool verbose = false;
  bool skip_download = false;
  std::string cache_path = "/tmp/mnncli_bin_verification_test";

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "-v" || arg == "--verbose") {
      verbose = true;
    } else if (arg == "--skip-download") {
      skip_download = true;
    } else if (arg == "--cache-path" && i + 1 < argc) {
      cache_path = argv[++i];
    }
  }

  // Enable verbose logging
  mnncli::LogUtils::SetVerbose(true);

  if (verbose) {
    std::cout << "Verbose mode enabled" << std::endl;
  }
  
  std::cout << "Cache path: " << cache_path << std::endl;
  std::cout << std::endl;
  
  try {
    std::string model_id = "taobao-mnn/SmolVLM-256M-Instruct-MNN";
    
    // Test 1: Compare metadata for .bin vs .json files
    testBinVsJsonComparison();
    
    // Test 2: Detailed metadata for each file
    testFileMetadataDetailed(model_id, "embeddings_bf16.bin");
    testFileMetadataDetailed(model_id, "config.json");
    
    if (!skip_download) {
      // Test 3: Download and verify .json file (should pass)
      testFileDownloadAndVerification(model_id, "config.json", cache_path);
      
      // Test 4: Download and verify .bin file (investigate failure)
      testFileDownloadAndVerification(model_id, "embeddings_bf16.bin", cache_path);
    } else {
      std::cout << "\n⚠️  Skipping download tests (--skip-download flag)" << std::endl;
    }
    
    std::cout << "\n==========================================" << std::endl;
    std::cout << "🎉 All tests completed!" << std::endl;
    std::cout << "==========================================" << std::endl;
    
    return 0;
    
  } catch (const std::exception& e) {
    std::cerr << "\n❌ Test suite failed with exception: " << e.what() << std::endl;
    return 1;
  }
}

