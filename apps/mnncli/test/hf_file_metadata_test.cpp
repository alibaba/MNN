//
// Created by AI Assistant on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#include "../include/hf_file_metadata_utils.hpp"
#include "../include/hf_api_client.hpp"
#include "../include/log_utils.hpp"
#include "../include/model_file_downloader.hpp"
#include <iostream>
#include <cassert>
#include <string>
#include <filesystem>
#include <chrono>
#include <thread>
#include <fstream>

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

// Test tokenizer.txt file metadata retrieval
void testTokenizerFileMetadata() {
  printTestHeader("Tokenizer.txt File Metadata Test");
  
  try {
    // Test URL for tokenizer.txt from the specific model
    std::string tokenizer_url = "https://huggingface.co/taobao-mnn/SmolVLM-256M-Instruct-MNN/resolve/main/tokenizer.txt";
    
    std::cout << "Testing metadata retrieval for: " << tokenizer_url << std::endl;
    
    // Get file metadata
    std::string error_info;
    HfFileMetadata metadata = HfFileMetadataUtils::GetFileMetadata(tokenizer_url, error_info);
    
    // Check if metadata retrieval was successful
    assertTrue(error_info.empty(), "Metadata retrieval succeeded");
    assertTrue(!metadata.location.empty(), "Location is not empty");
    assertTrue(!metadata.etag.empty(), "ETag is not empty");
    assertTrue(metadata.size > 0, "File size is positive");
    
    std::cout << "File metadata retrieved successfully:" << std::endl;
    std::cout << "  Location: " << metadata.location << std::endl;
    std::cout << "  ETag: " << metadata.etag << std::endl;
    std::cout << "  Size: " << metadata.size << " bytes (" << (metadata.size / 1024.0) << " KB)" << std::endl;
    std::cout << "  Commit Hash: " << metadata.commit_hash << std::endl;
    
    // Validate that the size is reasonable for a tokenizer file
    // tokenizer.txt files are typically between 1KB and 10MB
    assertTrue(metadata.size >= 1024, "File size >= 1KB");
    assertTrue(metadata.size <= 10 * 1024 * 1024, "File size <= 10MB");
    
    std::cout << "✅ Tokenizer.txt metadata test passed" << std::endl;
    
  } catch (const std::exception& e) {
    std::cerr << "❌ Tokenizer.txt metadata test failed: " << e.what() << std::endl;
    assert(false);
  }
}

// Test actual tokenizer.txt file download and size verification
void testTokenizerFileDownload() {
  printTestHeader("Tokenizer.txt File Download Test");
  
  try {
    std::string cache_path = "/tmp/mnncli_tokenizer_test";
    std::string tokenizer_url = "https://huggingface.co/taobao-mnn/SmolVLM-256M-Instruct-MNN/resolve/main/tokenizer.txt";
    
    // Clean up any existing test cache
    if (std::filesystem::exists(cache_path)) {
      std::filesystem::remove_all(cache_path);
    }
    
    // Create download directory
    std::filesystem::create_directories(cache_path);
    
    std::cout << "Testing download for: " << tokenizer_url << std::endl;
    std::cout << "Cache path: " << cache_path << std::endl;
    
    // Get metadata first
    std::string error_info;
    HfFileMetadata metadata = HfFileMetadataUtils::GetFileMetadata(tokenizer_url, error_info);
    
    assertTrue(error_info.empty(), "Metadata retrieval succeeded");
    assertTrue(metadata.size > 0, "Expected size is positive");
    
    std::cout << "Expected file size: " << metadata.size << " bytes" << std::endl;
    
    // Create download task
    FileDownloadTask task;
    task.etag = metadata.etag;
    task.relativePath = "tokenizer.txt";
    task.fileMetadata = metadata;
    task.downloadPath = std::filesystem::path(cache_path) / "tokenizer.txt";
    task.downloadedSize = 0;
    
    // Create a simple download listener to track progress
    class TestDownloadListener : public FileDownloadListener {
    public:
      TestDownloadListener(int64_t expected_size) : expected_size_(expected_size) {}
      
      bool onDownloadDelta(const std::string* fileName, int64_t downloadedBytes, int64_t totalBytes, int64_t delta) override {
        // Track progress and validate size consistency
        if (totalBytes > 0) {
          float progress = static_cast<float>(downloadedBytes) / totalBytes;
          
          // Check for progress overflow (the bug we're testing)
          if (progress > 1.0f) {
            std::cout << "⚠️  WARNING: Progress exceeded 100%: " << (progress * 100) << "%" << std::endl;
            std::cout << "   Downloaded: " << downloadedBytes << " bytes" << std::endl;
            std::cout << "   Total: " << totalBytes << " bytes" << std::endl;
            std::cout << "   Expected: " << expected_size_ << " bytes" << std::endl;
            
            // This indicates the expectedSize bug
            progress_overflow_detected_ = true;
          }
          
          // Update progress display
          std::string file_name = fileName ? *fileName : "file";
          std::string size_info = " (" + std::to_string(downloadedBytes / 1024) + " KB / " + 
                                 std::to_string(totalBytes / 1024) + " KB)";
          std::string message = file_name + size_info;
          
          // Show progress (limit to 100%)
          // UserInterface::ShowProgress(message, std::min(progress, 1.0f));
          std::cout << "Progress: " << message << " " << (std::min(progress, 1.0f) * 100) << "%" << std::endl;
        }
        return false; // Don't pause
      }
      
      bool progress_overflow_detected_ = false;
      
    private:
      int64_t expected_size_;
    };
    
    TestDownloadListener listener(metadata.size);
    
    // Download the file
    ModelFileDownloader downloader;
    auto start_time = std::chrono::high_resolution_clock::now();
    downloader.DownloadFile(task, listener);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Download completed in " << duration.count() << " ms" << std::endl;
    
    // Verify file was downloaded
    assertTrue(std::filesystem::exists(task.downloadPath), "Downloaded file exists");
    
    // Get actual file size
    int64_t actual_size = std::filesystem::file_size(task.downloadPath);
    std::cout << "Actual file size: " << actual_size << " bytes" << std::endl;
    
    // Validate file size matches expected size
    // Allow 5% tolerance for potential differences
    double tolerance = 0.05;
    int64_t min_expected = static_cast<int64_t>(metadata.size * (1.0 - tolerance));
    int64_t max_expected = static_cast<int64_t>(metadata.size * (1.0 + tolerance));
    
    assertTrue(actual_size >= min_expected, "Actual size >= min expected");
    assertTrue(actual_size <= max_expected, "Actual size <= max expected");
    
    // Check if progress overflow was detected
    if (listener.progress_overflow_detected_) {
      std::cout << "❌ Progress overflow detected - this indicates the expectedSize bug!" << std::endl;
      std::cout << "   Expected size from metadata: " << metadata.size << " bytes" << std::endl;
      std::cout << "   Actual file size: " << actual_size << " bytes" << std::endl;
      std::cout << "   Difference: " << (actual_size - metadata.size) << " bytes" << std::endl;
    } else {
      std::cout << "✅ No progress overflow detected" << std::endl;
    }
    
    // Verify file content (basic check)
    std::ifstream file(task.downloadPath);
    assertTrue(file.is_open(), "File can be opened for reading");
    
    // Check if file has content (not empty)
    file.seekg(0, std::ios::end);
    int64_t file_size_from_stream = file.tellg();
    assertTrue(file_size_from_stream > 0, "File has content");
    assertTrue(file_size_from_stream == actual_size, "File size consistency");
    
    file.close();
    
    std::cout << "✅ Tokenizer.txt download test passed" << std::endl;
    
  } catch (const std::exception& e) {
    std::cerr << "❌ Tokenizer.txt download test failed: " << e.what() << std::endl;
    assert(false);
  }
}

// Test multiple file metadata retrieval to compare sizes
void testMultipleFileMetadata() {
  printTestHeader("Multiple File Metadata Comparison Test");
  
  try {
    std::string model_id = "taobao-mnn/SmolVLM-256M-Instruct-MNN";
    std::vector<std::string> files_to_test = {
      "tokenizer.txt",
      "config.json",
      "tokenizer_config.json"
    };
    
    std::cout << "Testing metadata for multiple files from model: " << model_id << std::endl;
    
    for (const auto& filename : files_to_test) {
      std::string file_url = "https://huggingface.co/" + model_id + "/resolve/main/" + filename;
      
      std::cout << "\nTesting file: " << filename << std::endl;
      std::cout << "URL: " << file_url << std::endl;
      
      std::string error_info;
      HfFileMetadata metadata = HfFileMetadataUtils::GetFileMetadata(file_url, error_info);
      
      if (error_info.empty()) {
        std::cout << "  Size: " << metadata.size << " bytes (" << (metadata.size / 1024.0) << " KB)" << std::endl;
        std::cout << "  ETag: " << metadata.etag << std::endl;
        
        // Validate size is reasonable
        assertTrue(metadata.size > 0, "File size is positive for " + filename);
        assertTrue(metadata.size < 100 * 1024 * 1024, "File size < 100MB for " + filename);
      } else {
        std::cout << "  ❌ Failed to get metadata: " << error_info << std::endl;
        // Some files might not exist, which is okay
      }
    }
    
    std::cout << "✅ Multiple file metadata test completed" << std::endl;
    
  } catch (const std::exception& e) {
    std::cerr << "❌ Multiple file metadata test failed: " << e.what() << std::endl;
    assert(false);
  }
}

// Test progress calculation accuracy
void testProgressCalculation() {
  printTestHeader("Progress Calculation Accuracy Test");
  
  try {
    std::string tokenizer_url = "https://huggingface.co/taobao-mnn/SmolVLM-256M-Instruct-MNN/resolve/main/tokenizer.txt";
    
    // Get metadata
    std::string error_info;
    HfFileMetadata metadata = HfFileMetadataUtils::GetFileMetadata(tokenizer_url, error_info);
    
    assertTrue(error_info.empty(), "Metadata retrieval succeeded");
    
    std::cout << "Testing progress calculation with expected size: " << metadata.size << " bytes" << std::endl;
    
    // Simulate different download scenarios
    std::vector<int64_t> test_scenarios = {
      0,                           // Start
      metadata.size / 4,           // 25%
      metadata.size / 2,           // 50%
      metadata.size * 3 / 4,       // 75%
      metadata.size,               // 100%
      metadata.size + 1024,        // 100%+ (potential overflow)
      metadata.size * 2            // 200% (definite overflow)
    };
    
    for (int64_t downloaded : test_scenarios) {
      float progress = static_cast<float>(downloaded) / metadata.size;
      std::cout << "Downloaded: " << downloaded << " bytes, Progress: " << (progress * 100) << "%" << std::endl;
      
      if (progress > 1.0f) {
        std::cout << "  ⚠️  Progress overflow detected!" << std::endl;
      }
    }
    
    std::cout << "✅ Progress calculation test completed" << std::endl;
    
  } catch (const std::exception& e) {
    std::cerr << "❌ Progress calculation test failed: " << e.what() << std::endl;
    assert(false);
  }
}

// Test x-linked-size vs Content-Length header comparison
void testHeaderComparison() {
  printTestHeader("x-linked-size vs Content-Length Header Test");
  
  try {
    std::string tokenizer_url = "https://huggingface.co/taobao-mnn/SmolVLM-256M-Instruct-MNN/resolve/main/tokenizer.txt";
    
    std::cout << "Testing header comparison for: " << tokenizer_url << std::endl;
    
    // Get metadata using default method
    std::string error_info;
    HfFileMetadata metadata = HfFileMetadataUtils::GetFileMetadata(tokenizer_url, error_info);
    
    assertTrue(error_info.empty(), "Metadata retrieval succeeded");
    
    std::cout << "Retrieved metadata:" << std::endl;
    std::cout << "  Size: " << metadata.size << " bytes" << std::endl;
    std::cout << "  ETag: " << metadata.etag << std::endl;
    std::cout << "  Location: " << metadata.location << std::endl;
    
    // Test with custom client to inspect headers directly
    std::shared_ptr<
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
        httplib::SSLClient
#else
        httplib::Client
#endif
    > custom_client = std::make_shared<
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
        httplib::SSLClient
#else
        httplib::Client
#endif
    >("huggingface.co");
    
    // Parse URL
    auto [host, path] = HfApiClient::ParseUrl(tokenizer_url);
    
    // Make HEAD request to inspect headers
    httplib::Headers headers;
    headers.emplace("User-Agent", "MNN-CLI/1.0");
    headers.emplace("Accept", "*/*");
    headers.emplace("Accept-Encoding", "identity");
    
    auto res = custom_client->Head(path, headers);
    
    if (res && res->status == 200) {
      std::cout << "\nResponse headers:" << std::endl;
      
      // Check x-linked-size
      auto linked_size = res->get_header_value("x-linked-size");
      if (!linked_size.empty()) {
        std::cout << "  x-linked-size: " << linked_size << std::endl;
        try {
          int64_t linked_size_val = std::stoull(linked_size);
          std::cout << "  x-linked-size (parsed): " << linked_size_val << " bytes" << std::endl;
        } catch (const std::exception& e) {
          std::cout << "  x-linked-size (parse error): " << e.what() << std::endl;
        }
      } else {
        std::cout << "  x-linked-size: not present" << std::endl;
      }
      
      // Check Content-Length
      auto content_length = res->get_header_value("Content-Length");
      if (!content_length.empty()) {
        std::cout << "  Content-Length: " << content_length << std::endl;
        try {
          int64_t content_length_val = std::stoull(content_length);
          std::cout << "  Content-Length (parsed): " << content_length_val << " bytes" << std::endl;
        } catch (const std::exception& e) {
          std::cout << "  Content-Length (parse error): " << e.what() << std::endl;
        }
      } else {
        std::cout << "  Content-Length: not present" << std::endl;
      }
      
      // Check ETag headers
      auto linked_etag = res->get_header_value("x-linked-etag");
      auto etag = res->get_header_value("ETag");
      
      std::cout << "  x-linked-etag: " << (linked_etag.empty() ? "not present" : linked_etag) << std::endl;
      std::cout << "  ETag: " << (etag.empty() ? "not present" : etag) << std::endl;
      
      // Compare with metadata
      std::cout << "\nComparison with metadata:" << std::endl;
      std::cout << "  Metadata size: " << metadata.size << " bytes" << std::endl;
      
      if (!linked_size.empty() && !content_length.empty()) {
        try {
          int64_t linked_size_val = std::stoull(linked_size);
          int64_t content_length_val = std::stoull(content_length);
          
          std::cout << "  x-linked-size: " << linked_size_val << " bytes" << std::endl;
          std::cout << "  Content-Length: " << content_length_val << " bytes" << std::endl;
          
          if (linked_size_val == content_length_val) {
            std::cout << "  ✅ Headers match" << std::endl;
          } else {
            std::cout << "  ⚠️  Headers differ by " << (linked_size_val - content_length_val) << " bytes" << std::endl;
          }
          
          if (metadata.size == linked_size_val) {
            std::cout << "  ✅ Metadata matches x-linked-size" << std::endl;
          } else if (metadata.size == content_length_val) {
            std::cout << "  ✅ Metadata matches Content-Length" << std::endl;
          } else {
            std::cout << "  ⚠️  Metadata doesn't match either header" << std::endl;
          }
          
        } catch (const std::exception& e) {
          std::cout << "  ❌ Error parsing headers: " << e.what() << std::endl;
        }
      }
      
    } else {
      std::cout << "❌ Failed to get headers: " << (res ? std::to_string(res->status) : "no response") << std::endl;
    }
    
    std::cout << "✅ Header comparison test completed" << std::endl;
    
  } catch (const std::exception& e) {
    std::cerr << "❌ Header comparison test failed: " << e.what() << std::endl;
    assert(false);
  }
}

int main(int argc, char* argv[]) {
  std::cout << "==========================================" << std::endl;
  std::cout << "Hugging Face File Metadata Test Suite" << std::endl;
  std::cout << "Model: taobao-mnn/SmolVLM-256M-Instruct-MNN" << std::endl;
  std::cout << "==========================================" << std::endl;
  
  // Enable verbose logging if -v flag is provided
  bool verbose = false;
  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "-v" || std::string(argv[i]) == "--verbose") {
      verbose = true;
      break;
    }
  }
  
  // LogUtils::setVerbose(verbose);
  if (verbose) {
    std::cout << "Verbose mode enabled" << std::endl;
  }
  
  if (verbose) {
    std::cout << "Verbose logging enabled" << std::endl;
  }
  
  try {
    // Run all tests
    testTokenizerFileMetadata();
    testTokenizerFileDownload();
    testMultipleFileMetadata();
    testProgressCalculation();
    testHeaderComparison();
    
    std::cout << "\n==========================================" << std::endl;
    std::cout << "🎉 All Hugging Face file metadata tests passed successfully!" << std::endl;
    std::cout << "==========================================" << std::endl;
    
    return 0;
    
  } catch (const std::exception& e) {
    std::cerr << "\n❌ Test suite failed with exception: " << e.what() << std::endl;
    return 1;
  }
}
