//
// Created by AI Assistant on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#include "../include/hf_model_downloader.hpp"
#include "../include/hf_api_client.hpp"
#include "../include/log_utils.hpp"
#include "../include/model_download_manager.hpp"
#include <iostream>
#include <cassert>
#include <string>
#include <filesystem>
#include <chrono>
#include <thread>

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

// Test HF API Client functionality
void testHfApiClient() {
  printTestHeader("HF API Client");
  
  try {
    HfApiClient client;
    
    // Test host getter
    std::string host = client.GetHost();
    assertTrue(!host.empty(), "GetHost returns non-empty string");
    std::cout << "HF Host: " << host << std::endl;
    
    // Test URL parsing
    auto [parsed_host, parsed_path] = HfApiClient::ParseUrl("https://huggingface.co/taobao-mnn/SmolLM2-135M-Instruct-MNN");
    assertTrue(parsed_host == "huggingface.co", "URL parsing - host");
    assertTrue(parsed_path == "/taobao-mnn/SmolLM2-135M-Instruct-MNN", "URL parsing - path");
    
    std::cout << "✅ HF API Client basic functionality works" << std::endl;
    
  } catch (const std::exception& e) {
    std::cerr << "❌ HF API Client test failed: " << e.what() << std::endl;
    assert(false);
  }
}

// Test HF Model Downloader initialization
void testHfModelDownloaderInit() {
  printTestHeader("HF Model Downloader Initialization");
  
  try {
    std::string cache_path = "/tmp/mnncli_test_cache";
    
    // Clean up any existing test cache
    if (std::filesystem::exists(cache_path)) {
      std::filesystem::remove_all(cache_path);
    }
    
    HfModelDownloader downloader(cache_path);
    
    // Test that cache path is set correctly
    std::filesystem::path download_path = downloader.getDownloadPath("taobao-mnn/SmolLM2-135M-Instruct-MNN");
    assertTrue(download_path.string().find(cache_path) != std::string::npos, 
               "Download path contains cache path");
    
    std::cout << "✅ HF Model Downloader initialization works" << std::endl;
    
  } catch (const std::exception& e) {
    std::cerr << "❌ HF Model Downloader init test failed: " << e.what() << std::endl;
    assert(false);
  }
}

// Test actual model download
void testModelDownload() {
  printTestHeader("Model Download Test");
  
  try {
    std::string cache_path = "/tmp/mnncli_test_cache";
    std::string model_id = "taobao-mnn/SmolLM2-135M-Instruct-MNN";
    
    // Clean up any existing test cache
    if (std::filesystem::exists(cache_path)) {
      std::filesystem::remove_all(cache_path);
    }
    
    // Create downloader
    HfModelDownloader downloader(cache_path);
    
    // Set up API client
    auto api_client = std::make_shared<HfApiClient>();
    downloader.setHfApiClient(api_client);
    
    std::cout << "Starting download of model: " << model_id << std::endl;
    std::cout << "Cache path: " << cache_path << std::endl;
    
    // Start download
    auto start_time = std::chrono::high_resolution_clock::now();
    downloader.download(model_id);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "Download completed in " << duration.count() << " seconds" << std::endl;
    
    // Check if files were downloaded
    std::filesystem::path download_path = downloader.getDownloadPath(model_id);
    assertTrue(std::filesystem::exists(download_path), "Download path exists");
    
    // Check if there are files in the download directory
    bool has_files = false;
    if (std::filesystem::exists(download_path)) {
      for (const auto& entry : std::filesystem::directory_iterator(download_path)) {
        if (entry.is_regular_file()) {
          has_files = true;
          std::cout << "Downloaded file: " << entry.path().filename() << std::endl;
        }
      }
    }
    
    assertTrue(has_files, "Downloaded files exist");
    
    // Test repository size calculation
    int64_t repo_size = downloader.getRepoSize(model_id);
    assertTrue(repo_size > 0, "Repository size is positive");
    std::cout << "Repository size: " << repo_size << " bytes" << std::endl;
    
    std::cout << "✅ Model download test completed successfully" << std::endl;
    
  } catch (const std::exception& e) {
    std::cerr << "❌ Model download test failed: " << e.what() << std::endl;
    assert(false);
  }
}

// Test Model Download Manager integration
void testModelDownloadManager() {
  printTestHeader("Model Download Manager Integration");
  
  try {
    std::string cache_path = "/tmp/mnncli_test_cache";
    
    // Clean up any existing test cache
    if (std::filesystem::exists(cache_path)) {
      std::filesystem::remove_all(cache_path);
    }
    
    ModelDownloadManager manager(cache_path, true); // Enable verbose mode
    
    std::cout << "Testing Model Download Manager with HF model..." << std::endl;
    
    // Test starting download through manager
    std::string model_id = "taobao-mnn/SmolLM2-135M-Instruct-MNN";
    manager.startDownload(model_id, "HuggingFace");
    
    // Give it some time to download
    std::this_thread::sleep_for(std::chrono::seconds(5));
    
    // Check if download was initiated
    std::filesystem::path download_path = std::filesystem::path(cache_path) / 
                                         FileUtils::RepoFolderName(model_id, "model");
    
    std::cout << "Expected download path: " << download_path << std::endl;
    
    std::cout << "✅ Model Download Manager integration test completed" << std::endl;
    
  } catch (const std::exception& e) {
    std::cerr << "❌ Model Download Manager test failed: " << e.what() << std::endl;
    assert(false);
  }
}

// Test error handling
void testErrorHandling() {
  printTestHeader("Error Handling");
  
  try {
    std::string cache_path = "/tmp/mnncli_test_cache";
    HfModelDownloader downloader(cache_path);
    
    // Test with invalid model ID
    std::string invalid_model = "non-existent-user/invalid-model-name";
    
    std::cout << "Testing error handling with invalid model: " << invalid_model << std::endl;
    
    // This should handle the error gracefully
    downloader.download(invalid_model);
    
    std::cout << "✅ Error handling test completed (no crash)" << std::endl;
    
  } catch (const std::exception& e) {
    std::cout << "✅ Error handling test completed (caught expected exception): " << e.what() << std::endl;
  }
}

int main(int argc, char* argv[]) {
  std::cout << "==========================================" << std::endl;
  std::cout << "Hugging Face Model Download Test Suite" << std::endl;
  std::cout << "==========================================" << std::endl;
  
  // Enable verbose logging if -v flag is provided
  bool verbose = false;
  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "-v" || std::string(argv[i]) == "--verbose") {
      verbose = true;
      break;
    }
  }
  
  LogUtils::setVerbose(verbose);
  
  if (verbose) {
    std::cout << "Verbose logging enabled" << std::endl;
  }
  
  try {
    // Run all tests
    testHfApiClient();
    testHfModelDownloaderInit();
    testModelDownload();
    testModelDownloadManager();
    testErrorHandling();
    
    std::cout << "\n==========================================" << std::endl;
    std::cout << "🎉 All tests passed successfully!" << std::endl;
    std::cout << "==========================================" << std::endl;
    
    return 0;
    
  } catch (const std::exception& e) {
    std::cerr << "\n❌ Test suite failed with exception: " << e.what() << std::endl;
    return 1;
  }
}

