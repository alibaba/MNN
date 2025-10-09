//
// Created by AI Assistant on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#include <iostream>
#include <string>
#include <cassert>
#include <httplib.h>

// Simple test to check Hugging Face file metadata
void testTokenizerFileMetadata() {
  std::cout << "==========================================" << std::endl;
  std::cout << "Testing Hugging Face File Metadata" << std::endl;
  std::cout << "Model: taobao-mnn/SmolVLM-256M-Instruct-MNN" << std::endl;
  std::cout << "==========================================" << std::endl;
  
  try {
    // Test URL for tokenizer.txt from the specific model
    std::string tokenizer_url = "https://huggingface.co/taobao-mnn/SmolVLM-256M-Instruct-MNN/resolve/main/tokenizer.txt";
    
    std::cout << "Testing metadata retrieval for: " << tokenizer_url << std::endl;
    
    // Create HTTP client
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    httplib::SSLClient client("huggingface.co");
#else
    httplib::Client client("huggingface.co");
#endif
    
    // Set headers
    httplib::Headers headers;
    headers.emplace("User-Agent", "MNN-CLI/1.0");
    headers.emplace("Accept", "*/*");
    headers.emplace("Accept-Encoding", "identity");
    
    // Make HEAD request to get metadata
    auto res = client.Head("/taobao-mnn/SmolVLM-256M-Instruct-MNN/resolve/main/tokenizer.txt", headers);
    
    if (!res) {
      std::cerr << "❌ Failed to connect to server" << std::endl;
      return;
    }
    
    if (res->status != 200 && res->status < 300 && res->status >= 300) {
      std::cerr << "❌ HTTP error: " << res->status << std::endl;
      return;
    }
    
    // Handle redirects
    if (res->status >= 300 && res->status < 400) {
      std::cout << "🔄 Redirect detected (status: " << res->status << ")" << std::endl;
      auto location = res->get_header_value("Location");
      if (!location.empty()) {
        std::cout << "  Redirect to: " << location << std::endl;
        
        // Parse redirect URL
        size_t scheme_end = location.find("://");
        if (scheme_end != std::string::npos) {
          std::string scheme = location.substr(0, scheme_end);
          size_t host_start = scheme_end + 3;
          size_t path_start = location.find('/', host_start);
          
          std::string host = location.substr(host_start, path_start - host_start);
          std::string path = location.substr(path_start);
          
          std::cout << "  Host: " << host << std::endl;
          std::cout << "  Path: " << path << std::endl;
          
          // Create new client for redirect
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
          httplib::SSLClient redirect_client(host);
#else
          httplib::Client redirect_client(host);
#endif
          
          auto redirect_res = redirect_client.Head(path, headers);
          
          if (!redirect_res) {
            std::cerr << "❌ Failed to follow redirect" << std::endl;
            return;
          }
          
          if (redirect_res->status != 200) {
            std::cerr << "❌ Redirect failed with status: " << redirect_res->status << std::endl;
            return;
          }
          
          std::cout << "✅ Successfully followed redirect" << std::endl;
          // Use redirect response for header analysis
          res = std::move(redirect_res);
        }
      }
    }
    
    std::cout << "✅ Successfully connected to Hugging Face" << std::endl;
    
    // Print all response headers
    std::cout << "\nResponse headers:" << std::endl;
    for (const auto& header : res->headers) {
      std::cout << "  " << header.first << ": " << header.second << std::endl;
    }
    
    // Check file size headers
    auto linked_size = res->get_header_value("x-linked-size");
    auto content_length = res->get_header_value("Content-Length");
    
    std::cout << "\nFile size analysis:" << std::endl;
    
    if (!linked_size.empty()) {
      std::cout << "  x-linked-size: " << linked_size << std::endl;
      try {
        int64_t linked_size_val = std::stoull(linked_size);
        std::cout << "  x-linked-size (parsed): " << linked_size_val << " bytes (" << (linked_size_val / 1024.0) << " KB)" << std::endl;
        
        // Validate size is reasonable for a tokenizer file
        if (linked_size_val >= 1024 && linked_size_val <= 10 * 1024 * 1024) {
          std::cout << "  ✅ Size is reasonable for tokenizer.txt" << std::endl;
        } else {
          std::cout << "  ⚠️  Size seems unusual for tokenizer.txt" << std::endl;
        }
      } catch (const std::exception& e) {
        std::cout << "  ❌ Failed to parse x-linked-size: " << e.what() << std::endl;
      }
    } else {
      std::cout << "  x-linked-size: not present" << std::endl;
    }
    
    if (!content_length.empty()) {
      std::cout << "  Content-Length: " << content_length << std::endl;
      try {
        int64_t content_length_val = std::stoull(content_length);
        std::cout << "  Content-Length (parsed): " << content_length_val << " bytes (" << (content_length_val / 1024.0) << " KB)" << std::endl;
        
        // Validate size is reasonable for a tokenizer file
        if (content_length_val >= 1024 && content_length_val <= 10 * 1024 * 1024) {
          std::cout << "  ✅ Size is reasonable for tokenizer.txt" << std::endl;
        } else {
          std::cout << "  ⚠️  Size seems unusual for tokenizer.txt" << std::endl;
        }
      } catch (const std::exception& e) {
        std::cout << "  ❌ Failed to parse Content-Length: " << e.what() << std::endl;
      }
    } else {
      std::cout << "  Content-Length: not present" << std::endl;
    }
    
    // Compare headers if both are present
    if (!linked_size.empty() && !content_length.empty()) {
      try {
        int64_t linked_size_val = std::stoull(linked_size);
        int64_t content_length_val = std::stoull(content_length);
        
        std::cout << "\nHeader comparison:" << std::endl;
        std::cout << "  x-linked-size: " << linked_size_val << " bytes" << std::endl;
        std::cout << "  Content-Length: " << content_length_val << " bytes" << std::endl;
        
        if (linked_size_val == content_length_val) {
          std::cout << "  ✅ Headers match perfectly" << std::endl;
        } else {
          int64_t difference = linked_size_val - content_length_val;
          std::cout << "  ⚠️  Headers differ by " << difference << " bytes" << std::endl;
          
          if (abs(difference) > 1024) {
            std::cout << "  ❌ Significant difference detected - this could cause progress overflow!" << std::endl;
          } else {
            std::cout << "  ✅ Difference is minor" << std::endl;
          }
        }
      } catch (const std::exception& e) {
        std::cout << "  ❌ Error comparing headers: " << e.what() << std::endl;
      }
    }
    
    // Check ETag headers
    auto linked_etag = res->get_header_value("x-linked-etag");
    auto etag = res->get_header_value("ETag");
    
    std::cout << "\nETag analysis:" << std::endl;
    std::cout << "  x-linked-etag: " << (linked_etag.empty() ? "not present" : linked_etag) << std::endl;
    std::cout << "  ETag: " << (etag.empty() ? "not present" : etag) << std::endl;
    
    // Check commit hash
    auto commit_hash = res->get_header_value("x-repo-commit");
    if (!commit_hash.empty()) {
      std::cout << "  Commit hash: " << commit_hash << std::endl;
    }
    
    std::cout << "\n==========================================" << std::endl;
    std::cout << "🎉 Metadata test completed successfully!" << std::endl;
    std::cout << "==========================================" << std::endl;
    
  } catch (const std::exception& e) {
    std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
  }
}

int main(int argc, char* argv[]) {
  bool verbose = false;
  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "-v" || std::string(argv[i]) == "--verbose") {
      verbose = true;
      break;
    }
  }
  
  if (verbose) {
    std::cout << "Verbose mode enabled" << std::endl;
  }
  
  testTokenizerFileMetadata();
  
  return 0;
}
