//
// Simple compilation test for the multi-platform downloader
// This file tests that all includes and declarations are correct
//

#include "model_file_downloader.hpp"
#include <iostream>

int main() {
    // Test basic instantiation
    mnncli::RemoteModelDownloader downloader;
    
    // Test platform detection
    std::string platform1 = downloader.DetectPlatform("HuggingFace/bert-base-uncased");
    std::string platform2 = downloader.DetectPlatform("ModelScope/damo/model");
    std::string platform3 = downloader.DetectPlatform("Modelers/owner/repo");
    
    std::cout << "Platform detection test:" << std::endl;
    std::cout << "HuggingFace: " << platform1 << std::endl;
    std::cout << "ModelScope: " << platform2 << std::endl;
    std::cout << "Modelers: " << platform3 << std::endl;
    
    // Test unified structures
    mnncli::UnifiedFileMetadata file_meta;
    file_meta.location = "https://example.com/file";
    file_meta.etag = "abc123";
    file_meta.size = 1024;
    file_meta.platform = "hf";
    
    mnncli::UnifiedRepoInfo repo_info;
    repo_info.model_id = "test-model";
    repo_info.platform = "hf";
    repo_info.files.push_back(file_meta);
    
    std::cout << "Compilation test successful!" << std::endl;
    return 0;
}
