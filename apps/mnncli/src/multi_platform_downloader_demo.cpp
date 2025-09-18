//
// Created by ruoyi.sjd on 2024/12/18.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#include "model_file_downloader.hpp"
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

int main() {
    try {
        // Create downloader instance
        mnncli::RemoteModelDownloader downloader(3, 2);
        
        // Set storage folder
        fs::path storage_folder = fs::current_path() / "models";
        fs::create_directories(storage_folder);
        
        std::cout << "Multi-Platform Model Downloader Demo" << std::endl;
        std::cout << "===================================" << std::endl;
        
        // Example 1: Download HuggingFace model
        std::cout << "\n1. Downloading HuggingFace model..." << std::endl;
        std::string hf_model_id = "HuggingFace/bert-base-uncased";
        std::string hf_error;
        
        auto hf_result = downloader.DownloadHfModel(hf_model_id, storage_folder, hf_error);
        if (hf_error.empty()) {
            std::cout << "✓ HuggingFace model downloaded successfully to: " << hf_result << std::endl;
        } else {
            std::cout << "✗ HuggingFace model download failed: " << hf_error << std::endl;
        }
        
        // Example 2: Download ModelScope model
        std::cout << "\n2. Downloading ModelScope model..." << std::endl;
        std::string ms_model_id = "ModelScope/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch";
        std::string ms_error;
        
        auto ms_result = downloader.DownloadMsModel(ms_model_id, storage_folder, ms_error);
        if (ms_error.empty()) {
            std::cout << "✓ ModelScope model downloaded successfully to: " << ms_result << std::endl;
        } else {
            std::cout << "✗ ModelScope model download failed: " << ms_error << std::endl;
        }
        
        // Example 3: Download Modelers model
        std::cout << "\n3. Downloading Modelers model..." << std::endl;
        std::string ml_model_id = "Modelers/example/llm-model";
        std::string ml_error;
        
        auto ml_result = downloader.DownloadMlModel(ml_model_id, storage_folder, ml_error);
        if (ml_error.empty()) {
            std::cout << "✓ Modelers model downloaded successfully to: " << ml_result << std::endl;
        } else {
            std::cout << "✗ Modelers model download failed: " << ml_error << std::endl;
        }
        
        // Example 4: Auto-detect platform
        std::cout << "\n4. Auto-detecting platform..." << std::endl;
        std::string auto_model_id = "bert-base-uncased"; // Will default to HuggingFace
        std::string auto_error;
        
        auto auto_result = downloader.DownloadModel(auto_model_id, storage_folder, auto_error);
        if (auto_error.empty()) {
            std::cout << "✓ Auto-detected model downloaded successfully to: " << auto_result << std::endl;
        } else {
            std::cout << "✗ Auto-detected model download failed: " << auto_error << std::endl;
        }
        
        // Example 5: Get repo information
        std::cout << "\n5. Getting repo information..." << std::endl;
        std::string info_error;
        auto repo_info = downloader.GetRepoInfo(hf_model_id, true, info_error);
        if (info_error.empty()) {
            std::cout << "✓ Repo info retrieved successfully:" << std::endl;
            std::cout << "  - Platform: " << repo_info.platform << std::endl;
            std::cout << "  - Model ID: " << repo_info.model_id << std::endl;
            std::cout << "  - SHA: " << repo_info.sha << std::endl;
            std::cout << "  - Total size: " << repo_info.total_size << " bytes" << std::endl;
            std::cout << "  - Files count: " << repo_info.files.size() << std::endl;
        } else {
            std::cout << "✗ Failed to get repo info: " << info_error << std::endl;
        }
        
        // Example 6: Get repo size
        std::cout << "\n6. Getting repo size..." << std::endl;
        std::string size_error;
        auto repo_size = downloader.GetRepoSize(hf_model_id, size_error);
        if (size_error.empty()) {
            std::cout << "✓ Repo size: " << repo_size << " bytes (" 
                      << (repo_size / 1024.0 / 1024.0) << " MB)" << std::endl;
        } else {
            std::cout << "✗ Failed to get repo size: " << size_error << std::endl;
        }
        
        // Example 7: Check for updates
        std::cout << "\n7. Checking for updates..." << std::endl;
        std::string update_error;
        bool has_update = downloader.CheckUpdate(hf_model_id, update_error);
        if (update_error.empty()) {
            std::cout << "✓ Update check completed. Has update: " << (has_update ? "Yes" : "No") << std::endl;
        } else {
            std::cout << "✗ Update check failed: " << update_error << std::endl;
        }
        
        std::cout << "\nDemo completed!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
