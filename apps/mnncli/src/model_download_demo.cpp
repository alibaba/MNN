//
// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#include "model_download_manager.hpp"
#include <iostream>
#include <thread>
#include <chrono>

namespace mnncli {

// Simple download listener implementation
class DemoDownloadListener : public DownloadListener {
public:
    void onDownloadStart(const std::string& model_id) override {
        std::cout << "🚀 Download started: " << model_id << std::endl;
    }
    
    void onDownloadProgress(const std::string& model_id, const DownloadProgress& progress) override {
        std::cout << "📊 Progress for " << model_id << ": " 
                  << std::fixed << std::setprecision(1) << (progress.progress * 100.0) << "% "
                  << "(" << progress.saved_size << "/" << progress.total_size << " bytes) "
                  << "Stage: " << progress.stage;
        
        if (!progress.current_file.empty()) {
            std::cout << " File: " << progress.current_file;
        }
        std::cout << std::endl;
    }
    
    void onDownloadFinished(const std::string& model_id, const std::string& path) override {
        std::cout << "✅ Download finished: " << model_id << " -> " << path << std::endl;
    }
    
    void onDownloadFailed(const std::string& model_id, const std::string& error) override {
        std::cout << "❌ Download failed: " << model_id << " - " << error << std::endl;
    }
    
    void onDownloadPaused(const std::string& model_id) override {
        std::cout << "⏸️  Download paused: " << model_id << std::endl;
    }
    
    void onDownloadTaskAdded() override {
        std::cout << "➕ Download task added" << std::endl;
    }
    
    void onDownloadTaskRemoved() override {
        std::cout << "➖ Download task removed" << std::endl;
    }
    
    void onRepoInfo(const std::string& model_id, int64_t last_modified, int64_t repo_size) override {
        std::cout << "ℹ️  Repo info for " << model_id << ": "
                  << "Size: " << repo_size << " bytes, "
                  << "Modified: " << last_modified << std::endl;
    }
};

} // namespace mnncli

int main() {
    std::cout << "🤖 MNN Model Download Manager Demo" << std::endl;
    std::cout << "==================================" << std::endl;
    
    // Set cache path
    std::string cache_path = "./models";
    
    // Create download manager
    auto& download_manager = mnncli::ModelDownloadManager::getInstance(cache_path);
    download_manager.setVerbose(true);
    
    // Create and add listener
    auto listener = std::make_unique<mnncli::DemoDownloadListener>();
    download_manager.addListener(listener.get());
    
    std::cout << "\n📁 Cache path: " << cache_path << std::endl;
    std::cout << "\n🔧 Available commands:" << std::endl;
    std::cout << "1. download <model_id> - Start download" << std::endl;
    std::cout << "2. pause <model_id> - Pause download" << std::endl;
    std::cout << "3. resume <model_id> - Resume download" << std::endl;
    std::cout << "4. status <model_id> - Show download status" << std::endl;
    std::cout << "5. list - Show active downloads" << std::endl;
    std::cout << "6. quit - Exit program" << std::endl;
    
    std::cout << "\n💡 Example model IDs:" << std::endl;
    std::cout << "  - HuggingFace:taobao-mnn/chatglm3-6b" << std::endl;
    std::cout << "  - ModelScope:damo/nlp_gpt3_text-generation_1.3B" << std::endl;
    std::cout << "  - Modelers:alibaba/Qwen-VL-Chat" << std::endl;
    
    std::string command;
    while (true) {
        std::cout << "\n> ";
        std::getline(std::cin, command);
        
        if (command == "quit" || command == "exit") {
            break;
        }
        
        if (command.substr(0, 8) == "download") {
            if (command.length() > 9) {
                std::string model_id = command.substr(9);
                std::cout << "Starting download for: " << model_id << std::endl;
                download_manager.startDownload(model_id);
            } else {
                std::cout << "Usage: download <model_id>" << std::endl;
            }
        }
        else if (command.substr(0, 4) == "pause") {
            if (command.length() > 5) {
                std::string model_id = command.substr(5);
                std::cout << "Pausing download for: " << model_id << std::endl;
                download_manager.pauseDownload(model_id);
            } else {
                std::cout << "Usage: pause <model_id>" << std::endl;
            }
        }
        else if (command.substr(0, 6) == "resume") {
            if (command.length() > 7) {
                std::string model_id = command.substr(7);
                std::cout << "Resuming download for: " << model_id << std::endl;
                download_manager.resumeDownload(model_id);
            } else {
                std::cout << "Usage: resume <model_id>" << std::endl;
            }
        }
        else if (command.substr(0, 6) == "status") {
            if (command.length() > 7) {
                std::string model_id = command.substr(7);
                auto info = download_manager.getDownloadInfo(model_id);
                std::cout << "Status for " << model_id << ":" << std::endl;
                std::cout << "  State: " << static_cast<int>(info.state) << std::endl;
                std::cout << "  Progress: " << (info.progress * 100.0) << "%" << std::endl;
                std::cout << "  Downloaded: " << info.saved_size << " / " << info.total_size << " bytes" << std::endl;
            } else {
                std::cout << "Usage: status <model_id>" << std::endl;
            }
        }
        else if (command == "list") {
            auto active_downloads = download_manager.getActiveDownloads();
            if (active_downloads.empty()) {
                std::cout << "No active downloads" << std::endl;
            } else {
                std::cout << "Active downloads:" << std::endl;
                for (const auto& model_id : active_downloads) {
                    std::cout << "  - " << model_id << std::endl;
                }
            }
        }
        else if (command == "help") {
            std::cout << "🔧 Available commands:" << std::endl;
            std::cout << "1. download <model_id> - Start download" << std::endl;
            std::cout << "2. pause <model_id> - Pause download" << std::endl;
            std::cout << "3. resume <model_id> - Resume download" << std::endl;
            std::cout << "4. status <model_id> - Show download status" << std::endl;
            std::cout << "5. list - Show active downloads" << std::endl;
            std::cout << "6. quit - Exit program" << std::endl;
        }
        else if (!command.empty()) {
            std::cout << "Unknown command. Type 'help' for available commands." << std::endl;
        }
        
        // Small delay to allow download events to be processed
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    std::cout << "👋 Goodbye!" << std::endl;
    return 0;
}
