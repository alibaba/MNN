//
// Created by AI Assistant on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#include "model_repository.hpp"
#include <iostream>
#include <iomanip>

using namespace mnncli;

void printModelInfo(const ModelMarketItem& model) {
    std::cout << std::setw(30) << std::left << model.modelName
              << std::setw(15) << std::left << model.vendor
              << std::setw(8) << std::left << model.size_gb << "GB"
              << std::setw(20) << std::left << model.currentSource
              << std::setw(50) << std::left << model.modelId << "\n";
}

void printHeader() {
    std::cout << std::string(120, '=') << "\n";
    std::cout << std::setw(30) << std::left << "Model Name"
              << std::setw(15) << std::left << "Vendor"
              << std::setw(8) << std::left << "Size"
              << std::setw(20) << std::left << "Source"
              << std::setw(50) << std::left << "Model ID" << "\n";
    std::cout << std::string(120, '=') << "\n";
}

int main() {
    std::cout << "ModelRepository Example Program\n";
    std::cout << "==============================\n\n";
    
    try {
        // Create ModelRepository instance
        auto& repo = ModelRepository::getInstance("./");
        repo.setVerbose(true);
        
        // Load model market data
        std::cout << "Loading model market data...\n";
        auto marketData = repo.getModelMarketData();
        
        if (!marketData) {
            std::cerr << "Failed to load model market data!\n";
            std::cerr << "Make sure model_market.json is available in one of these locations:\n";
            std::cerr << "  - ./model_market.json\n";
            std::cerr << "  - ./assets/model_market.json\n";
            std::cerr << "  - ../assets/model_market.json\n";
            return 1;
        }
        
        std::cout << "✓ Successfully loaded model market data (version: " << marketData->version << ")\n\n";
        
        // Test different download providers
        std::vector<std::string> providers = {"HuggingFace", "ModelScope", "Modelers"};
        
        for (const auto& provider : providers) {
            std::cout << "Testing with provider: " << provider << "\n";
            std::cout << std::string(50, '-') << "\n";
            
            repo.setDownloadProvider(provider);
            
            // Get models with current provider
            auto models = repo.getModels();
            if (!models.empty()) {
                printHeader();
                
                // Show first 5 models
                int count = 0;
                for (const auto& model : models) {
                    if (count++ >= 5) break;
                    printModelInfo(model);
                }
                
                if (models.size() > 5) {
                    std::cout << "... and " << (models.size() - 5) << " more models\n";
                }
            } else {
                std::cout << "No models available with provider: " << provider << "\n";
            }
            
            std::cout << "\n";
        }
        
        // Test model ID creation for specific models
        std::cout << "Testing Model ID Creation\n";
        std::cout << std::string(50, '=') << "\n";
        
        std::vector<std::string> testModels = {
            "gpt-oss-20b-MNN",
            "Qwen3-4B-MNN",
            "DeepSeek-R1-7B-Qwen-MNN"
        };
        
        for (const auto& modelName : testModels) {
            std::cout << "Model: " << modelName << "\n";
            
            for (const auto& provider : providers) {
                auto modelId = repo.getModelIdForDownload(modelName, provider);
                if (modelId) {
                    std::string modelType = repo.getModelType(*modelId);
                    std::cout << "  " << std::setw(15) << std::left << provider 
                              << " -> " << *modelId << " (" << modelType << ")\n";
                } else {
                    std::cout << "  " << std::setw(15) << std::left << provider 
                              << " -> Not available\n";
                }
            }
            std::cout << "\n";
        }
        
        // Test TTS and ASR models
        std::cout << "Specialized Models\n";
        std::cout << std::string(50, '=') << "\n";
        
        auto ttsModels = repo.getTtsModels();
        std::cout << "TTS Models (" << ttsModels.size() << "):\n";
        if (!ttsModels.empty()) {
            printHeader();
            for (const auto& model : ttsModels) {
                printModelInfo(model);
            }
        }
        std::cout << "\n";
        
        auto asrModels = repo.getAsrModels();
        std::cout << "ASR Models (" << asrModels.size() << "):\n";
        if (!asrModels.empty()) {
            printHeader();
            for (const auto& model : asrModels) {
                printModelInfo(model);
            }
        }
        
        std::cout << "\n✓ Example completed successfully!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
