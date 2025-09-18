//
// Created by AI Assistant on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#pragma once

#include "jsonhpp/json.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <optional>

namespace mnncli {

// Forward declarations
class ModelDownloadManager;

// Model market item structure
struct ModelMarketItem {
    std::string modelName;
    std::vector<std::string> tags;
    std::vector<std::string> categories;
    std::unordered_map<std::string, std::string> sources;
    double size_gb;
    std::string vendor;
    int64_t file_size;
    
    // Runtime fields (set during processing)
    std::string currentSource;
    std::string currentRepoPath;
    std::string modelId;
};

// Model market data structure
struct ModelMarketData {
    std::string version;
    std::unordered_map<std::string, std::string> tagTranslations;
    std::vector<std::string> quickFilterTags;
    std::vector<std::string> vendorOrder;
    std::vector<ModelMarketItem> models;
    std::vector<ModelMarketItem> ttsModels;
    std::vector<ModelMarketItem> asrModels;
};

// Model repository class for managing model market data
class ModelRepository {
public:
    explicit ModelRepository(const std::string& cache_root_path = "");
    ~ModelRepository() = default;
    
    // Singleton access
    static ModelRepository& getInstance(const std::string& cache_root_path = "");
    
    // Get model market data from various sources
    std::optional<ModelMarketData> getModelMarketData();
    
    // Get models by type
    std::vector<ModelMarketItem> getModels();
    std::vector<ModelMarketItem> getTtsModels();
    std::vector<ModelMarketItem> getAsrModels();
    
    // Get model by name and create model ID for download
    std::optional<std::string> getModelIdForDownload(const std::string& modelName, const std::string& downloadProvider);
    
    // Get model type (ASR, TTS, or LLM)
    std::string getModelType(const std::string& modelId);
    
    // Search models by keyword (LLM models only, filtered by current source)
    std::vector<ModelMarketItem> searchModels(const std::string& keyword);
    
    // Set download provider
    void setDownloadProvider(const std::string& provider) { current_download_provider_ = provider; }
    std::string getDownloadProvider() const { return current_download_provider_; }

private:
    // Load data from different sources
    std::optional<ModelMarketData> loadFromAssets();
    std::optional<ModelMarketItem> findModelByName(const std::string& modelName);
    
    // Process models with current download provider
    std::vector<ModelMarketItem> processModels(const std::vector<ModelMarketItem>& models);
    
    // Version comparison
    bool isVersionLower(const std::string& version1, const std::string& version2);
    
    // Create model ID from source and repo path
    std::string createModelId(const std::string& source, const std::string& repoPath);

private:
    std::string cache_root_path_;
    std::string current_download_provider_;
    
    // Cached data
    std::optional<ModelMarketData> cached_model_market_data_;
    bool is_network_request_attempted_;
    
    // Constants
    static constexpr const char* TAG = "ModelRepository";
    static constexpr const char* ASSETS_FILE_NAME = "model_market.json";
    static constexpr const char* DEFAULT_DOWNLOAD_PROVIDER = "HuggingFace";
    
    // Default download providers (in order of preference)
    static const std::vector<std::string> DEFAULT_PROVIDERS;
};

} // namespace mnncli
