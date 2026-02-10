//
// Created by AI Assistant on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#pragma once

#include "json.hpp"
#include "model_download_manager.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <optional>

namespace mnncli {

// Forward declarations


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
    static ModelRepository& GetInstance(const std::string& cache_root_path = "");
    
    // Get model market data from various sources
    std::optional<ModelMarketData> GetModelMarketData();
    
    // Get models by type
    std::vector<ModelMarketItem> GetModels();
    std::vector<ModelMarketItem> GetTtsModels();
    std::vector<ModelMarketItem> GetAsrModels();
    
    // Get model by name and create model ID for download
    std::optional<std::string> GetModelIdForDownload(const std::string& modelName, const std::string& downloadProvider);
    
    // Get model type (ASR, TTS, or LLM)
    std::string GetModelType(const std::string& modelId);
    
    // Search models by keyword (LLM models only, filtered by current source)
    std::vector<ModelMarketItem> SearchModels(const std::string& keyword);
    
    // Set download provider
    void SetDownloadProvider(const std::string& provider) { current_download_provider_ = provider; }
    std::string GetDownloadProvider() const { return current_download_provider_; }

private:
    // Load data from different sources
    std::optional<ModelMarketData> LoadFromAssets();
    std::optional<ModelMarketItem> FindModelByName(const std::string& modelName);
    
    // Process models with current download provider
    std::vector<ModelMarketItem> ProcessModels(const std::vector<ModelMarketItem>& models);
    
    // Version comparison
    bool IsVersionLower(const std::string& version1, const std::string& version2);
    
    // Create model ID from source and repo path
    std::string CreateModelId(const std::string& source, const std::string& repoPath);

private:
    std::string cache_root_path_;
    std::string current_download_provider_;
    
    // Cached data
    std::optional<ModelMarketData> cached_model_market_data_;
    bool is_network_request_attempted_;
    
    // Constants
    static constexpr const char* kTag = "ModelRepository";
    static constexpr const char* kAssetsFileName = "model_market.json";
    static constexpr const char* kDefaultDownloadProvider = mnn::downloader::ModelSources::SOURCE_HUGGING_FACE;
    
    // Default download providers (in order of preference)
    static const std::vector<std::string> kDefaultProviders;
};

} // namespace mnncli
