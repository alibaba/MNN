//
// Created by AI Assistant on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#include "model_repository.hpp"
#include "file_utils.hpp"
#include "log_utils.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <filesystem>
#include <cctype>

namespace mnncli {

// Helper function for case-insensitive string comparison
bool caseInsensitiveEquals(const std::string& str1, const std::string& str2) {
    if (str1.length() != str2.length()) {
        return false;
    }
    
    return std::equal(str1.begin(), str1.end(), str2.begin(),
                     [](char a, char b) {
                         return std::tolower(static_cast<unsigned char>(a)) == 
                                std::tolower(static_cast<unsigned char>(b));
                     });
}

// Static member initialization
const std::vector<std::string> ModelRepository::DEFAULT_PROVIDERS = {
    "HuggingFace", "ModelScope", "Modelers"
};

ModelRepository::ModelRepository(const std::string& cache_root_path)
    : cache_root_path_(cache_root_path), is_network_request_attempted_(false) {
    
    // Set default download provider
    current_download_provider_ = DEFAULT_DOWNLOAD_PROVIDER;
}

ModelRepository& ModelRepository::getInstance(const std::string& cache_root_path) {
    static ModelRepository instance(cache_root_path);
    return instance;
}

std::optional<ModelMarketData> ModelRepository::getModelMarketData() {
    // For now, only load from assets (similar to Kotlin version when network is disabled)
    LOG_DEBUG_TAG("Loading model market data from assets", TAG);
    
    auto assetsData = loadFromAssets();
    if (assetsData) {
        cached_model_market_data_ = assetsData;
        LOG_DEBUG_TAG("Successfully loaded data from assets", TAG);
    }
    
    return cached_model_market_data_;
}

std::vector<ModelMarketItem> ModelRepository::getModels() {
    try {
        auto data = getModelMarketData();
        if (data && !data->models.empty()) {
            return processModels(data->models);
        }
    } catch (const std::exception& e) {
        LOG_DEBUG_TAG("Failed to get models: " + std::string(e.what()), TAG);
    }
    return {};
}

std::vector<ModelMarketItem> ModelRepository::getTtsModels() {
    try {
        auto data = getModelMarketData();
        if (data && !data->ttsModels.empty()) {
            return processModels(data->ttsModels);
        }
    } catch (const std::exception& e) {
        LOG_DEBUG_TAG("Failed to get TTS models: " + std::string(e.what()), TAG);
    }
    return {};
}

std::vector<ModelMarketItem> ModelRepository::getAsrModels() {
    try {
        auto data = getModelMarketData();
        if (data && !data->asrModels.empty()) {
            return processModels(data->asrModels);
        }
    } catch (const std::exception& e) {
        LOG_DEBUG_TAG("Failed to get ASR models: " + std::string(e.what()), TAG);
    }
    return {};
}

std::optional<std::string> ModelRepository::getModelIdForDownload(const std::string& modelName, const std::string& downloadProvider) {
    if (modelName.empty()) {
        LOG_DEBUG_TAG("Model name is empty", TAG);
        return std::nullopt;
    }
    
    // Find model by name
    auto model = findModelByName(modelName);
    if (!model) {
        LOG_DEBUG_TAG("Model not found: " + modelName, TAG);
        return std::nullopt;
    }
    
    // Check if the specified download provider is available (case-insensitive)
    std::string foundProvider;
    std::string foundRepoPath;
    bool providerFound = false;
    
    for (const auto& [source, repoPath] : model->sources) {
        if (caseInsensitiveEquals(source, downloadProvider)) {
            foundProvider = source;
            foundRepoPath = repoPath;
            providerFound = true;
            break;
        }
    }
    
    if (!providerFound) {
        LOG_DEBUG_TAG("Download provider '" + downloadProvider + "' not available for model: " + modelName, TAG);
        return std::nullopt;
    }
    
    // Create model ID in format: "HuggingFace/taobao-mnn/gpt-oss-20b-MNN"
    std::string modelId = createModelId(foundProvider, foundRepoPath);
    
    LOG_DEBUG_TAG("Created model ID: " + modelId + " for model: " + modelName, TAG);
    
    return modelId;
}

std::string ModelRepository::getModelType(const std::string& modelId) {
    try {
        auto data = getModelMarketData();
        if (!data) {
            return "LLM"; // Default to LLM if no data available
        }
        
        // Check ASR models
        for (const auto& model : data->asrModels) {
            for (const auto& [source, repoPath] : model.sources) {
                std::string testModelId = createModelId(source, repoPath);
                if (testModelId == modelId) {
                    LOG_DEBUG_TAG("Found ASR model: " + modelId, TAG);
                    return "ASR";
                }
            }
        }
        
        // Check TTS models
        for (const auto& model : data->ttsModels) {
            for (const auto& [source, repoPath] : model.sources) {
                std::string testModelId = createModelId(source, repoPath);
                if (testModelId == modelId) {
                    LOG_DEBUG_TAG("Found TTS model: " + modelId, TAG);
                    return "TTS";
                }
            }
        }
        
    } catch (const std::exception& e) {
        LOG_DEBUG_TAG("Failed to determine model type for " + modelId + ": " + std::string(e.what()), TAG);
    }
    
    // Default to LLM
    return "LLM";
}

std::optional<ModelMarketData> ModelRepository::loadFromAssets() {
    try {
        // Try to load from assets directory (similar to Android assets)
        std::string assetsPath = cache_root_path_ + "/assets/" + ASSETS_FILE_NAME;
        
        if (!std::filesystem::exists(assetsPath)) {
            // Try alternative paths
            std::vector<std::string> possiblePaths = {
                cache_root_path_ + "/" + ASSETS_FILE_NAME,
                std::string("./") + ASSETS_FILE_NAME,
                std::string("../assets/") + ASSETS_FILE_NAME,
                std::string("../../assets/") + ASSETS_FILE_NAME
            };
            
            for (const auto& path : possiblePaths) {
                if (std::filesystem::exists(path)) {
                    assetsPath = path;
                    break;
                }
            }
        }
        
        if (!std::filesystem::exists(assetsPath)) {
            LOG_DEBUG_TAG("Assets file not found: " + std::string(ASSETS_FILE_NAME), TAG);
            return std::nullopt;
        }
        
        // Read and parse JSON file
        std::ifstream file(assetsPath);
        if (!file.is_open()) {
            LOG_DEBUG_TAG("Failed to open assets file: " + assetsPath, TAG);
            return std::nullopt;
        }
        
        nlohmann::json jsonData;
        file >> jsonData;
        file.close();
        
        // Parse JSON into ModelMarketData structure
        ModelMarketData data;
        
        // Parse basic fields
        if (jsonData.contains("version")) {
            data.version = jsonData["version"];
        }
        
        if (jsonData.contains("tagTranslations")) {
            for (const auto& [key, value] : jsonData["tagTranslations"].items()) {
                data.tagTranslations[key] = value;
            }
        }
        
        if (jsonData.contains("quickFilterTags")) {
            data.quickFilterTags = jsonData["quickFilterTags"].get<std::vector<std::string>>();
        }
        
        if (jsonData.contains("vendorOrder")) {
            data.vendorOrder = jsonData["vendorOrder"].get<std::vector<std::string>>();
        }
        
        // Parse models
        if (jsonData.contains("models")) {
            for (const auto& modelJson : jsonData["models"]) {
                ModelMarketItem item;
                item.modelName = modelJson["modelName"];
                
                if (modelJson.contains("tags")) {
                    item.tags = modelJson["tags"].get<std::vector<std::string>>();
                }
                
                if (modelJson.contains("categories")) {
                    item.categories = modelJson["categories"].get<std::vector<std::string>>();
                }
                
                if (modelJson.contains("sources")) {
                    for (const auto& [key, value] : modelJson["sources"].items()) {
                        item.sources[key] = value;
                    }
                }
                
                if (modelJson.contains("size_gb")) {
                    item.size_gb = modelJson["size_gb"].get<double>();
                }
                
                if (modelJson.contains("vendor")) {
                    item.vendor = modelJson["vendor"];
                }
                
                if (modelJson.contains("file_size")) {
                    item.file_size = modelJson["file_size"].get<int64_t>();
                }
                
                data.models.push_back(item);
            }
        }
        
        // Parse TTS models
        if (jsonData.contains("tts_models")) {
            for (const auto& modelJson : jsonData["tts_models"]) {
                ModelMarketItem item;
                item.modelName = modelJson["modelName"];
                
                if (modelJson.contains("tags")) {
                    item.tags = modelJson["tags"].get<std::vector<std::string>>();
                }
                
                if (modelJson.contains("categories")) {
                    item.categories = modelJson["categories"].get<std::vector<std::string>>();
                }
                
                if (modelJson.contains("sources")) {
                    for (const auto& [key, value] : modelJson["sources"].items()) {
                        item.sources[key] = value;
                    }
                }
                
                if (modelJson.contains("size_gb")) {
                    item.size_gb = modelJson["size_gb"].get<double>();
                }
                
                if (modelJson.contains("vendor")) {
                    item.vendor = modelJson["vendor"];
                }
                
                if (modelJson.contains("file_size")) {
                    item.file_size = modelJson["file_size"].get<int64_t>();
                }
                
                data.ttsModels.push_back(item);
            }
        }
        
        // Parse ASR models
        if (jsonData.contains("asr_models")) {
            for (const auto& modelJson : jsonData["asr_models"]) {
                ModelMarketItem item;
                item.modelName = modelJson["modelName"];
                
                if (modelJson.contains("tags")) {
                    item.tags = modelJson["tags"].get<std::vector<std::string>>();
                }
                
                if (modelJson.contains("categories")) {
                    item.categories = modelJson["categories"].get<std::vector<std::string>>();
                }
                
                if (modelJson.contains("sources")) {
                    for (const auto& [key, value] : modelJson["sources"].items()) {
                        item.sources[key] = value;
                    }
                }
                
                if (modelJson.contains("size_gb")) {
                    item.size_gb = modelJson["size_gb"].get<double>();
                }
                
                if (modelJson.contains("vendor")) {
                    item.vendor = modelJson["vendor"];
                }
                
                if (modelJson.contains("file_size")) {
                    item.file_size = modelJson["file_size"].get<int64_t>();
                }
                
                data.asrModels.push_back(item);
            }
        }
        
        LOG_DEBUG_TAG("Successfully parsed model market data with " + 
                      std::to_string(data.models.size()) + " models, " + 
                      std::to_string(data.ttsModels.size()) + " TTS models, and " +
                      std::to_string(data.asrModels.size()) + " ASR models", TAG);
        
        return data;
        
    } catch (const std::exception& e) {
        LOG_DEBUG_TAG("Failed to load from assets: " + std::string(e.what()), TAG);
        return std::nullopt;
    }
}

std::optional<ModelMarketItem> ModelRepository::findModelByName(const std::string& modelName) {
    auto data = getModelMarketData();
    if (!data) {
        return std::nullopt;
    }
    
    // Search in regular models
    for (const auto& model : data->models) {
        if (caseInsensitiveEquals(model.modelName, modelName)) {
            return model;
        }
    }
    
    // Search in TTS models
    for (const auto& model : data->ttsModels) {
        if (caseInsensitiveEquals(model.modelName, modelName)) {
            return model;
        }
    }
    
    // Search in ASR models
    for (const auto& model : data->asrModels) {
        if (caseInsensitiveEquals(model.modelName, modelName)) {
            return model;
        }
    }
    
    return std::nullopt;
}

std::vector<ModelMarketItem> ModelRepository::processModels(const std::vector<ModelMarketItem>& models) {
    std::vector<ModelMarketItem> processedModels;
    
    for (const auto& model : models) {
        // Check if current download provider is available (case-insensitive)
        std::string foundProvider;
        std::string foundRepoPath;
        bool providerFound = false;
        
        for (const auto& [source, repoPath] : model.sources) {
            if (caseInsensitiveEquals(source, current_download_provider_)) {
                foundProvider = source;
                foundRepoPath = repoPath;
                providerFound = true;
                break;
            }
        }
        
        if (!providerFound) {
            continue; // Skip models that don't support current provider
        }
        
        // Create a copy and set runtime fields
        ModelMarketItem processedModel = model;
        processedModel.currentSource = foundProvider;
        processedModel.currentRepoPath = foundRepoPath;
        processedModel.modelId = createModelId(foundProvider, foundRepoPath);
        
        processedModels.push_back(processedModel);
    }
    
            LOG_DEBUG_TAG("Processed " + std::to_string(processedModels.size()) + 
                      " models with provider: " + current_download_provider_, TAG);
    
    return processedModels;
}

bool ModelRepository::isVersionLower(const std::string& version1, const std::string& version2) {
    try {
        int v1 = std::stoi(version1);
        int v2 = std::stoi(version2);
        return v1 < v2;
    } catch (const std::exception& e) {
        LOG_DEBUG_TAG("Failed to parse version numbers: " + version1 + ", " + version2, TAG);
        // If parsing fails, treat as equal (use network data)
        return false;
    }
}

std::string ModelRepository::createModelId(const std::string& source, const std::string& repoPath) {
    // Format: "HuggingFace/taobao-mnn/gpt-oss-20b-MNN"
    return source + "/" + repoPath;
}

std::vector<ModelMarketItem> ModelRepository::searchModels(const std::string& keyword) {
    std::vector<ModelMarketItem> searchResults;
    
    if (keyword.empty()) {
        LOG_DEBUG_TAG("Search keyword is empty, returning all LLM models", TAG);
        // Return all LLM models filtered by current source
        return processModels(getModels());
    }
    
    try {
        auto data = getModelMarketData();
        if (!data) {
            LOG_DEBUG_TAG("No model market data available for search", TAG);
            return searchResults;
        }
        
        LOG_DEBUG_TAG("Searching for models with keyword: '" + keyword + "'", TAG);
        LOG_DEBUG_TAG("Current download provider: " + current_download_provider_, TAG);
        
        // Helper function to check if text contains keyword (case-insensitive)
        auto containsKeyword = [&keyword](const std::string& text) -> bool {
            if (text.empty()) return false;
            std::string lowerText = text;
            std::string lowerKeyword = keyword;
            
            // Convert to lowercase for case-insensitive comparison
            std::transform(lowerText.begin(), lowerText.end(), lowerText.begin(), ::tolower);
            std::transform(lowerKeyword.begin(), lowerKeyword.end(), lowerKeyword.begin(), ::tolower);
            
            return lowerText.find(lowerKeyword) != std::string::npos;
        };
        
        // Helper function to check if model supports current source
        auto supportsCurrentSource = [this](const ModelMarketItem& model) -> bool {
            for (const auto& [source, repoPath] : model.sources) {
                if (caseInsensitiveEquals(source, current_download_provider_)) {
                    return true;
                }
            }
            return false;
        };
        
        // Search only in LLM models (not TTS or ASR)
        for (const auto& model : data->models) {
            // First check if model supports current source
            if (!supportsCurrentSource(model)) {
                LOG_DEBUG_TAG("Model '" + model.modelName + "' does not support current source '" + current_download_provider_ + "', skipping", TAG);
                continue;
            }
            
            // Check if model name contains keyword
            bool nameMatches = containsKeyword(model.modelName);
            
            // Check if any tags contain keyword
            bool tagMatches = false;
            for (const auto& tag : model.tags) {
                if (containsKeyword(tag)) {
                    tagMatches = true;
                    break;
                }
            }
            
            // Check if any categories contain keyword
            bool categoryMatches = false;
            for (const auto& category : model.categories) {
                if (containsKeyword(category)) {
                    categoryMatches = true;
                    break;
                }
            }
            
            // Check if vendor contains keyword
            bool vendorMatches = containsKeyword(model.vendor);
            
            // If any field matches, add to results
            if (nameMatches || tagMatches || categoryMatches || vendorMatches) {
                LOG_DEBUG_TAG("Found matching model: " + model.modelName + " (source: " + current_download_provider_ + ")", TAG);
                
                // Create a copy and set runtime fields
                ModelMarketItem resultModel = model;
                
                // Find the matching source and repo path
                for (const auto& [source, repoPath] : model.sources) {
                    if (caseInsensitiveEquals(source, current_download_provider_)) {
                        resultModel.currentSource = source;
                        resultModel.currentRepoPath = repoPath;
                        resultModel.modelId = createModelId(source, repoPath);
                        break;
                    }
                }
                
                searchResults.push_back(resultModel);
            }
        }
        
        LOG_DEBUG_TAG("Search completed. Found " + std::to_string(searchResults.size()) + " matching models", TAG);
        
    } catch (const std::exception& e) {
        LOG_DEBUG_TAG("Error during search: " + std::string(e.what()), TAG);
    }
    
    return searchResults;
}

} // namespace mnncli
