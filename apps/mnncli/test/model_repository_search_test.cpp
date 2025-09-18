//
// Created by AI Assistant on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#include "../include/model_repository.hpp"
#include <iostream>
#include <cassert>
#include <string>
#include <vector>

using namespace mnncli;

// Test helper function
void assertEqual(const std::string& actual, const std::string& expected, const std::string& testName) {
    if (actual != expected) {
        std::cerr << "❌ Test failed: " << testName << std::endl;
        std::cerr << "   Expected: '" << expected << "'" << std::endl;
        std::cerr << "   Actual:   '" << actual << "'" << std::endl;
        assert(false);
    } else {
        std::cout << "✅ " << testName << " passed" << std::endl;
    }
}

void assertEqual(size_t actual, size_t expected, const std::string& testName) {
    if (actual != expected) {
        std::cerr << "❌ Test failed: " << testName << std::endl;
        std::cerr << "   Expected: " << expected << std::endl;
        std::cerr << "   Actual:   " << actual << std::endl;
        assert(false);
    } else {
        std::cout << "✅ " << testName << " passed" << std::endl;
    }
}

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

// Test search with empty keyword (should return all LLM models for current source)
void testSearchEmptyKeyword() {
    std::cout << "\n=== Testing Search with Empty Keyword ===" << std::endl;
    
    auto& repo = ModelRepository::getInstance("./test_assets");
    
    // Test with HuggingFace provider
    repo.setDownloadProvider("HuggingFace");
    auto results = repo.searchModels("");
    
    std::cout << "Found " << results.size() << " models for HuggingFace provider" << std::endl;
    
    // Verify all returned models support HuggingFace
    for (const auto& model : results) {
        assertTrue(model.currentSource == "HuggingFace", 
                  "Model " + model.modelName + " should have HuggingFace as current source");
        assertFalse(model.modelId.empty(), 
                   "Model " + model.modelName + " should have a valid model ID");
    }
    
    // Test with ModelScope provider
    repo.setDownloadProvider("ModelScope");
    results = repo.searchModels("");
    
    std::cout << "Found " << results.size() << " models for ModelScope provider" << std::endl;
    
    // Verify all returned models support ModelScope
    for (const auto& model : results) {
        assertTrue(model.currentSource == "ModelScope", 
                  "Model " + model.modelName + " should have ModelScope as current source");
        assertFalse(model.modelId.empty(), 
                   "Model " + model.modelName + " should have a valid model ID");
    }
}

// Test search with specific keyword
void testSearchWithKeyword() {
    std::cout << "\n=== Testing Search with Specific Keyword ===" << std::endl;
    
    auto& repo = ModelRepository::getInstance("./test_assets");
    repo.setDownloadProvider("HuggingFace");
    
    // Search for models containing "gpt"
    auto results = repo.searchModels("gpt");
    std::cout << "Found " << results.size() << " models containing 'gpt'" << std::endl;
    
    // Verify all results contain "gpt" in some field and support HuggingFace
    for (const auto& model : results) {
        bool containsGpt = false;
        
        // Check model name
        std::string lowerName = model.modelName;
        std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), ::tolower);
        if (lowerName.find("gpt") != std::string::npos) {
            containsGpt = true;
        }
        
        // Check tags
        for (const auto& tag : model.tags) {
            std::string lowerTag = tag;
            std::transform(lowerTag.begin(), lowerTag.end(), lowerTag.begin(), ::tolower);
            if (lowerTag.find("gpt") != std::string::npos) {
                containsGpt = true;
                break;
            }
        }
        
        // Check categories
        for (const auto& category : model.categories) {
            std::string lowerCategory = category;
            std::transform(lowerCategory.begin(), lowerCategory.end(), lowerCategory.begin(), ::tolower);
            if (lowerCategory.find("gpt") != std::string::npos) {
                containsGpt = true;
                break;
            }
        }
        
        // Check vendor
        std::string lowerVendor = model.vendor;
        std::transform(lowerVendor.begin(), lowerVendor.end(), lowerVendor.begin(), ::tolower);
        if (lowerVendor.find("gpt") != std::string::npos) {
            containsGpt = true;
        }
        
        assertTrue(containsGpt, "Model " + model.modelName + " should contain 'gpt' in some field");
        assertTrue(model.currentSource == "HuggingFace", 
                  "Model " + model.modelName + " should support HuggingFace");
    }
}

// Test source filtering - models without current source should be excluded
void testSourceFiltering() {
    std::cout << "\n=== Testing Source Filtering ===" << std::endl;
    
    auto& repo = ModelRepository::getInstance("./test_assets");
    
    // Test with HuggingFace provider
    repo.setDownloadProvider("HuggingFace");
    auto results = repo.searchModels("");
    
    std::cout << "Found " << results.size() << " models for HuggingFace provider" << std::endl;
    
    // Verify no models with only ModelScope sources are included
    for (const auto& model : results) {
        bool hasHuggingFace = false;
        for (const auto& [source, repoPath] : model.sources) {
            if (source == "HuggingFace") {
                hasHuggingFace = true;
                break;
            }
        }
        assertTrue(hasHuggingFace, 
                  "Model " + model.modelName + " should have HuggingFace in sources");
    }
    
    // Test with ModelScope provider
    repo.setDownloadProvider("ModelScope");
    results = repo.searchModels("");
    
    std::cout << "Found " << results.size() << " models for ModelScope provider" << std::endl;
    
    // Verify no models with only HuggingFace sources are included
    for (const auto& model : results) {
        bool hasModelScope = false;
        for (const auto& [source, repoPath] : model.sources) {
            if (source == "ModelScope") {
                hasModelScope = true;
                break;
            }
        }
        assertTrue(hasModelScope, 
                  "Model " + model.modelName + " should have ModelScope in sources");
    }
}

// Test case-insensitive search
void testCaseInsensitiveSearch() {
    std::cout << "\n=== Testing Case-Insensitive Search ===" << std::endl;
    
    auto& repo = ModelRepository::getInstance("./test_assets");
    repo.setDownloadProvider("HuggingFace");
    
    // Search with different case variations
    auto results1 = repo.searchModels("gpt");
    auto results2 = repo.searchModels("GPT");
    auto results3 = repo.searchModels("Gpt");
    
    std::cout << "Results for 'gpt': " << results1.size() << std::endl;
    std::cout << "Results for 'GPT': " << results2.size() << std::endl;
    std::cout << "Results for 'Gpt': " << results3.size() << std::endl;
    
    // All searches should return the same number of results
    assertEqual(results1.size(), results2.size(), "Case-insensitive search should return same results for 'gpt' and 'GPT'");
    assertEqual(results1.size(), results3.size(), "Case-insensitive search should return same results for 'gpt' and 'Gpt'");
}

// Test search with non-existent keyword
void testSearchNonExistentKeyword() {
    std::cout << "\n=== Testing Search with Non-Existent Keyword ===" << std::endl;
    
    auto& repo = ModelRepository::getInstance("./test_assets");
    repo.setDownloadProvider("HuggingFace");
    
    // Search for a keyword that shouldn't exist
    auto results = repo.searchModels("nonexistentkeyword12345");
    
    std::cout << "Results for non-existent keyword: " << results.size() << std::endl;
    
    // Should return no results
    assertEqual(results.size(), 0, "Search with non-existent keyword should return no results");
}

// Test search with special characters
void testSearchWithSpecialCharacters() {
    std::cout << "\n=== Testing Search with Special Characters ===" << std::endl;
    
    auto& repo = ModelRepository::getInstance("./test_assets");
    repo.setDownloadProvider("HuggingFace");
    
    // Search with special characters
    auto results = repo.searchModels("2.5");
    
    std::cout << "Results for '2.5': " << results.size() << std::endl;
    
    // Should handle special characters gracefully
    assertTrue(results.size() >= 0, "Search with special characters should not crash");
}

int main() {
    std::cout << "🚀 Starting ModelRepository Search Tests" << std::endl;
    
    try {
        testSearchEmptyKeyword();
        testSearchWithKeyword();
        testSourceFiltering();
        testCaseInsensitiveSearch();
        testSearchNonExistentKeyword();
        testSearchWithSpecialCharacters();
        
        std::cout << "\n🎉 All tests passed successfully!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Test suite failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "\n❌ Test suite failed with unknown exception" << std::endl;
        return 1;
    }
}
