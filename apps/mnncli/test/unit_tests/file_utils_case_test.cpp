#include <catch2/catch_test_macros.hpp>
#include "file_utils.hpp"
#include "mnncli_config.hpp"
#include "cli_config_manager.hpp"

#include <string>
#include <filesystem>

namespace fs = std::filesystem;

// Helper function to create a test directory structure
std::string create_test_cache_dir() {
    std::string test_dir = fs::temp_directory_path() / "mnncli_test_cache";
    fs::create_directories(test_dir);
    return test_dir;
}

void cleanup_test_cache_dir(const std::string& test_dir) {
    fs::remove_all(test_dir);
}

TEST_CASE("GetModelPath - Empty model_id", "[file_utils]") {
    mnncli::Config config;
    config.download_provider = "modelscope";
    
    std::string result = mnncli::FileUtils::GetModelPath("", config);
    REQUIRE(result == "");
}

TEST_CASE("GetModelPath - ModelScope simple name", "[file_utils]") {
    std::string test_dir = create_test_cache_dir();
    
    // Create model directory structure
    fs::create_directories(test_dir + "/ModelScope/MNN/qwen-7b");
    
    // Set up config and ConfigManager
    mnncli::Config config;
    config.download_provider = "modelscope";
    config.cache_dir = test_dir;
    
    // Configure ConfigManager to use test cache directory
    auto& config_mgr = mnncli::ConfigManager::GetInstance();
    config_mgr.SetConfigValue("cache_dir", test_dir);
    
    std::string result = mnncli::FileUtils::GetModelPath("qwen-7b", config);
    
    cleanup_test_cache_dir(test_dir);
    REQUIRE(result == test_dir + "/ModelScope/MNN/qwen-7b");
}

TEST_CASE("GetModelPath - HuggingFace simple name", "[file_utils]") {
    std::string test_dir = create_test_cache_dir();
    
    fs::create_directories(test_dir + "/HuggingFace/taobao-mnn/qwen-7b");
    
    mnncli::Config config;
    config.download_provider = "huggingface";
    config.cache_dir = test_dir;
    
    // Configure ConfigManager to use test cache directory
    auto& config_mgr = mnncli::ConfigManager::GetInstance();
    config_mgr.SetConfigValue("cache_dir", test_dir);
    
    std::string result = mnncli::FileUtils::GetModelPath("qwen-7b", config);
    
    cleanup_test_cache_dir(test_dir);
    REQUIRE(result == test_dir + "/HuggingFace/taobao-mnn/qwen-7b");
}

TEST_CASE("GetModelPath - ModelScope custom org", "[file_utils]") {
    std::string test_dir = create_test_cache_dir();
    
    fs::create_directories(test_dir + "/ModelScope/custom/model");
    
    mnncli::Config config;
    config.download_provider = "modelscope";
    config.cache_dir = test_dir;
    
    // Configure ConfigManager to use test cache directory
    auto& config_mgr = mnncli::ConfigManager::GetInstance();
    config_mgr.SetConfigValue("cache_dir", test_dir);
    
    std::string result = mnncli::FileUtils::GetModelPath("custom/model", config);
    
    cleanup_test_cache_dir(test_dir);
    REQUIRE(result == test_dir + "/ModelScope/custom/model");
}

TEST_CASE("GetModelPath - HuggingFace custom org", "[file_utils]") {
    std::string test_dir = create_test_cache_dir();
    
    fs::create_directories(test_dir + "/HuggingFace/org/model");
    
    mnncli::Config config;
    config.download_provider = "huggingface";
    config.cache_dir = test_dir;
    
    // Configure ConfigManager to use test cache directory
    auto& config_mgr = mnncli::ConfigManager::GetInstance();
    config_mgr.SetConfigValue("cache_dir", test_dir);
    
    std::string result = mnncli::FileUtils::GetModelPath("org/model", config);
    
    cleanup_test_cache_dir(test_dir);
    REQUIRE(result == test_dir + "/HuggingFace/org/model");
}

// TEST_CASE("GetModelPath - Fallback mechanism", "[file_utils]") {
//     std::string test_dir = create_test_cache_dir();
    
//     // Model not in primary provider directory but exists in fallback
//     fs::create_directories(test_dir + "/HuggingFace/MNN/qwen-7b");
    
//     mnncli::Config config;
//     config.download_provider = "modelscope";  // Primary provider
//     config.cache_dir = test_dir;
    
//     std::string result = mnncli::FileUtils::GetModelPath("qwen-7b", config);
    
//     cleanup_test_cache_dir(test_dir);
//     REQUIRE(result == test_dir + "/HuggingFace/MNN/qwen-7b");
// }
