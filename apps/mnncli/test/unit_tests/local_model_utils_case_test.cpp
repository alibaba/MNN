#include <catch2/catch_test_macros.hpp>
#include "local_model_utils.hpp"

#include <string>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

// Helper function to create a test directory structure
std::string create_test_cache_dir() {
    std::string test_dir = (fs::temp_directory_path() / "mnncli_local_model_test").string();
    fs::create_directories(test_dir);
    return test_dir;
}

void cleanup_test_cache_dir(const std::string& test_dir) {
    std::error_code ec;
    fs::remove_all(test_dir, ec);
}

// Helper to create a complete model with .mnncli/.complete marker
void create_complete_model(const std::string& base_dir, const std::string& provider, 
                          const std::string& owner, const std::string& model) {
    fs::path model_path = fs::path(base_dir) / provider / owner / model;
    fs::create_directories(model_path);
    
    // Create .mnncli/.complete marker
    fs::path marker_dir = model_path / ".mnncli";
    fs::create_directories(marker_dir);
    std::ofstream marker_file(marker_dir / ".complete");
    marker_file << "1";
    marker_file.close();
}

// Helper to create an incomplete model (no .mnncli/.complete marker)
void create_incomplete_model(const std::string& base_dir, const std::string& provider,
                            const std::string& owner, const std::string& model) {
    fs::path model_path = fs::path(base_dir) / provider / owner / model;
    fs::create_directories(model_path);
    // No .mnncli/.complete marker
}

TEST_CASE("ListLocalModelsInner - Non-existent provider", "[local_model_utils]") {
    std::string test_dir = create_test_cache_dir();
    
    auto models = mnncli::LocalModelUtils::ListLocalModelsInner("NonExistentProvider", test_dir);
    
    cleanup_test_cache_dir(test_dir);
    REQUIRE(models.empty());
}

TEST_CASE("ListLocalModelsInner - Empty provider directory", "[local_model_utils]") {
    std::string test_dir = create_test_cache_dir();
    
    // Create empty provider directory
    fs::create_directories(fs::path(test_dir) / "HuggingFace");
    
    auto models = mnncli::LocalModelUtils::ListLocalModelsInner("HuggingFace", test_dir);
    
    cleanup_test_cache_dir(test_dir);
    REQUIRE(models.empty());
}

TEST_CASE("ListLocalModelsInner - Single complete model from HuggingFace", "[local_model_utils]") {
    std::string test_dir = create_test_cache_dir();
    
    // Create a complete model
    create_complete_model(test_dir, "HuggingFace", "taobao-mnn", "qwen-7b");
    
    auto models = mnncli::LocalModelUtils::ListLocalModelsInner("HuggingFace", test_dir);
    
    cleanup_test_cache_dir(test_dir);
    REQUIRE(models.size() == 1);
    REQUIRE(models[0] == "HuggingFace/taobao-mnn/qwen-7b");
}

TEST_CASE("ListLocalModelsInner - Single complete model from ModelScope", "[local_model_utils]") {
    std::string test_dir = create_test_cache_dir();
    
    // Create a complete model
    create_complete_model(test_dir, "ModelScope", "MNN", "Qwen2-7B-Instruct");
    
    auto models = mnncli::LocalModelUtils::ListLocalModelsInner("ModelScope", test_dir);
    
    cleanup_test_cache_dir(test_dir);
    REQUIRE(models.size() == 1);
    REQUIRE(models[0] == "ModelScope/MNN/Qwen2-7B-Instruct");
}

TEST_CASE("ListLocalModelsInner - Multiple complete models", "[local_model_utils]") {
    std::string test_dir = create_test_cache_dir();
    
    // Create multiple complete models
    create_complete_model(test_dir, "ModelScope", "MNN", "Qwen2-7B-Instruct");
    create_complete_model(test_dir, "ModelScope", "MNN", "Llama-3-8B");
    create_complete_model(test_dir, "ModelScope", "OpenAI", "gpt-model");
    
    auto models = mnncli::LocalModelUtils::ListLocalModelsInner("ModelScope", test_dir);
    
    cleanup_test_cache_dir(test_dir);
    REQUIRE(models.size() == 3);
    // Results should be sorted
    REQUIRE(models[0] == "ModelScope/MNN/Llama-3-8B");
    REQUIRE(models[1] == "ModelScope/MNN/Qwen2-7B-Instruct");
    REQUIRE(models[2] == "ModelScope/OpenAI/gpt-model");
}

TEST_CASE("ListLocalModelsInner - Filter out incomplete models", "[local_model_utils]") {
    std::string test_dir = create_test_cache_dir();
    
    // Create one complete and one incomplete model
    create_complete_model(test_dir, "HuggingFace", "taobao-mnn", "complete-model");
    create_incomplete_model(test_dir, "HuggingFace", "taobao-mnn", "incomplete-model");
    
    auto models = mnncli::LocalModelUtils::ListLocalModelsInner("HuggingFace", test_dir);
    
    cleanup_test_cache_dir(test_dir);
    // Should only return the complete model
    REQUIRE(models.size() == 1);
    REQUIRE(models[0] == "HuggingFace/taobao-mnn/complete-model");
}

TEST_CASE("ListLocalModelsInner - Only incomplete models", "[local_model_utils]") {
    std::string test_dir = create_test_cache_dir();
    
    // Create only incomplete models
    create_incomplete_model(test_dir, "ModelScope", "MNN", "incomplete-1");
    create_incomplete_model(test_dir, "ModelScope", "MNN", "incomplete-2");
    
    auto models = mnncli::LocalModelUtils::ListLocalModelsInner("ModelScope", test_dir);
    
    cleanup_test_cache_dir(test_dir);
    // Should return empty since none are complete
    REQUIRE(models.empty());
}

TEST_CASE("ListLocalModelsInner - Skip hidden directories", "[local_model_utils]") {
    std::string test_dir = create_test_cache_dir();
    
    // Create models including hidden directories
    create_complete_model(test_dir, "HuggingFace", "taobao-mnn", "visible-model");
    create_complete_model(test_dir, "HuggingFace", ".hidden-owner", "model");
    create_complete_model(test_dir, "HuggingFace", "taobao-mnn", ".hidden-model");
    
    auto models = mnncli::LocalModelUtils::ListLocalModelsInner("HuggingFace", test_dir);
    
    cleanup_test_cache_dir(test_dir);
    // Should only return the visible model
    REQUIRE(models.size() == 1);
    REQUIRE(models[0] == "HuggingFace/taobao-mnn/visible-model");
}

TEST_CASE("ListLocalModelsInner - Multiple providers", "[local_model_utils]") {
    std::string test_dir = create_test_cache_dir();
    
    // Create models in different providers
    create_complete_model(test_dir, "HuggingFace", "taobao-mnn", "hf-model");
    create_complete_model(test_dir, "ModelScope", "MNN", "ms-model");
    create_complete_model(test_dir, "Modelers", "MNN", "ml-model");
    
    auto hf_models = mnncli::LocalModelUtils::ListLocalModelsInner("HuggingFace", test_dir);
    auto ms_models = mnncli::LocalModelUtils::ListLocalModelsInner("ModelScope", test_dir);
    auto ml_models = mnncli::LocalModelUtils::ListLocalModelsInner("Modelers", test_dir);
    
    cleanup_test_cache_dir(test_dir);
    
    REQUIRE(hf_models.size() == 1);
    REQUIRE(hf_models[0] == "HuggingFace/taobao-mnn/hf-model");
    
    REQUIRE(ms_models.size() == 1);
    REQUIRE(ms_models[0] == "ModelScope/MNN/ms-model");
    
    REQUIRE(ml_models.size() == 1);
    REQUIRE(ml_models[0] == "Modelers/MNN/ml-model");
}

TEST_CASE("ListLocalModelsInner - Models sorted alphabetically", "[local_model_utils]") {
    std::string test_dir = create_test_cache_dir();
    
    // Create models in non-alphabetical order
    create_complete_model(test_dir, "ModelScope", "MNN", "zebra-model");
    create_complete_model(test_dir, "ModelScope", "MNN", "alpha-model");
    create_complete_model(test_dir, "ModelScope", "MNN", "beta-model");
    
    auto models = mnncli::LocalModelUtils::ListLocalModelsInner("ModelScope", test_dir);
    
    cleanup_test_cache_dir(test_dir);
    
    REQUIRE(models.size() == 3);
    // Should be sorted
    REQUIRE(models[0] == "ModelScope/MNN/alpha-model");
    REQUIRE(models[1] == "ModelScope/MNN/beta-model");
    REQUIRE(models[2] == "ModelScope/MNN/zebra-model");
}

TEST_CASE("ListLocalModelsInner - Mixed complete and incomplete across owners", "[local_model_utils]") {
    std::string test_dir = create_test_cache_dir();
    
    // Create models from different owners, some complete, some not
    create_complete_model(test_dir, "HuggingFace", "owner1", "model-a");
    create_incomplete_model(test_dir, "HuggingFace", "owner1", "model-b");
    create_complete_model(test_dir, "HuggingFace", "owner2", "model-c");
    create_incomplete_model(test_dir, "HuggingFace", "owner3", "model-d");
    
    auto models = mnncli::LocalModelUtils::ListLocalModelsInner("HuggingFace", test_dir);
    
    cleanup_test_cache_dir(test_dir);
    
    REQUIRE(models.size() == 2);
    REQUIRE(models[0] == "HuggingFace/owner1/model-a");
    REQUIRE(models[1] == "HuggingFace/owner2/model-c");
}

