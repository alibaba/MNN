#include <catch2/catch_test_macros.hpp>
#include "model_name_utils.hpp"
#include "mnncli_config.hpp"

#include <string>

TEST_CASE("GetFullModelId - ModelScope provider", "[model_name_utils]") {
    mnncli::Config config;
    config.download_provider = "modelscope";
    REQUIRE(mnncli::ModelNameUtils::GetFullModelId("qwen-7b", config) == "ModelScope/MNN/qwen-7b");
}

TEST_CASE("GetFullModelId - HuggingFace provider", "[model_name_utils]") {
    mnncli::Config config;
    config.download_provider = "huggingface";
    REQUIRE(mnncli::ModelNameUtils::GetFullModelId("qwen-7b-chat", config) == "HuggingFace/taobao-mnn/qwen-7b-chat");
}

TEST_CASE("GetFullModelId - Modelers provider", "[model_name_utils]") {
    mnncli::Config config;
    config.download_provider = "modelers";
    REQUIRE(mnncli::ModelNameUtils::GetFullModelId("bert-base", config) == "Modelers/MNN/bert-base");
}

TEST_CASE("GetFullModelId - ModelScope with custom org", "[model_name_utils]") {
    mnncli::Config config;
    config.download_provider = "modelscope";
    REQUIRE(mnncli::ModelNameUtils::GetFullModelId("custom_org/qwen-7b", config) == "ModelScope/custom_org/qwen-7b");
}

TEST_CASE("GetFullModelId - Modelers with custom org", "[model_name_utils]") {
    mnncli::Config config;
    config.download_provider = "modelers";
    REQUIRE(mnncli::ModelNameUtils::GetFullModelId("custom_org/bert-base", config) == "Modelers/custom_org/bert-base");
}

TEST_CASE("GetFullModelId - HuggingFace with custom org", "[model_name_utils]") {
    mnncli::Config config;
    config.download_provider = "huggingface";
    REQUIRE(mnncli::ModelNameUtils::GetFullModelId("custom_org/qwen-7b", config) == "HuggingFace/custom_org/qwen-7b");
}

TEST_CASE("GetFullModelId - MS abbreviation", "[model_name_utils]") {
    mnncli::Config config;
    config.download_provider = "ms";
    REQUIRE(mnncli::ModelNameUtils::GetFullModelId("qwen-7b", config) == "ModelScope/MNN/qwen-7b");
}

TEST_CASE("GetFullModelId - HF abbreviation", "[model_name_utils]") {
    mnncli::Config config;
    config.download_provider = "hf";
    REQUIRE(mnncli::ModelNameUtils::GetFullModelId("qwen-7b", config) == "HuggingFace/taobao-mnn/qwen-7b");
}

TEST_CASE("GetFullModelId - ModelScope already qualified", "[model_name_utils]") {
    mnncli::Config config;
    config.download_provider = "modelscope";
    REQUIRE(mnncli::ModelNameUtils::GetFullModelId("ModelScope/MNN/qwen-7b", config) == "ModelScope/MNN/qwen-7b");
}

TEST_CASE("GetFullModelId - HuggingFace already qualified", "[model_name_utils]") {
    mnncli::Config config;
    config.download_provider = "huggingface";
    REQUIRE(mnncli::ModelNameUtils::GetFullModelId("huggingface/taobao-mnn/qwen-7b-chat", config) == "huggingface/taobao-mnn/qwen-7b-chat");
}

TEST_CASE("GetFullModelId - Modelers already qualified", "[model_name_utils]") {
    mnncli::Config config;
    config.download_provider = "modelers";
    REQUIRE(mnncli::ModelNameUtils::GetFullModelId("modelers/MNN/bert-base", config) == "modelers/MNN/bert-base");
}

TEST_CASE("GetFullModelId - Lowercase provider", "[model_name_utils]") {
    mnncli::Config config;
    config.download_provider = "modelscope";
    REQUIRE(mnncli::ModelNameUtils::GetFullModelId("modelscope/MNN/qwen-7b", config) == "modelscope/MNN/qwen-7b");
}

TEST_CASE("GetFullModelId - Mixed case provider", "[model_name_utils]") {
    mnncli::Config config;
    config.download_provider = "modelscope";
    REQUIRE(mnncli::ModelNameUtils::GetFullModelId("ModelScope/MNN/qwen-7b", config) == "ModelScope/MNN/qwen-7b");
}

TEST_CASE("GetFullModelId - HuggingFace case variations", "[model_name_utils]") {
    mnncli::Config config;
    config.download_provider = "huggingface";
    REQUIRE(mnncli::ModelNameUtils::GetFullModelId("HuggingFace/taobao-mnn/qwen-7b", config) == "HuggingFace/taobao-mnn/qwen-7b");
}

TEST_CASE("GetFullModelId - Empty model name", "[model_name_utils]") {
    mnncli::Config config;
    config.download_provider = "modelscope";
    REQUIRE(mnncli::ModelNameUtils::GetFullModelId("", config) == "");
}

TEST_CASE("GetFullModelId - Unknown provider defaults to ModelScope", "[model_name_utils]") {
    mnncli::Config config;
    config.download_provider = "unknown";
    REQUIRE(mnncli::ModelNameUtils::GetFullModelId("qwen-7b", config) == "ModelScope/MNN/qwen-7b");
}

TEST_CASE("GetFullModelId - Empty provider defaults to ModelScope", "[model_name_utils]") {
    mnncli::Config config;
    config.download_provider = "";
    REQUIRE(mnncli::ModelNameUtils::GetFullModelId("qwen-7b", config) == "ModelScope/MNN/qwen-7b");
}

TEST_CASE("GetFullModelId - Model name with special characters", "[model_name_utils]") {
    mnncli::Config config;
    config.download_provider = "modelscope";
    REQUIRE(mnncli::ModelNameUtils::GetFullModelId("qwen-7b-chat-v1.5", config) == "ModelScope/MNN/qwen-7b-chat-v1.5");
}

