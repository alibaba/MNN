//
//  llm_manager.cpp
//
//  LLM model management and lifecycle implementation
//

#include "llm_manager.hpp"
#include "json.hpp"
#include <fstream>
#include "../../../transformers/llm/engine/include/llm/llm.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>

namespace mnncli {

std::unique_ptr<MNN::Transformer::Llm> LLMManager::CreateLLM(
    const std::string& config_path, 
    bool use_template
) {
    std::unique_ptr<MNN::Transformer::Llm> llm(MNN::Transformer::Llm::createLLM(config_path));
    if (!llm) {
        return nullptr;
    }

    const auto config = nlohmann::json::parse(llm->dump_config());
    auto overrides = nlohmann::json::object();
    // Preserve per-instance cache directories supplied by the runtime config.
    if (config.value("tmp_path", std::string()).empty()) overrides["tmp_path"] = "tmp";
    if (!use_template) overrides["use_template"] = false;
    std::ifstream runtime_file(config_path);
    const auto runtime_config = nlohmann::json::parse(runtime_file, nullptr, false);
    // Model metadata is merged last by LlmConfig; an explicitly supplied runtime template takes precedence here.
    if (runtime_config.contains("jinja") && runtime_config["jinja"].is_object() &&
        runtime_config["jinja"].contains("chat_template")) {
        overrides["jinja"] = runtime_config["jinja"];
    }
    llm->set_config(overrides.dump());
    
    {
        AUTOTIME;
        if (!llm->load()) {
            return nullptr;
        }
    }
    
    if (true) {
        AUTOTIME;
        TuningPrepare(llm.get());
    }
    
    return llm;
}

void LLMManager::PrepareTuning(MNN::Transformer::Llm* llm) {
    TuningPrepare(llm);
}

void LLMManager::TuningPrepare(MNN::Transformer::Llm* llm) {
    llm->tuning(MNN::Transformer::OP_ENCODER_NUMBER, {1, 5, 10, 20, 30, 50, 100});
}

} // namespace mnncli

