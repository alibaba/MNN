//
//  llm_manager.cpp
//
//  LLM model management and lifecycle implementation
//

#include "llm_manager.hpp"
#include "../../../transformers/llm/engine/include/llm/llm.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>

namespace mnncli {

std::unique_ptr<MNN::Transformer::Llm> LLMManager::CreateLLM(
    const std::string& config_path, 
    bool use_template
) {
    std::unique_ptr<MNN::Transformer::Llm> llm(MNN::Transformer::Llm::createLLM(config_path));
    
    if (use_template) {
        llm->set_config("{\"tmp_path\":\"tmp\"}");
    } else {
        llm->set_config("{\"tmp_path\":\"tmp\",\"use_template\":false}");
    }
    
    {
        AUTOTIME;
        llm->load();
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

