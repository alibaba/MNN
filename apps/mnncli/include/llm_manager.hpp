//
//  llm_manager.hpp
//
//  LLM model management and lifecycle
//

#pragma once

#include <string>
#include <memory>

namespace MNN::Transformer {
    class Llm;
}

namespace mnncli {

class LLMManager {
public:
    // Create and load an LLM instance
    static std::unique_ptr<MNN::Transformer::Llm> CreateLLM(
        const std::string& config_path, 
        bool use_template
    );
    
    // Prepare tuning for the model
    static void PrepareTuning(MNN::Transformer::Llm* llm);

private:
    // Tuning configuration
    static void TuningPrepare(MNN::Transformer::Llm* llm);
};

} // namespace mnncli

