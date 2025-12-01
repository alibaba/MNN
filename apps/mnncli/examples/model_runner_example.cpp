//
//  model_runner_example.cpp
//
//  Created by MNN on 2024/01/01.
//  Example usage of ModelRunner class
//

#include "../include/model_runner.hpp"
#include "../../../transformers/llm/engine/include/llm/llm.hpp"
#include <iostream>
#include <memory>

using namespace MNN::Transformer;

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " config.json [prompt.txt] or -p \"prompt\"" << std::endl;
        std::cout << "Examples:" << std::endl;
        std::cout << "  " << argv[0] << " config.json                    # Interactive chat mode" << std::endl;
        std::cout << "  " << argv[0] << " config.json prompts.txt        # Evaluate prompts from file" << std::endl;
        std::cout << "  " << argv[0] << " config.json -p \"Hello, world!\" # Process single prompt" << std::endl;
#ifdef LLM_SUPPORT_VISION
        std::cout << "  " << argv[0] << " config.json -p \"Describe this video:<video>/path/to/video.mp4</video>\" # Process video" << std::endl;
#endif
        return 0;
    }

    try {
        // Initialize LLM
        std::string config_path = argv[1];
        std::cout << "Loading LLM from config: " << config_path << std::endl;
        
        std::unique_ptr<Llm> llm(Llm::createLLM(config_path));
        llm->set_config("{\"tmp_path\":\"tmp\"}");
        llm->load();
        
        // Create ModelRunner
        ModelRunner runner(llm.get());
        
        // Handle different command line arguments
        if (argc > 2) {
            std::string prompt_arg = argv[2];
            
            if (prompt_arg == "-p") {
                if (argc > 3) {
                    std::string prompt_str = argv[3];
                    std::cout << "Processing prompt: " << prompt_str << std::endl;
                    runner.ProcessPrompt(prompt_str);
                } else {
                    std::cerr << "Error: -p flag requires a prompt string." << std::endl;
                    return 1;
                }
            } else {
                // Treat as prompt file
                std::cout << "Evaluating prompts from file: " << prompt_arg << std::endl;
                runner.EvalFile(prompt_arg);
            }
        } else {
            // Interactive chat mode
            std::cout << "Starting interactive chat mode..." << std::endl;
            runner.InteractiveChat();
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
