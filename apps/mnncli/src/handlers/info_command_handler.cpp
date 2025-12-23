#include "model_sources.hpp"
#include "file_utils.hpp"
#include "dl_config.hpp"
//
//  info_command_handler.cpp
//
//  Handler for 'info' command
//

#include "handlers/info_command_handler.hpp"
#include "cli_command_parser.hpp"
#include "cli_command_spec.hpp"
#include <iostream>
#include <MNN/AutoTime.hpp>

namespace mnncli {
using namespace mnn::downloader;

std::string InfoCommandHandler::CommandName() const {
    return "info";
}

const CommandSpec& InfoCommandHandler::GetSpec() const {
    return mnncli::GetSpec("info");
}

int InfoCommandHandler::Handle(const ParsedCommand& cmd) {
    std::cout << "MNN CLI Information\n";
    std::cout << "==================\n\n";
    
    std::cout << "Version: 1.0.0\n";
    std::cout << "MNN Framework: Enabled\n";
    std::cout << "LLM Support: Yes\n";
    std::cout << "GPU Support: Available\n";
    std::cout << "\n";
    
    std::cout << "Supported Commands:\n";
    std::cout << "  - list           List local models\n";
    std::cout << "  - search         Search remote models\n";
    std::cout << "  - download       Download a model\n";
    std::cout << "  - delete         Delete a model\n";
    std::cout << "  - model_info     Show model details\n";
    std::cout << "  - run            Run model inference\n";
    std::cout << "  - serve          Start model server\n";
    std::cout << "  - benchmark      Run performance test\n";
    std::cout << "  - config         Manage configuration\n";
    
    return 0;
}

} // namespace mnncli
