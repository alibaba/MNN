#include "model_sources.hpp"
#include "file_utils.hpp"
#include "dl_config.hpp"
//
//  model_info_command_handler.cpp
//
//  Handler for 'model_info' command
//

#include "handlers/model_info_command_handler.hpp"
#include "cli_command_parser.hpp"
#include "cli_command_spec.hpp"
#include "model_manager.hpp"

namespace mnncli {
using namespace mnn::downloader;

std::string ModelInfoCommandHandler::CommandName() const {
    return "model_info";
}

const CommandSpec& ModelInfoCommandHandler::GetSpec() const {
    return mnncli::GetSpec("model_info");
}

int ModelInfoCommandHandler::Handle(const ParsedCommand& cmd) {
    if (cmd.arguments.empty()) {
        return 1;
    }
    std::string model_name = cmd.arguments[0];
    return ModelManager::ShowModelInfo(model_name, cmd.verbose);
}

} // namespace mnncli
