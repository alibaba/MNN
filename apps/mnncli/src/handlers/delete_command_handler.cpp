#include "model_sources.hpp"
#include "file_utils.hpp"
#include "dl_config.hpp"
//
//  delete_command_handler.cpp
//
//  Handler for 'delete' command
//

#include "handlers/delete_command_handler.hpp"
#include "cli_command_parser.hpp"
#include "cli_command_spec.hpp"
#include "model_manager.hpp"

namespace mnncli {
using namespace mnn::downloader;

std::string DeleteCommandHandler::CommandName() const {
    return "delete";
}

const CommandSpec& DeleteCommandHandler::GetSpec() const {
    return mnncli::GetSpec("delete");
}

int DeleteCommandHandler::Handle(const ParsedCommand& cmd) {
    if (cmd.arguments.empty()) {
        // This should not happen if parsing is correct, but handle gracefully
        return 1;
    }
    
    std::string model_name = cmd.arguments[0];
    LOG_DEBUG_TAG("DeleteCommandHandler: model_name = " + model_name, "DeleteCommandHandler");
    return ModelManager::DeleteModel(model_name);
}

} // namespace mnncli
